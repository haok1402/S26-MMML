"""
vLLM inference wrapper for VLM swing prediction.
"""

import math
import re
import time

import torch
from vllm import LLM, SamplingParams

from evaluate.prompts import format_messages

MODEL_REGISTRY = dict()
MODEL_REGISTRY["qwen3-vl-8b"] = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REGISTRY["qwen3-vl-32b"] = "Qwen/Qwen3-VL-32B-Instruct"


def run_inference(examples, model="qwen3-vl-8b", strategy="zero-shot"):
    """
    Run batched VLM inference over all examples.

    Parameters
    ----
    examples : list[dict]
        Per-pitch examples from data.load_examples().
    model : str
        Model key from MODEL_REGISTRY or a HuggingFace model path.
    strategy : str
        Prompting strategy: "zero-shot" or "reasoning".

    Returns
    ----
    results : list[dict]
        Each dict has keys: raw_output, prediction (bool or None), latency_ms,
        confidence (dict with p_yes, p_no, predicted_prob or None).
    """
    model_path = MODEL_REGISTRY.get(model, model)

    llm_kwargs = dict()
    llm_kwargs["model"] = model_path
    llm_kwargs["trust_remote_code"] = True
    llm_kwargs["tensor_parallel_size"] = torch.cuda.device_count()
    llm_kwargs["gpu_memory_utilization"] = 0.9
    if torch.cuda.device_count() > 1:
        llm_kwargs["compilation_config"] = {"pass_config": {"fuse_allreduce_rms": False}}
    llm = LLM(**llm_kwargs)

    sp_kwargs = dict()
    sp_kwargs["max_tokens"] = 4096
    sp_kwargs["temperature"] = 0.0
    sp_kwargs["logprobs"] = 20
    sampling_params = SamplingParams(**sp_kwargs)

    all_messages = [format_messages(ex, strategy) for ex in examples]

    start = time.perf_counter()
    outputs = llm.chat(all_messages, sampling_params=sampling_params)
    total_ms = (time.perf_counter() - start) * 1000
    per_example_ms = total_ms / len(examples) if examples else 0

    results = []
    for output in outputs:
        gen = output.outputs[0]
        result = dict()
        result["raw_output"] = gen.text
        if strategy == "zone-ocr":
            result["prediction"] = parse_zone(gen.text)
        else:
            result["prediction"] = parse_prediction(gen.text, strategy)
        result["latency_ms"] = per_example_ms
        result["confidence"] = extract_confidence(gen.logprobs, result["prediction"], strategy)
        results.append(result)

    return results


def extract_confidence(logprobs, prediction, strategy):
    """
    Extract P(Yes) and P(No) from token logprobs.

    For zero-shot, uses the first token. For reasoning, scans for the last
    token that matches Yes/No.

    Parameters
    ----
    logprobs : list
        Per-token logprobs from vllm output.
    prediction : bool or None
        The parsed prediction.
    strategy : str
        Prompting strategy.

    Returns
    ----
    confidence : dict
        Keys: p_yes, p_no, predicted_prob (prob of the chosen answer).
    """
    if logprobs is None or prediction is None:
        return None

    yes_tokens = {"yes", "Yes", "YES", " Yes", " yes"}
    no_tokens = {"no", "No", "NO", " No", " no"}
    all_tokens = yes_tokens | no_tokens

    # For zero-shot / 3-history, look at the first generated token
    if strategy in ("zero-shot", "3-history"):
        token_logprobs = logprobs[0] if logprobs else None
    else:
        # For reasoning, scan backwards for the last token position
        # where the *generated* token (highest prob) is a Yes/No variant
        token_logprobs = None
        for lp in reversed(logprobs):
            if lp is None:
                continue
            # The generated token is the one with the highest logprob
            best = max(lp.values(), key=lambda v: v.logprob)
            if best.decoded_token.strip().lower() in ("yes", "no"):
                token_logprobs = lp
                break

    if token_logprobs is None:
        return None

    p_yes = 0.0
    p_no = 0.0
    for token_id, logprob_obj in token_logprobs.items():
        token_str = logprob_obj.decoded_token
        prob = math.exp(logprob_obj.logprob)
        if token_str in yes_tokens:
            p_yes += prob
        elif token_str in no_tokens:
            p_no += prob

    confidence = dict()
    confidence["p_yes"] = round(p_yes, 6)
    confidence["p_no"] = round(p_no, 6)
    return confidence


def parse_zone(raw_output):
    """
    Extract zone number from model output.

    Parameters
    ----
    raw_output : str
        Raw model output text.

    Returns
    ----
    zone : int or None
        Predicted zone number, or None if unparseable.
    """
    text = raw_output.strip()
    valid_zones = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14}

    matches = re.findall(r"\b(\d{1,2})\b", text)
    for m in matches:
        val = int(m)
        if val in valid_zones:
            return val

    return None


def parse_prediction(raw_output, strategy="zero-shot"):
    """
    Extract Yes/No prediction from model output.

    Parameters
    ----
    raw_output : str
        Raw model output text.
    strategy : str
        Prompting strategy (affects parsing logic).

    Returns
    ----
    prediction : bool or None
        True for Yes (swing), False for No, None if unparseable.
    """
    text = raw_output.strip()

    # Try explicit "Answer: Yes/No" pattern first
    m = re.search(r"Answer:\s*(Yes|No)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "yes"

    # Fallback: find the last standalone Yes/No in the text
    matches = re.findall(r"\b(yes|no)\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].lower() == "yes"

    return None
