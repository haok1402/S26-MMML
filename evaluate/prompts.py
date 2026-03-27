"""
Prompt formatting for VLM swing prediction evaluation.
"""

import base64


ZERO_SHOT_TEMPLATE = """The image shows a baseball pitch trajectory with front (catcher view), side, and plate views including the strike zone.

{context}

Will the batter swing at this pitch? Answer with just Yes or No."""

NAIVE_REASONING_TEMPLATE = """The image shows a baseball pitch trajectory with front (catcher view), side, and plate views including the strike zone.

{context}

Analyze the pitch location relative to the strike zone, the pitch movement, and the game situation. Think step by step, then conclude with your final answer on a new line: "Answer: Yes" or "Answer: No"."""

STRUCTURED_REASONING_TEMPLATE = """The image shows a baseball pitch trajectory with front (catcher view), side, and plate views including the strike zone.

{context}

Analyze this pitch by following these steps:
1. Pitch location: Based on the trajectory image, is the ball in or out of the strike zone?
2. Pitch movement: What type of pitch does the trajectory suggest (fastball, breaking ball, offspeed)?
3. Count context: Given the current situation, does the count favor the pitcher or the batter?
4. Batter tendency: Is the batter likely to be aggressive (swinging) or patient (taking) in this situation?
5. Prediction: Considering all of the above, will the batter swing?

Provide your analysis for each step, then conclude with your final answer on a new line: "Answer: Yes" or "Answer: No"."""

HISTORY_ZERO_SHOT_TEMPLATE = """The image shows a baseball pitch trajectory with front (catcher view), side, and plate views including the strike zone.

{context}

{history}

Will the batter swing at this pitch? Answer with just Yes or No."""


ZONE_OCR_TEMPLATE = """The image shows a baseball pitch trajectory with front (catcher view), side, and plate views including the strike zone.

The strike zone is divided into a 3x3 grid (zones 1-9, numbered left-to-right, top-to-bottom). Pitches outside the strike zone are labeled as zone 11 (above), 12 (below), 13 (left), or 14 (right).

{context}

Based on the pitch trajectory image, which zone did the ball arrive in? Answer with just the zone number."""


def format_messages(example, strategy="zero-shot"):
    """
    Build OpenAI-style chat messages for a single pitch example.

    Parameters
    ----
    example : dict
        A pitch example from data.load_examples().
    strategy : str
        One of "zero-shot", "structured-reasoning", "3-history",
        "3-history-structured-reasoning", "zone-ocr".

    Returns
    ----
    messages : list[dict]
        Chat messages compatible with vllm's chat API.
    """
    templates = dict()
    templates["zero-shot"] = ZERO_SHOT_TEMPLATE
    templates["naive-reasoning"] = NAIVE_REASONING_TEMPLATE
    templates["structured-reasoning"] = STRUCTURED_REASONING_TEMPLATE
    templates["3-history"] = HISTORY_ZERO_SHOT_TEMPLATE
    templates["zone-ocr"] = ZONE_OCR_TEMPLATE

    if strategy not in templates:
        raise ValueError(f"Unknown strategy: {strategy}")
    template = templates[strategy]

    fmt_kwargs = dict()
    fmt_kwargs["context"] = example["prompt_context"]

    if strategy.startswith("3-history"):
        fmt_kwargs["history"] = format_history(example["at_bat_history"], window=3)

    text_content = template.format(**fmt_kwargs)
    image_url = image_to_data_url(example["image_bytes"])

    content = []
    content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": text_content})

    messages = [{"role": "user", "content": content}]
    return messages


def format_history(at_bat_history, window=3):
    """
    Format prior pitches in the at-bat as a text block.

    Parameters
    ----
    at_bat_history : list[dict]
        Prior pitches in this at-bat, each with zone, in_zone, swing.
    window : int
        Maximum number of prior pitches to include.

    Returns
    ----
    history_text : str
        Formatted history block, or note if no prior pitches.
    """
    recent = at_bat_history[-window:]
    if not recent:
        return "Previous pitches in this at-bat: None (first pitch)."

    lines = ["Previous pitches in this at-bat:"]
    for i, p in enumerate(recent, 1):
        swing_str = "Yes" if p["swing"] else "No"
        zone_str = "in zone" if p["in_zone"] else "out of zone"
        lines.append(f"  - Pitch {i}: Zone {p['zone']} ({zone_str}), Batter swung: {swing_str}")
    return "\n".join(lines)


def image_to_data_url(image_bytes):
    """
    Convert raw PNG bytes to a data URL.

    Parameters
    ----
    image_bytes : bytes
        Raw PNG file content.

    Returns
    ----
    url : str
        Base64-encoded data URL.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"
