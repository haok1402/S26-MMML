"""
Load evaluation.parquet into per-pitch examples for VLM evaluation.
"""

from collections import defaultdict

import pyarrow.parquet as pq


def load_examples(parquet_path):
    """
    Load evaluation.parquet and return a flat list of per-pitch example dicts.

    Parameters
    ----
    parquet_path : str or Path
        Path to evaluation.parquet.

    Returns
    ----
    examples : list[dict]
        Each dict has keys: filename, game_date, sequence, atbat_pitch_number,
        pitcher, batter, in_zone, zone, swing, image_bytes, prompt_context,
        at_bat_history.
    """
    table = pq.read_table(str(parquet_path))

    examples = []
    for i in range(len(table)):
        ex = dict()
        ex["filename"] = table.column("filename")[i].as_py()
        ex["game_date"] = table.column("game_date")[i].as_py()
        ex["sequence"] = table.column("sequence")[i].as_py()
        ex["atbat_pitch_number"] = table.column("atbat_pitch_number")[i].as_py()
        ex["pitcher"] = table.column("pitcher")[i].as_py()
        ex["batter"] = table.column("batter")[i].as_py()
        ex["in_zone"] = table.column("in_zone")[i].as_py()
        ex["zone"] = table.column("zone")[i].as_py()
        ex["swing"] = table.column("swing")[i].as_py()
        ex["image_bytes"] = table.column("image")[i].as_py()
        examples.append(ex)

    build_atbat_history(examples)
    build_prompt_context(examples)
    return examples


def build_atbat_history(examples):
    """
    Populate at_bat_history for each example by grouping pitches into at-bats.

    Parameters
    ----
    examples : list[dict]
        Mutated in place. Each dict gets an at_bat_history key.
    """
    groups = defaultdict(list)
    for i, ex in enumerate(examples):
        key = (ex["game_date"], ex["pitcher"], ex["batter"])
        groups[key].append(i)

    for indices in groups.values():
        indices.sort(key=lambda i: examples[i]["sequence"])

        current_atbat = []
        for i in indices:
            ex = examples[i]
            if ex["atbat_pitch_number"] == 1:
                current_atbat = []

            ex["at_bat_history"] = list(current_atbat)

            prior = dict()
            prior["zone"] = ex["zone"]
            prior["in_zone"] = ex["in_zone"]
            prior["swing"] = ex["swing"]
            current_atbat.append(prior)


def build_prompt_context(examples):
    """
    Build a text prompt context for each example from available fields.

    Parameters
    ----
    examples : list[dict]
        Mutated in place. Each dict gets a prompt_context key.
    """
    for ex in examples:
        lines = []
        lines.append(f"Pitcher: {ex['pitcher']}")
        lines.append(f"Batter: {ex['batter']}")
        ex["prompt_context"] = "\n".join(lines)
