"""
Compute evaluation metrics for swing prediction and zone OCR.
"""


def compute_metrics(examples, results):
    """
    Compute Bat Overall, Bat In-Zone, and Bat Out-Zone accuracy.

    Parameters
    ----
    examples : list[dict]
        Per-pitch examples with ground truth (swing, in_zone).
    results : list[dict]
        Inference results with prediction (bool or None).

    Returns
    ----
    metrics : dict
        Keys: bat_overall_acc, bat_overall_n, bat_iz_acc, bat_iz_n,
        bat_oz_acc, bat_oz_n, unparseable_n, total_n, avg_latency_ms.
    """
    overall_correct = 0
    overall_total = 0
    iz_correct = 0
    iz_total = 0
    oz_correct = 0
    oz_total = 0
    unparseable = 0
    latencies = []

    for ex, res in zip(examples, results):
        pred = res["prediction"]
        latencies.append(res["latency_ms"])

        if pred is None:
            unparseable += 1
            continue

        gt = ex["swing"]
        correct = pred == gt
        overall_correct += int(correct)
        overall_total += 1

        if ex["in_zone"]:
            iz_correct += int(correct)
            iz_total += 1
        else:
            oz_correct += int(correct)
            oz_total += 1

    metrics = dict()
    metrics["bat_overall_acc"] = overall_correct / overall_total if overall_total > 0 else 0.0
    metrics["bat_overall_n"] = overall_total
    metrics["bat_iz_acc"] = iz_correct / iz_total if iz_total > 0 else 0.0
    metrics["bat_iz_n"] = iz_total
    metrics["bat_oz_acc"] = oz_correct / oz_total if oz_total > 0 else 0.0
    metrics["bat_oz_n"] = oz_total
    metrics["unparseable_n"] = unparseable
    metrics["total_n"] = len(examples)
    metrics["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0
    return metrics


def print_metrics(metrics):
    """
    Pretty-print evaluation metrics.

    Parameters
    ----
    metrics : dict
        Output of compute_metrics().
    """
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Bat Overall:   {metrics['bat_overall_acc']:.3f}  (n={metrics['bat_overall_n']})")
    print(f"  Bat In-Zone:   {metrics['bat_iz_acc']:.3f}  (n={metrics['bat_iz_n']})")
    print(f"  Bat Out-Zone:  {metrics['bat_oz_acc']:.3f}  (n={metrics['bat_oz_n']})")
    print(f"  Unparseable:   {metrics['unparseable_n']}")
    print(f"  Total:         {metrics['total_n']}")
    print(f"  Avg Latency:   {metrics['avg_latency_ms']:.1f} ms/example")
    print("=" * 50)


def compute_zone_metrics(examples, results):
    """
    Compute zone OCR accuracy.

    Parameters
    ----
    examples : list[dict]
        Per-pitch examples with ground truth zone.
    results : list[dict]
        Inference results with prediction (int or None).

    Returns
    ----
    metrics : dict
        Keys: zone_overall_acc, zone_iz_acc, zone_oz_acc, counts, unparseable_n.
    """
    overall_correct = 0
    overall_total = 0
    iz_correct = 0
    iz_total = 0
    oz_correct = 0
    oz_total = 0
    unparseable = 0
    latencies = []

    for ex, res in zip(examples, results):
        pred = res["prediction"]
        latencies.append(res["latency_ms"])

        if pred is None:
            unparseable += 1
            continue

        gt = ex["zone"]
        correct = pred == gt
        overall_correct += int(correct)
        overall_total += 1

        if ex["in_zone"]:
            iz_correct += int(correct)
            iz_total += 1
        else:
            oz_correct += int(correct)
            oz_total += 1

    metrics = dict()
    metrics["zone_overall_acc"] = overall_correct / overall_total if overall_total > 0 else 0.0
    metrics["zone_overall_n"] = overall_total
    metrics["zone_iz_acc"] = iz_correct / iz_total if iz_total > 0 else 0.0
    metrics["zone_iz_n"] = iz_total
    metrics["zone_oz_acc"] = oz_correct / oz_total if oz_total > 0 else 0.0
    metrics["zone_oz_n"] = oz_total
    metrics["unparseable_n"] = unparseable
    metrics["total_n"] = len(examples)
    metrics["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0
    return metrics


def print_zone_metrics(metrics):
    """
    Pretty-print zone OCR metrics.

    Parameters
    ----
    metrics : dict
        Output of compute_zone_metrics().
    """
    print("=" * 50)
    print("Zone OCR Results")
    print("=" * 50)
    print(f"  Zone Overall:  {metrics['zone_overall_acc']:.3f}  (n={metrics['zone_overall_n']})")
    print(f"  Zone In-Zone:  {metrics['zone_iz_acc']:.3f}  (n={metrics['zone_iz_n']})")
    print(f"  Zone Out-Zone: {metrics['zone_oz_acc']:.3f}  (n={metrics['zone_oz_n']})")
    print(f"  Unparseable:   {metrics['unparseable_n']}")
    print(f"  Total:         {metrics['total_n']}")
    print(f"  Avg Latency:   {metrics['avg_latency_ms']:.1f} ms/example")
