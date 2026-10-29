"""Per-class TP / FP / FN breakdown at a chosen match distance.

Place next to detection_failures.py and main.py, then in main.py:

    from per_class_report import per_class_breakdown
    report = per_class_breakdown(ground_truth, predictions, loader.nusc,
                                 distance_threshold=1.0, score_threshold=0.3)
"""
from collections import Counter

from detection_failures import CLASS_NAMES, analyze_detection_failures


def _fmt(value):
    return "  -  " if value is None else f"{value:.2f}"


def _print_reason_table(title, per_class_reasons, class_rows):
    reasons = sorted({r for c in per_class_reasons.values() for r in c})
    if not reasons:
        print(f"\n{title}: none")
        return
    print(f"\n{title}")
    header = f"{'class':<22}" + "".join(f"{r[:34]:>36}" for r in reasons)
    print(header)
    print("-" * len(header))
    for row in class_rows:
        counts = per_class_reasons.get(row["class"], {})
        print(f"{row['class']:<22}" + "".join(f"{counts.get(r, 0):>36}" for r in reasons))


def per_class_breakdown(ground_truth, predictions, nusc, distance_threshold=1.0,
                        score_threshold=0.3, print_table=True, **kwargs):
    """Run the failure analysis at one match radius and summarize per class.

    Args:
        distance_threshold: match radius in meters (1.0, 2.0, 4.0, or a
            {class_name: meters} dict).
        score_threshold: minimum score for a prediction to count (float or dict).
        print_table: print the tables as well as returning them.
        **kwargs: forwarded to analyze_detection_failures (e.g. diagnostic_radius,
            apply_devkit_filters).

    Returns:
        {"distance_threshold", "score_threshold", "rows", "fn_reasons",
         "fp_reasons", "results"}; rows is a list of per-class dicts and
        results is the full output of analyze_detection_failures.
    """
    results = analyze_detection_failures(
        ground_truth, predictions, nusc,
        distance_threshold=distance_threshold,
        score_threshold=score_threshold, **kwargs)

    fn_reasons, fp_reasons = {}, {}
    for f in results["false_negatives"]:
        fn_reasons.setdefault(f["ground_truth_class"], Counter())[f["reason"]] += 1
    for f in results["false_positives"]:
        fp_reasons.setdefault(f["predicted_class"], Counter())[f["reason"]] += 1

    rows = []
    for class_name in CLASS_NAMES:
        c = results["per_class"][class_name]
        if c["ground_truth"] == 0 and c["predictions"] == 0:
            continue  # class absent from both GT and predictions
        rows.append({"class": class_name, **c,
                     "fn_reasons": dict(fn_reasons.get(class_name, {})),
                     "fp_reasons": dict(fp_reasons.get(class_name, {}))})

    if print_table:
        print(f"\nPer-class results | match distance = {distance_threshold} m | "
              f"score >= {score_threshold}")
        header = (f"{'class':<22}{'GT':>6}{'pred':>7}{'TP':>6}{'FP':>6}{'FN':>6}"
                  f"{'prec':>7}{'recall':>8}")
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['class']:<22}{r['ground_truth']:>6}{r['predictions']:>7}"
                  f"{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
                  f"{_fmt(r['diagnostic_precision']):>7}{_fmt(r['diagnostic_recall']):>8}")
        t = results["totals"]
        print("-" * len(header))
        print(f"{'TOTAL':<22}{t['ground_truth']:>6}{t['predictions']:>7}"
              f"{t['tp']:>6}{t['fp']:>6}{t['fn']:>6}")
        _print_reason_table("FN reasons by class", fn_reasons, rows)
        _print_reason_table("FP reasons by class", fp_reasons, rows)

    return {"distance_threshold": distance_threshold, "score_threshold": score_threshold,
            "rows": rows, "fn_reasons": {k: dict(v) for k, v in fn_reasons.items()},
            "fp_reasons": {k: dict(v) for k, v in fp_reasons.items()},
            "results": results}