from nuscenes_mini_dataloader import NuScenesMiniDataLoader
from detection_failures import (
    analyze_detection_failures,
    make_gt_as_predictions,
    sweep_distance_thresholds,
)

from per_class_report import per_class_breakdown


def main():

    loader = NuScenesMiniDataLoader(split="mini_val")
    ground_truth, predictions = loader.load()

    # 1) Pipeline sanity check: perfect predictions built from GT
    gt_preds = make_gt_as_predictions(ground_truth, loader.nusc)
    check = analyze_detection_failures(ground_truth, gt_preds, loader.nusc)
    print(check["totals"], check["ignored"])
    # Expect fn == 0 and ~0 m errors in check["matched_detections"].
    # FPs here come only from zero-point GT (reason "near_ignored_gt").

    # 2) The real analysis
    results = analyze_detection_failures(
        ground_truth=ground_truth,
        predictions=predictions,
        nusc=loader.nusc,
        distance_threshold=2.0,   # or {"pedestrian": 1.0, "bus": 3.0, ...}
        score_threshold=0.3,      # see note below
    )

    # 3) Check which conclusions depend on the match radius
    print(sweep_distance_thresholds(ground_truth, predictions, loader.nusc,
                                    score_threshold=0.3))


    report_1m = per_class_breakdown(ground_truth, predictions, loader.nusc,
                                    distance_threshold=1.0, score_threshold=0.3)

    # other radii, same call
    # per_class_breakdown(ground_truth, predictions, loader.nusc,
    #                     distance_threshold=2.0, score_threshold=0.3)
    # per_class_breakdown(ground_truth, predictions, loader.nusc,
    #                     distance_threshold=4.0, score_threshold=0.3)



if __name__ == "__main__":
    main()