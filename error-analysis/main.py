from nuscenes_mini_dataloader import NuScenesMiniDataLoader
from detection_failures import analyze_detection_failures



def main():

    # load min val ground truth and predictions
    loader = NuScenesMiniDataLoader(split="mini_val")
    ground_truth, predictions = loader.load()

    # Analyze detection failures
    results = analyze_detection_failures(
        ground_truth=ground_truth,
        predictions=predictions,
        nusc=loader.nusc,
    )

    print("Totals:", results["totals"])
    print("Car results:", results["per_class"]["car"])
    print("False positives:", len(results["false_positives"]))
    print("False negatives:", len(results["false_negatives"]))



if __name__ == "__main__":
    main()