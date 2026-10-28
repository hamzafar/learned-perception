from nuscenes_mini_dataloader import NuScenesMiniDataLoader


def main():
    # The dataloader owns the dataset paths and split configuration.
    loader = NuScenesMiniDataLoader()

    # Dictionaries keyed by nuScenes sample_token:
    # ground_truth[token] -> list of GT annotation dictionaries
    # predictions[token]  -> prediction dictionary with boxes/scores/labels
    ground_truth, predictions = loader.load()

    total_gt = sum(len(annotations) for annotations in ground_truth.values())
    total_predictions = sum(
        len(sample_predictions["scores_3d"])
        for sample_predictions in predictions.values()
    )

    print(f"Split: {loader.split}")
    print(f"Samples loaded: {len(ground_truth)}")
    print(f"GT annotations loaded: {total_gt}")
    print(f"Predictions loaded: {total_predictions}")

    if not ground_truth:
        print("No ground-truth samples were loaded.")
        return

    # Show a small sanity-check example.
    sample_token = next(iter(ground_truth))
    sample_gt = ground_truth[sample_token]
    sample_pred = predictions[sample_token]

    print(f"\nExample sample token: {sample_token}")
    print(f"GT boxes: {len(sample_gt)}")
    print(f"Prediction boxes: {len(sample_pred['scores_3d'])}")
    print(f"Raw prediction box shape: {sample_pred['boxes_3d'].shape}")
    print(f"First GT class: {sample_gt[0]['class_name'] if sample_gt else 'none'}")
    print(
        "First prediction class: "
        f"{sample_pred['class_names'][0] if sample_pred['class_names'] else 'none'}"
    )

    # These variables are now ready for the next analysis stage:
    # ground_truth: dict[str, list[dict]]
    # predictions: dict[str, dict]
    return ground_truth, predictions


if __name__ == "__main__":
    main()