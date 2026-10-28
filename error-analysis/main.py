from nuscenes_mini_dataloader import NuScenesMiniDataLoader


def main():

    # load min val ground truth and predictions
    loader = NuScenesMiniDataLoader(split="mini_val")
    ground_truth, predictions = loader.load()

    print(f"GT samples: {len(ground_truth)}")
    print(f"Prediction samples: {len(predictions)}")



if __name__ == "__main__":
    main()