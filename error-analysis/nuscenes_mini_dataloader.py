"""
BEVFusion + nuScenes-mini data loader.

Loads lightweight BEVFusion prediction NPZ files and matching nuScenes ground-truth
annotations. The default split is nuScenes mini_val (81 samples in v1.0-mini).

Coordinate frames are deliberately preserved:
- Ground truth boxes: nuScenes global frame.
- Raw predictions: BEVFusion LiDAR frame, convention
  [x, y, z_bottom, dx, dy, dz, yaw, vx, vy].

This loader does NOT transform boxes or perform matching. That keeps data loading
separate from error-analysis logic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

CATEGORY_PREFIX_TO_CLASS = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}


def category_to_detection_class(category_name: str) -> Optional[str]:
    """Map a nuScenes category to one of the ten detection classes."""
    for prefix, class_name in CATEGORY_PREFIX_TO_CLASS.items():
        if category_name == prefix or category_name.startswith(prefix + "."):
            return class_name
    return None


class NuScenesMiniDataLoader:
    """
    Load matched nuScenes ground truth and saved BEVFusion predictions.

    Args:
        predictions_dir: Directory containing one .npz file per sample.
        dataroot: nuScenes dataset root containing v1.0-mini/.
        version: nuScenes version, normally "v1.0-mini".
        split: nuScenes split, normally "mini_val".
        verbose: Passed to the nuScenes devkit.

    Main API:
        loader.load() -> (ground_truth, predictions)
        loader.load_sample(sample_token) -> (ground_truth_for_sample, prediction_for_sample)

    Returned dictionaries are keyed by nuScenes sample_token. Samples without a
    prediction file raise an error instead of being silently skipped.
    """

    def __init__(
        self,
        predictions_dir: str = (
            "/mnt/e/autonomous_vision/code_backup/scripts/learned-bev/"
            "bevfusion_lightweight_predictions"
        ),
        dataroot: str = r"/mnt/e/autonomous_vision/learned-perception/nuScenes-mini-dataset",
        version: str = "v1.0-mini",
        split: str = "mini_val",
        verbose: bool = False,
    ):
        self.predictions_dir = Path(predictions_dir)
        self.dataroot = Path(dataroot)
        self.version = version
        self.split = split

        if not self.predictions_dir.is_dir():
            raise FileNotFoundError(
                f"Predictions directory not found: {self.predictions_dir}"
            )
        if not self.dataroot.is_dir():
            raise FileNotFoundError(f"nuScenes dataroot not found: {self.dataroot}")
        if not (self.dataroot / self.version).is_dir():
            raise FileNotFoundError(
                f"Expected metadata directory not found: {self.dataroot / self.version}"
            )

        self.nusc = NuScenes(
            version=self.version,
            dataroot=str(self.dataroot),
            verbose=verbose,
        )
        self.samples = self._select_split_samples()
        self.sample_tokens = [sample["token"] for sample in self.samples]
        self._sample_by_token = {sample["token"]: sample for sample in self.samples}
        self._prediction_file_by_token = self._index_prediction_files()

    def _select_split_samples(self) -> List[dict]:
        splits = create_splits_scenes()
        if self.split not in splits:
            raise ValueError(
                f"Unknown split {self.split!r}. Available splits: {sorted(splits)}"
            )

        wanted_scene_names = set(splits[self.split])
        wanted_scene_tokens = {
            scene["token"]
            for scene in self.nusc.scene
            if scene["name"] in wanted_scene_names
        }
        samples = [
            sample
            for sample in self.nusc.sample
            if sample["scene_token"] in wanted_scene_tokens
        ]
        samples.sort(key=lambda item: (item["scene_token"], item["timestamp"]))
        if not samples:
            raise RuntimeError(
                f"No samples selected for split={self.split!r}. "
                "Check the dataset version and dataroot."
            )
        return samples

    def _index_prediction_files(self) -> Dict[str, Path]:
        """Index NPZ files by embedded sample_token, not by filename/index."""
        paths = sorted(self.predictions_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"No .npz prediction files found in {self.predictions_dir}"
            )

        by_token: Dict[str, Path] = {}
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "sample_token",
                    "sample_index",
                    "boxes_3d",
                    "scores_3d",
                    "labels_3d",
                }
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(f"{path.name} is missing fields: {sorted(missing)}")

                token_array = data["sample_token"]
                token = str(
                    token_array.item()
                    if token_array.ndim == 0
                    else token_array.tolist()
                )
                if token in by_token:
                    raise ValueError(
                        f"Duplicate sample_token {token} in {by_token[token].name} "
                        f"and {path.name}"
                    )
                by_token[token] = path

        missing_tokens = [
            token for token in self.sample_tokens if token not in by_token
        ]
        if missing_tokens:
            raise RuntimeError(
                f"{len(missing_tokens)} samples in split {self.split!r} have no "
                f"prediction file. First missing tokens: {missing_tokens[:5]}"
            )
        return by_token

    def _load_ground_truth_for_sample(self, sample: dict) -> List[dict]:
        """Load supported GT annotations; boxes remain in global coordinates."""
        annotations = []
        for annotation_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", annotation_token)
            # nuScenes sample_annotation has instance_token, not category_token.
            # Resolve category through the linked instance record.
            instance = self.nusc.get("instance", ann["instance_token"])
            category_name = self.nusc.get(
                "category", instance["category_token"]
            )["name"]
            class_name = category_to_detection_class(category_name)
            if class_name is None:
                continue

            annotations.append(
                {
                    "sample_token": sample["token"],
                    "annotation_token": ann["token"],
                    "instance_token": ann["instance_token"],
                    "class_name": class_name,
                    "category_name": category_name,
                    "translation_global": np.asarray(
                        ann["translation"], dtype=np.float32
                    ),
                    "size_wlh": np.asarray(ann["size"], dtype=np.float32),
                    "rotation_global_wxyz": np.asarray(
                        ann["rotation"], dtype=np.float32
                    ),
                    "yaw_global": float(
                        Quaternion(ann["rotation"]).yaw_pitch_roll[0]
                    ),
                    "num_lidar_pts": int(ann.get("num_lidar_pts", 0)),
                    "num_radar_pts": int(ann.get("num_radar_pts", 0)),
                    "attribute_tokens": list(ann.get("attribute_tokens", [])),
                }
            )
        return annotations

    def _load_predictions_for_sample(self, sample_token: str) -> dict:
        """Load prediction arrays exactly as saved; no transform/filter applied."""
        path = self._prediction_file_by_token[sample_token]
        with np.load(path, allow_pickle=False) as data:
            boxes = np.asarray(data["boxes_3d"], dtype=np.float32)
            scores = np.asarray(data["scores_3d"], dtype=np.float32)
            labels = np.asarray(data["labels_3d"], dtype=np.int32)
            sample_index = int(np.asarray(data["sample_index"]).item())
            stored_token = str(np.asarray(data["sample_token"]).item())

        if boxes.ndim != 2 or boxes.shape[1] != 9:
            raise ValueError(
                f"{path.name}: expected boxes_3d shape (N, 9), got {boxes.shape}"
            )
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError(f"{path.name}: boxes/scores/labels lengths differ")
        if stored_token != sample_token:
            raise ValueError(
                f"Token mismatch in {path.name}: {stored_token} != {sample_token}"
            )

        return {
            "sample_token": sample_token,
            "sample_index": sample_index,
            "source_file": str(path),
            "coordinate_frame": "LIDAR_TOP",
            "box_convention": [
                "x", "y", "z_bottom", "dx", "dy", "dz", "yaw", "vx", "vy"
            ],
            "boxes_3d": boxes,
            "scores_3d": scores,
            "labels_3d": labels,
            "class_names": [
                CLASS_NAMES[label] if 0 <= int(label) < len(CLASS_NAMES) else "unknown"
                for label in labels
            ],
        }

    def load_sample(self, sample_token: str) -> Tuple[List[dict], dict]:
        """Return (ground_truth_list, raw_prediction_dict) for one sample token."""
        if sample_token not in self._sample_by_token:
            raise KeyError(
                f"Sample token is not part of split {self.split!r}: {sample_token}"
            )
        sample = self._sample_by_token[sample_token]
        gt = self._load_ground_truth_for_sample(sample)
        predictions = self._load_predictions_for_sample(sample_token)
        return gt, predictions

    def load(self) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
        """
        Load all samples in the selected split.

        Returns:
            ground_truth: {sample_token: [GT annotation dict, ...]}
            predictions:  {sample_token: raw prediction dict}
        """
        ground_truth: Dict[str, List[dict]] = {}
        predictions: Dict[str, dict] = {}
        for sample in self.samples:
            token = sample["token"]
            gt, pred = self.load_sample(token)
            ground_truth[token] = gt
            predictions[token] = pred
        return ground_truth, predictions

    def get_sample_metadata(self, sample_token: str) -> dict:
        """Return sensor/pose records needed for a later coordinate transform."""
        if sample_token not in self._sample_by_token:
            raise KeyError(f"Unknown sample token for split {self.split}: {sample_token}")
        sample = self._sample_by_token[sample_token]
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        calibrated_sensor = self.nusc.get(
            "calibrated_sensor", lidar_sd["calibrated_sensor_token"]
        )
        ego_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        return {
            "sample_token": sample_token,
            "timestamp": sample["timestamp"],
            "scene_token": sample["scene_token"],
            "lidar_sample_data_token": lidar_sd["token"],
            "calibrated_sensor": calibrated_sensor,
            "ego_pose": ego_pose,
        }


# if __name__ == "__main__":
#     loader = NuScenesMiniDataLoader()
#     gt_by_token, pred_by_token = loader.load()

#     print(f"Split: {loader.split}")
#     print(f"Samples loaded: {len(gt_by_token)}")
#     print(f"GT annotations loaded: {sum(len(v) for v in gt_by_token.values())}")
#     print(
#         "Predictions loaded: "
#         f"{sum(len(v['scores_3d']) for v in pred_by_token.values())}"
#     )

#     first_token = loader.sample_tokens[0]
#     gt, pred = loader.load_sample(first_token)
#     print(f"\nExample sample token: {first_token}")
#     print(f"GT boxes: {len(gt)}")
#     print(f"Prediction boxes: {len(pred['scores_3d'])}")
#     print(f"Raw prediction box shape: {pred['boxes_3d'].shape}")
#     print(f"First GT class: {gt[0]['class_name'] if gt else 'no supported GT'}")
#     print(f"First prediction class: {pred['class_names'][0] if pred['class_names'] else 'none'}")
