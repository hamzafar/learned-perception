"""In-memory diagnostic detection-failure analysis for nuScenes-mini.

Call analyze_detection_failures(ground_truth, predictions, nusc) from main.py.
No files are written. Returned results are ordinary Python dictionaries/lists.
Matching is diagnostic class-aware XY center-distance matching, not official
nuScenes evaluation.
"""
import numpy as np
from pyquaternion import Quaternion

CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def _prediction_to_global(box, calibrated_sensor, ego_pose):
    """Transform [x,y,z_bottom,dx,dy,dz,yaw,vx,vy] from LiDAR to global."""
    x, y, z_bottom, dx, dy, dz, yaw, vx, vy = map(float, box)
    q_sensor = Quaternion(calibrated_sensor["rotation"])
    q_ego = Quaternion(ego_pose["rotation"])
    center_lidar = np.array([x, y, z_bottom + dz / 2.0])
    center_ego = q_sensor.rotate(center_lidar) + np.asarray(calibrated_sensor["translation"])
    center_global = q_ego.rotate(center_ego) + np.asarray(ego_pose["translation"])
    q_box_global = q_ego * q_sensor * Quaternion(axis=[0, 0, 1], radians=yaw)
    return center_global, float(q_box_global.yaw_pitch_roll[0])


def _greedy_match(preds, gts, distance_threshold, class_aware=True):
    """One-to-one greedy match by XY distance."""
    candidates = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if class_aware and pred["class_name"] != gt["class_name"]:
                continue
            distance = float(np.linalg.norm(pred["center"][:2] - gt["center"][:2]))
            if distance <= distance_threshold:
                candidates.append((distance, pi, gi))
    used_preds, used_gts, pairs = set(), set(), []
    for distance, pi, gi in sorted(candidates):
        if pi not in used_preds and gi not in used_gts:
            used_preds.add(pi)
            used_gts.add(gi)
            pairs.append((pi, gi, distance))
    return pairs, used_preds, used_gts


def analyze_detection_failures(ground_truth, predictions, nusc, distance_threshold=2.0,
                               score_threshold=0.0):
    """Analyze detections and return all results in a dictionary.

    Args:
        ground_truth: GT dictionary returned by NuScenesMiniDataLoader.load().
        predictions: prediction dictionary returned by NuScenesMiniDataLoader.load().
        nusc: the loader's nuScenes devkit object (loader.nusc).
        distance_threshold: maximum XY center distance in meters for a match.
        score_threshold: optional minimum prediction score; default retains all.

    Returns:
        Dictionary with per-class counts, FP cases, FN cases, matched detections,
        candidate class confusions, and overall totals.
    """
    stats = {name: {"ground_truth": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
             for name in CLASS_NAMES}
    false_positives, false_negatives, matched_detections, class_confusions = [], [], [], []

    for sample_token, gt_records in ground_truth.items():
        sample = nusc.get("sample", sample_token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        calibrated_sensor = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        gts = [{
            "class_name": gt["class_name"],
            "center": np.asarray(gt["translation_global"], dtype=float),
            "annotation_token": gt.get("annotation_token"),
            "instance_token": gt.get("instance_token"),
        } for gt in gt_records]

        sample_pred = predictions.get(sample_token)
        if sample_pred is None:
            sample_pred = {"boxes_3d": [], "scores_3d": [], "labels_3d": []}
        preds = []
        for box, score, label in zip(sample_pred["boxes_3d"], sample_pred["scores_3d"],
                                     sample_pred["labels_3d"]):
            label, score = int(label), float(score)
            if score < score_threshold or not (0 <= label < len(CLASS_NAMES)):
                continue
            center, yaw_global = _prediction_to_global(box, calibrated_sensor, ego_pose)
            preds.append({"class_name": CLASS_NAMES[label], "score": score,
                          "center": center, "yaw_global": yaw_global,
                          "box_lidar": np.asarray(box, dtype=float)})

        for gt in gts:
            stats[gt["class_name"]]["ground_truth"] += 1
        for pred in preds:
            stats[pred["class_name"]]["predictions"] += 1

        pairs, used_preds, used_gts = _greedy_match(preds, gts, distance_threshold, True)
        for pi, gi, distance in pairs:
            pred, gt = preds[pi], gts[gi]
            stats[pred["class_name"]]["tp"] += 1
            matched_detections.append({
                "sample_token": sample_token,
                "class_name": pred["class_name"],
                "score": pred["score"],
                "center_distance_m": distance,
                "pred_center_global": pred["center"].tolist(),
                "gt_center_global": gt["center"].tolist(),
                "gt_annotation_token": gt["annotation_token"],
            })

        fp_indices = [i for i in range(len(preds)) if i not in used_preds]
        fn_indices = [i for i in range(len(gts)) if i not in used_gts]
        remaining_preds = [preds[i] for i in fp_indices]
        remaining_gts = [gts[i] for i in fn_indices]
        confusion_pairs, _, _ = _greedy_match(remaining_preds, remaining_gts,
                                               distance_threshold, False)
        confusion_pred_indices, confusion_gt_indices = set(), set()
        for rpi, rgi, distance in confusion_pairs:
            pred, gt = remaining_preds[rpi], remaining_gts[rgi]
            if pred["class_name"] != gt["class_name"]:
                confusion_pred_indices.add(rpi)
                confusion_gt_indices.add(rgi)
                class_confusions.append({
                    "sample_token": sample_token,
                    "predicted_class": pred["class_name"],
                    "ground_truth_class": gt["class_name"],
                    "score": pred["score"],
                    "center_distance_m": distance,
                    "gt_annotation_token": gt["annotation_token"],
                    "pred_center_global": pred["center"].tolist(),
                    "gt_center_global": gt["center"].tolist(),
                })

        for local_i, pred_i in enumerate(fp_indices):
            pred = preds[pred_i]
            stats[pred["class_name"]]["fp"] += 1
            nearest_same_class = min(
                (float(np.linalg.norm(pred["center"][:2] - gt["center"][:2]))
                 for gt in gts if gt["class_name"] == pred["class_name"]),
                default=float("inf"),
            )
            if local_i in confusion_pred_indices:
                reason = "possible_class_confusion"
            elif nearest_same_class <= distance_threshold:
                reason = "possible_duplicate_or_unmatched_nearby_detection"
            else:
                reason = "no_nearby_same_class_gt_or_localization_error"
            false_positives.append({
                "sample_token": sample_token,
                "predicted_class": pred["class_name"],
                "score": pred["score"],
                "pred_center_global": pred["center"].tolist(),
                "reason": reason,
            })

        for local_i, gt_i in enumerate(fn_indices):
            gt = gts[gt_i]
            stats[gt["class_name"]]["fn"] += 1
            nearest_same_class = min(
                (float(np.linalg.norm(gt["center"][:2] - pred["center"][:2]))
                 for pred in preds if pred["class_name"] == gt["class_name"]),
                default=float("inf"),
            )
            if local_i in confusion_gt_indices:
                reason = "possible_class_confusion"
            elif nearest_same_class <= distance_threshold:
                reason = "nearby_same_class_prediction_not_matched"
            else:
                reason = "no_nearby_same_class_prediction"
            false_negatives.append({
                "sample_token": sample_token,
                "ground_truth_class": gt["class_name"],
                "gt_center_global": gt["center"].tolist(),
                "gt_annotation_token": gt["annotation_token"],
                "instance_token": gt["instance_token"],
                "reason": reason,
            })

    per_class = {}
    for class_name, counts in stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        per_class[class_name] = {
            **counts,
            "diagnostic_precision": tp / (tp + fp) if tp + fp else 0.0,
            "diagnostic_recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    totals = {key: sum(item[key] for item in stats.values())
              for key in ("ground_truth", "predictions", "tp", "fp", "fn")}
    return {
        "settings": {"distance_threshold_m": float(distance_threshold),
                     "score_threshold": float(score_threshold),
                     "matching": "greedy class-aware XY center distance; diagnostic only"},
        "totals": totals,
        "per_class": per_class,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matched_detections": matched_detections,
        "candidate_class_confusions": class_confusions,
    }
