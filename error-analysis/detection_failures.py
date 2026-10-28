"""In-memory diagnostic detection-failure analysis for nuScenes-mini.

Call analyze_detection_failures(ground_truth, predictions, nusc) from main.py.
No files are written. Returned results are ordinary Python dictionaries/lists.

Matching is a diagnostic approximation of the official nuScenes evaluation:
  * GT and predictions are filtered with the devkit's per-class range
    and the GT zero-point rule (switchable, see apply_devkit_filters).
  * Predictions are matched in descending score order, each to the nearest
    unmatched same-class GT by XY center distance (devkit style).
It is still NOT the official metric; use it to explain errors, not to report them.

Expected inputs
  ground_truth[sample_token] -> list of dicts with at least
      class_name, translation_global, annotation_token (and instance_token).
      Size, yaw, num_lidar_pts, num_radar_pts and visibility are read from
      nusc.get("sample_annotation", annotation_token) when a token is present.
  predictions[sample_token]  -> dict with boxes_3d, scores_3d, labels_3d, where
      boxes_3d rows are [x, y, z_bottom, dx, dy, dz, yaw, vx, vy] in LiDAR frame
      and (w, l, h) = (dy, dx, dz), as in the conversion that worked for eval.
"""
import warnings

import numpy as np
from pyquaternion import Quaternion

CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# Per-class evaluation range in meters (nuScenes detection config).
CLASS_RANGE_M = {
    "car": 50.0, "truck": 50.0, "bus": 50.0, "trailer": 50.0,
    "construction_vehicle": 50.0, "pedestrian": 40.0, "motorcycle": 40.0,
    "bicycle": 40.0, "traffic_cone": 30.0, "barrier": 30.0,
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _resolve(value, class_name, default=None):
    """Return value for class_name when value is a dict, else value itself."""
    if isinstance(value, dict):
        return float(value.get(class_name, default))
    return float(value)


def _wrap_pi(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _xy_dist(a, b):
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def _nearest(items, center):
    """Nearest item (by XY distance) -> (distance, item); (inf, None) if empty."""
    best_d, best_item = float("inf"), None
    for item in items:
        d = _xy_dist(item["center"], center)
        if d < best_d:
            best_d, best_item = d, item
    return best_d, best_item


# --------------------------------------------------------------------------
# Frame transforms
# --------------------------------------------------------------------------
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


def _global_to_lidar_box(center_global, size_wlh, rotation_global, calibrated_sensor, ego_pose):
    """Inverse of _prediction_to_global; returns a 9-value LiDAR-frame box."""
    q_sensor = Quaternion(calibrated_sensor["rotation"])
    q_ego = Quaternion(ego_pose["rotation"])
    center_ego = q_ego.inverse.rotate(np.asarray(center_global) - np.asarray(ego_pose["translation"]))
    center_lidar = q_sensor.inverse.rotate(center_ego - np.asarray(calibrated_sensor["translation"]))
    q_rel = (q_ego * q_sensor).inverse * Quaternion(rotation_global)
    yaw = float(q_rel.yaw_pitch_roll[0])
    w, l, h = map(float, size_wlh)
    return [float(center_lidar[0]), float(center_lidar[1]), float(center_lidar[2] - h / 2.0),
            l, w, h, yaw, 0.0, 0.0]


def make_gt_as_predictions(ground_truth, nusc):
    """Turn GT into a perfect prediction set (sanity check for the pipeline).

    Feeding the result to analyze_detection_failures should give zero FP and
    zero FN (apart from GT removed by the devkit filters) and ~0 m errors. If it
    does not, the problem is in the transforms or conventions, not the model.
    """
    predictions = {}
    for sample_token, gt_records in ground_truth.items():
        sample = nusc.get("sample", sample_token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        calibrated_sensor = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        boxes, scores, labels = [], [], []
        for gt in gt_records:
            if gt["class_name"] not in CLASS_NAMES or not gt.get("annotation_token"):
                continue
            ann = nusc.get("sample_annotation", gt["annotation_token"])
            boxes.append(_global_to_lidar_box(gt["translation_global"], ann["size"],
                                              ann["rotation"], calibrated_sensor, ego_pose))
            scores.append(1.0)
            labels.append(CLASS_NAMES.index(gt["class_name"]))
        predictions[sample_token] = {
            "boxes_3d": np.asarray(boxes, dtype=float).reshape(-1, 9),
            "scores_3d": np.asarray(scores, dtype=float),
            "labels_3d": np.asarray(labels, dtype=int),
        }
    return predictions


# --------------------------------------------------------------------------
# Record builders
# --------------------------------------------------------------------------
def _build_gts(gt_records, nusc, ego_xy, apply_filters, counters):
    """Split GT into kept and ignored, attaching diagnostic metadata."""
    kept, ignored = [], []
    for gt in gt_records:
        class_name = gt["class_name"]
        if class_name not in CLASS_NAMES:
            counters["skipped_unknown_class"] += 1
            continue
        center = np.asarray(gt["translation_global"], dtype=float)
        token = gt.get("annotation_token")
        size = yaw = visibility = num_lidar = num_radar = None
        if token:
            ann = nusc.get("sample_annotation", token)
            size = [float(v) for v in ann["size"]]  # (w, l, h)
            yaw = float(Quaternion(ann["rotation"]).yaw_pitch_roll[0])
            visibility = ann.get("visibility_token")
            num_lidar = ann.get("num_lidar_pts")
            num_radar = ann.get("num_radar_pts")
        record = {
            "class_name": class_name, "center": center,
            "annotation_token": token, "instance_token": gt.get("instance_token"),
            "size_wlh": size, "yaw_global": yaw, "visibility": visibility,
            "num_lidar_pts": num_lidar, "num_radar_pts": num_radar,
            "ego_distance_m": _xy_dist(center, ego_xy),
        }
        ignore_reason = None
        if apply_filters:
            if record["ego_distance_m"] > CLASS_RANGE_M[class_name]:
                ignore_reason = "out_of_range"
            elif num_lidar is not None and num_radar is not None and (num_lidar + num_radar) == 0:
                ignore_reason = "no_lidar_or_radar_points"
        if ignore_reason:
            record["ignore_reason"] = ignore_reason
            ignored.append(record)
        else:
            kept.append(record)
    return kept, ignored


def _gt_fields(gt):
    return {
        "ground_truth_class": gt["class_name"],
        "gt_center_global": gt["center"].tolist(),
        "gt_annotation_token": gt["annotation_token"],
        "instance_token": gt["instance_token"],
        "gt_size_wlh": gt["size_wlh"],
        "gt_yaw_global": gt["yaw_global"],
        "gt_visibility": gt["visibility"],
        "gt_num_lidar_pts": gt["num_lidar_pts"],
        "gt_num_radar_pts": gt["num_radar_pts"],
        "ego_distance_m": gt["ego_distance_m"],
    }


def _pred_fields(pred):
    return {
        "predicted_class": pred["class_name"],
        "score": pred["score"],
        "pred_center_global": pred["center"].tolist(),
        "pred_yaw_global": pred["yaw_global"],
        "pred_size_wlh": pred["size_wlh"],
        "pred_box_lidar": pred["box_lidar"].tolist(),
        "ego_distance_m": pred["ego_distance_m"],
    }


def _pair_errors(pred, gt):
    """Signed error components of a matched pair (GT heading frame)."""
    diff = pred["center"] - gt["center"]
    out = {"center_distance_xy_m": float(np.linalg.norm(diff[:2])),
           "z_error_m": float(diff[2])}
    if gt["yaw_global"] is not None:
        c, s = np.cos(gt["yaw_global"]), np.sin(gt["yaw_global"])
        yaw_err = _wrap_pi(pred["yaw_global"] - gt["yaw_global"])
        out.update({
            "longitudinal_error_m": float(c * diff[0] + s * diff[1]),  # + = ahead of GT
            "lateral_error_m": float(-s * diff[0] + c * diff[1]),      # + = left of GT
            "yaw_error_rad": yaw_err,                                  # signed, wrapped
            "yaw_flipped": bool(abs(yaw_err) > np.pi / 2),             # ~180 deg flip
        })
    if gt["size_wlh"] is not None:
        out["size_ratio_wlh"] = [float(p / g) if g else None
                                 for p, g in zip(pred["size_wlh"], gt["size_wlh"])]
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def _match_by_score(preds, gts, threshold_for, class_aware):
    """Devkit-style matching: predictions in descending score order, each takes
    the nearest still-unmatched GT within its class threshold.
    Returns list of (pred_index, gt_index, distance)."""
    used_gts, pairs = set(), []
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i]["score"]):
        pred = preds[pi]
        thr = threshold_for(pred["class_name"])
        best = None
        for gi, gt in enumerate(gts):
            if gi in used_gts:
                continue
            if class_aware and gt["class_name"] != pred["class_name"]:
                continue
            d = _xy_dist(pred["center"], gt["center"])
            if d <= thr and (best is None or d < best[0]):
                best = (d, gi)
        if best is not None:
            used_gts.add(best[1])
            pairs.append((pi, best[1], best[0]))
    return pairs


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def analyze_detection_failures(ground_truth, predictions, nusc, distance_threshold=2.0,
                               score_threshold=0.0, diagnostic_radius=4.0,
                               apply_devkit_filters=True):
    """Analyze detections and return all results in a dictionary.

    Args:
        ground_truth: GT dictionary returned by NuScenesMiniDataLoader.load().
        predictions: prediction dictionary returned by NuScenesMiniDataLoader.load().
        nusc: the loader's nuScenes devkit object (loader.nusc).
        distance_threshold: match radius in meters; a float or {class_name: float}.
        score_threshold: minimum score for a prediction to count as a detection;
            a float or {class_name: float}. Predictions below it are NOT discarded:
            they are kept to explain FNs ("detected_below_score_threshold").
        diagnostic_radius: wider radius (m) used only to label near-misses
            (mislocalized boxes) instead of lumping them with pure misses.
        apply_devkit_filters: drop GT/predictions beyond the class range and GT with
            zero LiDAR+radar points, as the official evaluation does.

    Returns:
        Dictionary with settings, totals, per-class counts, FP cases, FN cases,
        matched detections (with signed errors), class confusions, and ignored counts.
    """
    gt_keys, pred_keys = set(ground_truth), set(predictions)
    missing_pred = sorted(gt_keys - pred_keys)
    extra_pred = sorted(pred_keys - gt_keys)
    if missing_pred:
        warnings.warn(f"{len(missing_pred)} GT samples have no predictions; their objects "
                      f"will all count as FN. Check GT and predictions use the same split.")
    if extra_pred:
        warnings.warn(f"{len(extra_pred)} prediction samples have no GT and are ignored.")

    def thr_for(class_name):
        return _resolve(distance_threshold, class_name, 2.0)

    def score_thr_for(class_name):
        return _resolve(score_threshold, class_name, 0.0)

    stats = {name: {"ground_truth": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
             for name in CLASS_NAMES}
    counters = {"skipped_unknown_class": 0}
    ignored_gt_counts = {"out_of_range": 0, "no_lidar_or_radar_points": 0}
    ignored_pred_out_of_range = 0
    below_threshold_predictions = 0
    false_positives, false_negatives, matched_detections, class_confusions = [], [], [], []

    for sample_token, gt_records in ground_truth.items():
        sample = nusc.get("sample", sample_token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        calibrated_sensor = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_xy = np.asarray(ego_pose["translation"], dtype=float)[:2]

        gts, ignored_gts = _build_gts(gt_records, nusc, ego_xy, apply_devkit_filters, counters)
        for g in ignored_gts:
            ignored_gt_counts[g["ignore_reason"]] += 1

        sample_pred = predictions.get(sample_token)
        if sample_pred is None:
            sample_pred = {"boxes_3d": [], "scores_3d": [], "labels_3d": []}
        active, below = [], []
        for box, score, label in zip(sample_pred["boxes_3d"], sample_pred["scores_3d"],
                                     sample_pred["labels_3d"]):
            label, score = int(label), float(score)
            if not (0 <= label < len(CLASS_NAMES)):
                continue
            class_name = CLASS_NAMES[label]
            center, yaw_global = _prediction_to_global(box, calibrated_sensor, ego_pose)
            ego_distance = _xy_dist(center, ego_xy)
            if apply_devkit_filters and ego_distance > CLASS_RANGE_M[class_name]:
                ignored_pred_out_of_range += 1
                continue
            box = np.asarray(box, dtype=float)
            pred = {"class_name": class_name, "score": score, "center": center,
                    "yaw_global": yaw_global, "box_lidar": box,
                    "size_wlh": [float(box[4]), float(box[3]), float(box[5])],  # (dy, dx, dz)
                    "ego_distance_m": ego_distance}
            (active if score >= score_thr_for(class_name) else below).append(pred)
        below_threshold_predictions += len(below)

        for gt in gts:
            stats[gt["class_name"]]["ground_truth"] += 1
        for pred in active:
            stats[pred["class_name"]]["predictions"] += 1

        # --- class-aware matching (devkit style) ---
        pairs = _match_by_score(active, gts, thr_for, class_aware=True)
        matched_pred, matched_gt = {}, {}
        for pi, gi, distance in pairs:
            matched_pred[pi], matched_gt[gi] = gi, pi
            pred, gt = active[pi], gts[gi]
            stats[pred["class_name"]]["tp"] += 1
            matched_detections.append({
                "sample_token": sample_token, "class_name": pred["class_name"],
                "score": pred["score"], **_pair_errors(pred, gt),
                "pred_center_global": pred["center"].tolist(),
                "gt_center_global": gt["center"].tolist(),
                "gt_annotation_token": gt["annotation_token"],
                "instance_token": gt["instance_token"],
                "ego_distance_m": gt["ego_distance_m"],
                "gt_visibility": gt["visibility"],
                "gt_num_lidar_pts": gt["num_lidar_pts"],
                "gt_num_radar_pts": gt["num_radar_pts"],
            })

        fp_indices = [i for i in range(len(active)) if i not in matched_pred]
        fn_indices = [i for i in range(len(gts)) if i not in matched_gt]

        # --- class-agnostic re-match of leftovers -> class confusions ---
        rem_preds = [active[i] for i in fp_indices]
        rem_gts = [gts[i] for i in fn_indices]
        confusion_pairs = _match_by_score(rem_preds, rem_gts, thr_for, class_aware=False)
        confusion_pred_idx, confusion_gt_idx = set(), set()
        for rpi, rgi, distance in confusion_pairs:
            pred, gt = rem_preds[rpi], rem_gts[rgi]
            if pred["class_name"] == gt["class_name"]:
                continue  # cannot happen (they would have matched), kept as a guard
            confusion_pred_idx.add(fp_indices[rpi])
            confusion_gt_idx.add(fn_indices[rgi])
            class_confusions.append({
                "sample_token": sample_token,
                "predicted_class": pred["class_name"],
                "ground_truth_class": gt["class_name"],
                "score": pred["score"], "center_distance_m": distance,
                "gt_annotation_token": gt["annotation_token"],
                "pred_center_global": pred["center"].tolist(),
                "gt_center_global": gt["center"].tolist(),
                "ego_distance_m": gt["ego_distance_m"],
                "gt_visibility": gt["visibility"],
                "gt_num_lidar_pts": gt["num_lidar_pts"],
            })

        # --- false positives ---
        for pi in fp_indices:
            pred = active[pi]
            stats[pred["class_name"]]["fp"] += 1
            thr = thr_for(pred["class_name"])
            same_gts = [g for g in gts if g["class_name"] == pred["class_name"]]
            same_ignored = [g for g in ignored_gts if g["class_name"] == pred["class_name"]]
            d_same, _ = _nearest(same_gts, pred["center"])
            d_ign, ign_gt = _nearest(same_ignored, pred["center"])
            d_any, gt_any = _nearest(gts, pred["center"])
            if pi in confusion_pred_idx:
                reason = "class_confusion"
            elif d_same <= thr:
                reason = "duplicate_of_matched_gt"  # its GT was already taken by a better-scored box
            elif d_ign <= thr:
                reason = "near_ignored_gt"  # GT removed by range or zero-point filter
            elif d_same <= diagnostic_radius:
                reason = "mislocalized_same_class_gt_nearby"
            else:
                reason = "background_no_gt_nearby"
            false_positives.append({
                "sample_token": sample_token, **_pred_fields(pred), "reason": reason,
                "nearest_same_class_gt_distance_m": d_same,
                "nearest_any_gt_distance_m": d_any,
                "nearest_any_gt_class": gt_any["class_name"] if gt_any else None,
                "nearest_any_gt_annotation_token": gt_any["annotation_token"] if gt_any else None,
                "near_ignored_gt_reason": ign_gt["ignore_reason"] if ign_gt and d_ign <= thr else None,
            })

        # --- false negatives ---
        for gi in fn_indices:
            gt = gts[gi]
            stats[gt["class_name"]]["fn"] += 1
            thr = thr_for(gt["class_name"])
            same_active = [p for p in active if p["class_name"] == gt["class_name"]]
            same_below = [p for p in below if p["class_name"] == gt["class_name"]]
            d_same, p_same = _nearest(same_active, gt["center"])
            d_below, p_below = _nearest(same_below, gt["center"])
            d_any, p_any = _nearest(active + below, gt["center"])
            if gi in confusion_gt_idx:
                reason = "class_confusion"
            elif d_below <= thr:
                reason = "detected_below_score_threshold"
            elif d_same <= thr:
                reason = "nearby_prediction_claimed_by_other_gt"
            elif min(d_same, d_below) <= diagnostic_radius:
                reason = "mislocalized_same_class_prediction_nearby"  # active or below-threshold
            else:
                reason = "no_same_class_prediction_nearby"
            false_negatives.append({
                "sample_token": sample_token, **_gt_fields(gt), "reason": reason,
                "nearest_same_class_pred_distance_m": d_same,
                "nearest_same_class_pred_score": p_same["score"] if p_same else None,
                "nearest_below_threshold_pred_distance_m": d_below,
                "nearest_below_threshold_pred_score": p_below["score"] if p_below else None,
                "nearest_any_pred_distance_m": d_any,
                "nearest_any_pred_class": p_any["class_name"] if p_any else None,
                "nearest_any_pred_score": p_any["score"] if p_any else None,
            })

    per_class = {}
    for class_name, counts in stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        per_class[class_name] = {
            **counts,
            # None = undefined (no predictions / no GT), not "zero".
            "diagnostic_precision": tp / (tp + fp) if tp + fp else None,
            "diagnostic_recall": tp / (tp + fn) if tp + fn else None,
        }

    totals = {key: sum(item[key] for item in stats.values())
              for key in ("ground_truth", "predictions", "tp", "fp", "fn")}
    return {
        "settings": {
            "distance_threshold_m": distance_threshold,
            "score_threshold": score_threshold,
            "diagnostic_radius_m": float(diagnostic_radius),
            "apply_devkit_filters": bool(apply_devkit_filters),
            "matching": "score-descending, class-aware XY center distance; diagnostic only",
            "samples_without_predictions": missing_pred,
            "prediction_samples_without_gt": extra_pred,
        },
        "totals": totals,
        "per_class": per_class,
        "ignored": {
            "ground_truth": ignored_gt_counts,
            "predictions_out_of_range": ignored_pred_out_of_range,
            "skipped_unknown_class_gt": counters["skipped_unknown_class"],
            "predictions_below_score_threshold": below_threshold_predictions,
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "matched_detections": matched_detections,
        "candidate_class_confusions": class_confusions,
    }


def sweep_distance_thresholds(ground_truth, predictions, nusc, thresholds=(1.0, 2.0, 4.0), **kwargs):
    """Run the analysis at several match radii and return per-threshold totals,
    to see which conclusions depend on the radius."""
    return {float(t): analyze_detection_failures(ground_truth, predictions, nusc,
                                                  distance_threshold=t, **kwargs)["totals"]
            for t in thresholds}