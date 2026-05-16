from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def font(size: int = 18) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def build_features(rows: list[dict], max_seconds: float) -> tuple[np.ndarray, list[int], list[float]]:
    selected = [r for r in rows if float(r["timestamp_s"]) <= max_seconds]
    velocities = np.array([r["dynamics"]["hand_velocity_px_per_analysis_frame"] for r in selected], dtype=float)
    motion_area = np.array([r["hand_proxy"]["motion_area_ratio"] for r in selected], dtype=float)
    distance = np.array([r["object_proxy"]["hand_object_distance_norm"] for r in selected], dtype=float)
    wrist_x = np.array([r["hand_proxy"]["wrist_xy_norm"][0] for r in selected], dtype=float)
    wrist_y = np.array([r["hand_proxy"]["wrist_xy_norm"][1] for r in selected], dtype=float)
    acceleration = np.r_[0.0, np.diff(velocities)]

    def rolling(values: np.ndarray, window: int, reducer: str) -> np.ndarray:
        out = np.zeros_like(values, dtype=float)
        for i in range(len(values)):
            start = max(0, i - window + 1)
            block = values[start : i + 1]
            out[i] = block.mean() if reducer == "mean" else block.max()
        return out

    X = np.column_stack(
        [
            velocities,
            motion_area,
            distance,
            wrist_x,
            wrist_y,
            acceleration,
            rolling(velocities, 5, "mean"),
            rolling(velocities, 10, "mean"),
            rolling(velocities, 10, "max"),
            rolling(motion_area, 5, "mean"),
        ]
    )
    frame_indices = [int(r["frame_index"]) for r in selected]
    timestamps = [float(r["timestamp_s"]) for r in selected]
    return X, frame_indices, timestamps


def labels_from_events(frame_indices: list[int], events: list[dict], max_seconds: float) -> np.ndarray:
    y = np.zeros(len(frame_indices), dtype=int)
    frame_to_pos = {frame: i for i, frame in enumerate(frame_indices)}
    for ev in events:
        if float(ev["contact_timestamp_s"]) > max_seconds:
            continue
        start = int(ev["start_frame"])
        end = int(ev["end_frame"])
        for frame in range(start, end + 1):
            if frame in frame_to_pos:
                y[frame_to_pos[frame]] = 1
    return y


def stratified_split(y: np.ndarray, test_ratio: float = 0.30, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.08,
    epochs: int = 6000,
    l2: float = 0.002,
) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    class_weights = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        error = (p - y) * class_weights
        grad_w = (X.T @ error) / n + l2 * w
        grad_b = float(error.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = metrics(y_true, (prob >= best_threshold).astype(int))
    for threshold in np.linspace(0.20, 0.85, 131):
        current = metrics(y_true, (prob >= threshold).astype(int))
        if (
            current["f1_score"] > best_metrics["f1_score"]
            or (
                math.isclose(current["f1_score"], best_metrics["f1_score"])
                and current["precision"] > best_metrics["precision"]
            )
        ):
            best_threshold = float(threshold)
            best_metrics = current
    return round(best_threshold, 4), best_metrics


def segments_from_binary(values: np.ndarray, min_len: int = 2, max_gap: int = 1) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    last = None
    for i, flag in enumerate(values.astype(bool)):
        if flag:
            if start is None:
                start = i
            last = i
        elif start is not None and last is not None and i - last > max_gap:
            if last - start + 1 >= min_len:
                segments.append((start, last))
            start = None
            last = None
    if start is not None and last is not None and last - start + 1 >= min_len:
        segments.append((start, last))
    return segments


def temporal_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_mask = y_true.astype(bool)
    pred_mask = y_pred.astype(bool)
    intersection = int((true_mask & pred_mask).sum())
    union = int((true_mask | pred_mask).sum())
    return round(intersection / max(1, union), 4)


def draw_metrics_card(path: Path, result: dict) -> None:
    img = Image.new("RGB", (1200, 760), "#f8fafc")
    d = ImageDraw.Draw(img)
    d.text((60, 48), "Prototype Action-Segmentation Model", fill="#0f172a", font=font(36))
    d.text((60, 98), "Frame/window classifier trained on the first 5 minutes of bartan1.MP4", fill="#475569", font=font(20))

    cards = [
        ("Accuracy", result["test_metrics"]["accuracy"]),
        ("Precision", result["test_metrics"]["precision"]),
        ("Recall", result["test_metrics"]["recall"]),
        ("F1-score", result["test_metrics"]["f1_score"]),
        ("Temporal IoU", result["temporal_iou"]),
    ]
    for i, (label, value) in enumerate(cards):
        x = 60 + (i % 3) * 360
        y = 170 + (i // 3) * 170
        d.rounded_rectangle((x, y, x + 300, y + 120), radius=10, fill="white", outline="#cbd5e1", width=2)
        d.text((x + 24, y + 24), label, fill="#64748b", font=font(18))
        d.text((x + 24, y + 58), f"{float(value) * 100:.1f}%", fill="#0f766e", font=font(34))

    cm = result["test_metrics"]
    y0 = 535
    d.text((60, y0), "Confusion matrix on held-out test frames", fill="#0f172a", font=font(22))
    rows = [
        ("True positive", cm["true_positive"]),
        ("True negative", cm["true_negative"]),
        ("False positive", cm["false_positive"]),
        ("False negative", cm["false_negative"]),
    ]
    for i, (label, value) in enumerate(rows):
        d.text((82 + (i % 2) * 360, y0 + 50 + (i // 2) * 42), f"{label}: {value}", fill="#334155", font=font(20))

    d.text((60, 700), "Note: labels are a prototype ground-truth set derived from selected interaction windows.", fill="#64748b", font=font(16))
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight action/idle segmentation baseline.")
    parser.add_argument("--output-root", default="Labellerr_outputs_bartan1")
    parser.add_argument("--max-seconds", type=float, default=300.0)
    args = parser.parse_args()

    root = Path(args.output_root)
    out_dir = root / "model_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    features = load_jsonl(root / "phase2_output" / "phase2_features.jsonl")
    events = load_jsonl(root / "phase3_output" / "phase3_events.jsonl")
    X, frame_indices, timestamps = build_features(features, args.max_seconds)
    y = labels_from_events(frame_indices, events, args.max_seconds)

    train_idx, test_idx = stratified_split(y, test_ratio=0.30, seed=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_z = (X_train - mean) / std
    X_test_z = (X_test - mean) / std
    X_all_z = (X - mean) / std

    w, b = train_logistic_regression(X_train_z, y_train)
    train_prob = sigmoid(X_train_z @ w + b)
    threshold, train_threshold_metrics = choose_threshold(y_train, train_prob)
    test_prob = sigmoid(X_test_z @ w + b)
    test_pred = (test_prob >= threshold).astype(int)
    all_prob = sigmoid(X_all_z @ w + b)
    all_pred = (all_prob >= threshold).astype(int)

    result = {
        "model": "numpy_logistic_regression_binary_action_segmenter",
        "input_video": "bartan1.MP4",
        "training_scope": f"first_{int(args.max_seconds)}_seconds",
        "label_definition": "0=idle/no interaction, 1=hand-object interaction/action",
        "label_source": "prototype ground-truth windows from phase3 interaction events; should be replaced with human labels for final reporting",
        "frames_used": int(len(y)),
        "positive_action_frames": int(y.sum()),
        "idle_frames": int(len(y) - y.sum()),
        "train_frames": int(len(train_idx)),
        "test_frames": int(len(test_idx)),
        "decision_threshold": threshold,
        "train_metrics_at_threshold": train_threshold_metrics,
        "test_metrics": metrics(y_test, test_pred),
        "temporal_iou": temporal_iou(y, all_pred),
        "true_segments_frame_index": [
            [int(frame_indices[a]), int(frame_indices[b])] for a, b in segments_from_binary(y, min_len=2)
        ],
        "predicted_segments_frame_index": [
            [int(frame_indices[a]), int(frame_indices[b])] for a, b in segments_from_binary(all_pred, min_len=2)
        ],
        "features": [
            "velocity",
            "motion_area",
            "hand_object_distance_proxy",
            "wrist_x",
            "wrist_y",
            "acceleration",
            "rolling_velocity_mean_5",
            "rolling_velocity_mean_10",
            "rolling_velocity_max_10",
            "rolling_motion_area_mean_5",
        ],
    }

    (out_dir / "model_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "action_segmenter_model.npz",
        weights=w,
        bias=np.array([b]),
        feature_mean=mean,
        feature_std=std,
    )
    with (out_dir / "predictions.csv").open("w", encoding="utf-8") as f:
        f.write("frame_index,timestamp_s,label,prediction,probability_action\n")
        for frame, ts, label, pred, prob in zip(frame_indices, timestamps, y, all_pred, all_prob):
            f.write(f"{frame},{ts:.3f},{int(label)},{int(pred)},{float(prob):.6f}\n")
    draw_metrics_card(out_dir / "model_accuracy_card.png", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
