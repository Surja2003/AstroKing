from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import os
import threading
import time


@dataclass(frozen=True)
class PalmCvResult:
    status: str
    features: dict[str, Any]
    traits: list[str]
    overlay_jpeg_bytes: Optional[bytes] = None


@dataclass(frozen=True)
class FrameQualityResult:
    status: str
    message: str
    metrics: dict[str, Any]


_mp = None

_landmarker_image_lock = threading.Lock()
_landmarker_image = None
_landmarker_video_lock = threading.Lock()
_landmarker_video = None

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def _lazy_import_cv():
    import numpy as np  # type: ignore
    import cv2  # type: ignore

    return np, cv2


def _lazy_import_mediapipe():
    global _mp
    if _mp is None:
        import mediapipe as mp  # type: ignore

        _mp = mp
    return _mp


def _ensure_hand_landmarker_model() -> str:
    """Ensure the HandLandmarker .task model exists locally (download if missing)."""

    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "mp_models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "hand_landmarker.task")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    import requests  # type: ignore

    tmp_path = model_path + ".part"
    last_err: Optional[Exception] = None

    for _ in range(2):
        try:
            with requests.get(_MODEL_URL, stream=True, timeout=(10, 180)) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, model_path)
            last_err = None
            break
        except Exception as e:
            last_err = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    if last_err is not None:
        raise RuntimeError(
            "Failed to download MediaPipe HandLandmarker model. "
            "If you're offline, manually download this file and place it at: "
            f"{model_path} (URL: {_MODEL_URL})"
        ) from last_err

    return model_path


def _get_hand_landmarker_image():
    """Singleton HandLandmarker in IMAGE mode."""
    global _landmarker_image
    if _landmarker_image is not None:
        return _landmarker_image

    with _landmarker_image_lock:
        if _landmarker_image is not None:
            return _landmarker_image

        mp = _lazy_import_mediapipe()
        model_path = _ensure_hand_landmarker_model()

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        _landmarker_image = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return _landmarker_image


def _get_hand_landmarker_video():
    """Singleton HandLandmarker in VIDEO mode (best for preview frame polling)."""
    global _landmarker_video
    if _landmarker_video is not None:
        return _landmarker_video

    with _landmarker_video_lock:
        if _landmarker_video is not None:
            return _landmarker_video

        mp = _lazy_import_mediapipe()
        model_path = _ensure_hand_landmarker_model()

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        _landmarker_video = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return _landmarker_video


def _decode_bgr(image_bytes: bytes):
    np, cv2 = _lazy_import_cv()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _compute_quality_metrics(bgr_img) -> dict[str, float]:
    np, cv2 = _lazy_import_cv()
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    mean_gray = float(np.mean(gray))

    # Focus/blur proxy
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"mean_gray": mean_gray, "sharpness": sharpness}


def _auto_enhance_bgr(bgr_img):
    """Simple lighting normalization: histogram equalization on L channel in LAB."""
    _, cv2 = _lazy_import_cv()
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def assess_frame_quality(
    image_bytes: bytes,
    *,
    min_brightness: float = 60.0,
    min_sharpness: float = 100.0,
    min_hand_area_ratio: float = 0.25,
) -> FrameQualityResult:
    """Fast gate for camera preview frames.

    Returns a small status + message that the app can use to enable/disable Capture.
    """

    _, cv2 = _lazy_import_cv()
    _lazy_import_mediapipe()  # Ensure it can import.

    bgr = _decode_bgr(image_bytes)
    qm = _compute_quality_metrics(bgr)

    if qm["mean_gray"] < min_brightness:
        return FrameQualityResult(
            status="too_dark",
            message="Too dark — move to brighter light",
            metrics=qm,
        )

    if qm["sharpness"] < min_sharpness:
        return FrameQualityResult(
            status="too_blurry",
            message="Too blurry — hold the phone steady",
            metrics=qm,
        )

    enhanced = _auto_enhance_bgr(bgr)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    mp = _lazy_import_mediapipe()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker = _get_hand_landmarker_video()
    ts_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    if not getattr(result, "hand_landmarks", None):
        return FrameQualityResult(
            status="no_hand_detected",
            message="No hand detected — place your palm in the frame",
            metrics=qm,
        )

    hand_landmarks = result.hand_landmarks[0]
    h, w = enhanced.shape[:2]
    x1, y1, x2, y2 = _landmark_bbox(hand_landmarks, w=w, h=h, pad_ratio=0.10)

    hand_area = float(max(1, (x2 - x1) * (y2 - y1)))
    frame_area = float(max(1, w * h))
    area_ratio = float(hand_area / frame_area)

    metrics = {**qm, "hand_area_ratio": area_ratio}

    if area_ratio < min_hand_area_ratio:
        return FrameQualityResult(
            status="hand_too_small",
            message="Move your hand closer — fill more of the frame",
            metrics=metrics,
        )

    return FrameQualityResult(
        status="ok",
        message="Ready",
        metrics=metrics,
    )


def detect_hand_live(
    image_bytes: bytes,
    *,
    min_hand_area_ratio: float = 0.25,
) -> dict[str, Any]:
    """Ultra-lightweight hand detection for preview-frame gating.

    Returns normalized bbox coordinates so the client can draw an overlay easily.
    """

    _, cv2 = _lazy_import_cv()
    mp = _lazy_import_mediapipe()

    bgr = _decode_bgr(image_bytes)
    h, w = bgr.shape[:2]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker = _get_hand_landmarker_video()
    ts_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    if not getattr(result, "hand_landmarks", None):
        return {"detected": False, "reason": "no_hand"}

    hand_landmarks = result.hand_landmarks[0]

    # Normalized bbox from landmarks (no padding to keep box stable).
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    xmin = max(0.0, min(xs))
    xmax = min(1.0, max(xs))
    ymin = max(0.0, min(ys))
    ymax = min(1.0, max(ys))

    area_ratio = float(max(0.0, (xmax - xmin) * (ymax - ymin)))
    if area_ratio < min_hand_area_ratio:
        return {"detected": False, "reason": "move_closer", "hand_area_ratio": area_ratio}

    # Use handedness score as a proxy confidence when available.
    confidence = None
    handedness = None
    try:
        if getattr(result, "handedness", None) and len(result.handedness) > 0:
            confidence = float(result.handedness[0][0].score)
            handedness = result.handedness[0][0].category_name
    except Exception:
        confidence = None
        handedness = None

    return {
        "detected": True,
        "confidence": confidence,
        "hand_area_ratio": area_ratio,
        "box": {"x1": xmin, "y1": ymin, "x2": xmax, "y2": ymax},
        "frame": {"width": int(w), "height": int(h)},
        "handedness": handedness,
        "engine": "mediapipe_tasks_hand_landmarker_video",
    }


def _landmark_bbox(hand_landmarks, *, w: int, h: int, pad_ratio: float = 0.15):
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    xmin = max(0.0, min(xs))
    xmax = min(1.0, max(xs))
    ymin = max(0.0, min(ys))
    ymax = min(1.0, max(ys))

    # Pad in normalized coords.
    pad_x = (xmax - xmin) * pad_ratio
    pad_y = (ymax - ymin) * pad_ratio

    xmin = max(0.0, xmin - pad_x)
    xmax = min(1.0, xmax + pad_x)
    ymin = max(0.0, ymin - pad_y)
    ymax = min(1.0, ymax + pad_y)

    x1 = int(xmin * w)
    x2 = int(xmax * w)
    y1 = int(ymin * h)
    y2 = int(ymax * h)

    # Clamp and ensure valid slice.
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    return x1, y1, x2, y2


def _extract_geometric_features(hand_landmarks) -> dict[str, float]:
    # MediaPipe indices: wrist=0, thumb_cmc=1, index_mcp=5, index_tip=8, ring_tip=16, pinky_mcp=17
    wrist = hand_landmarks[0]
    thumb_base = hand_landmarks[1]
    index_mcp = hand_landmarks[5]
    index_tip = hand_landmarks[8]
    ring_tip = hand_landmarks[16]
    pinky_base = hand_landmarks[17]

    palm_width = abs(thumb_base.x - pinky_base.x)
    finger_spread = abs(index_mcp.x - pinky_base.x)

    # Tip-to-wrist vertical delta gives a rough finger length proxy (normalized).
    index_len = abs(index_tip.y - wrist.y)
    ring_len = abs(ring_tip.y - wrist.y)
    finger_ratio = float(index_len - ring_len)

    return {
        "palm_width": float(palm_width),
        "finger_spread": float(finger_spread),
        "finger_ratio": finger_ratio,
    }


def _traits_from_features(features: dict[str, Any]) -> list[str]:
    traits: list[str] = []

    pw = float(features.get("palm_width", 0.0))
    spread = float(features.get("finger_spread", 0.0))
    ratio = float(features.get("finger_ratio", 0.0))

    # Transparent heuristics (keep this grounded and non-medical).
    if pw >= 0.25:
        traits.append("Strong presence and confident energy")
    else:
        traits.append("Steady and measured approach")

    if ratio >= 0.02:
        traits.append("Analytical and strategic thinking style")
    else:
        traits.append("Intuitive and feeling-led thinking style")

    if spread >= 0.20:
        traits.append("Independent and open-minded")

    # Add CV quality hints if present.
    if isinstance(features.get("mean_gray"), (int, float)) and features["mean_gray"] < 60:
        traits.append("Tip: brighter lighting improves scan accuracy")

    return traits


def analyze_palm_image(
    image_bytes: bytes,
    *,
    return_overlay: bool = False,
) -> PalmCvResult:
    """Detect a hand using MediaPipe Hands and extract a few stable, interpretable features.

    This does NOT attempt medical inference; it returns simple, user-facing "traits" based on
    geometric ratios.

    If MediaPipe/OpenCV are not installed, raises ImportError.
    """

    np, cv2 = _lazy_import_cv()
    mp = _lazy_import_mediapipe()

    bgr = _decode_bgr(image_bytes)
    qm = _compute_quality_metrics(bgr)

    # Gate 1: brightness / blur (prevents low-confidence scans).
    if qm["mean_gray"] < 60.0:
        return PalmCvResult(
            status="lighting_too_low",
            features={"mean_gray": qm["mean_gray"], "sharpness": qm["sharpness"]},
            traits=["Lighting too low — move to a brighter area"],
            overlay_jpeg_bytes=None,
        )

    if qm["sharpness"] < 100.0:
        return PalmCvResult(
            status="too_blurry",
            features={"mean_gray": qm["mean_gray"], "sharpness": qm["sharpness"]},
            traits=["Image blurry — hold the phone steady"],
            overlay_jpeg_bytes=None,
        )

    # Auto-enhance before detection.
    bgr = _auto_enhance_bgr(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker = _get_hand_landmarker_image()
    result = landmarker.detect(mp_image)

    if not getattr(result, "hand_landmarks", None):
        return PalmCvResult(
            status="no_hand_detected",
            features={"mean_gray": qm["mean_gray"], "sharpness": qm["sharpness"]},
            traits=["No hand detected — fill the frame with your palm"],
            overlay_jpeg_bytes=None,
        )

    hand_landmarks = result.hand_landmarks[0]

    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _landmark_bbox(hand_landmarks, w=w, h=h, pad_ratio=0.18)

    # Gate 2: ensure palm/hand fills enough of the frame.
    hand_area = float(max(1, (x2 - x1) * (y2 - y1)))
    frame_area = float(max(1, w * h))
    area_ratio = float(hand_area / frame_area)
    if area_ratio < 0.25:
        return PalmCvResult(
            status="hand_too_small",
            features={"mean_gray": qm["mean_gray"], "sharpness": qm["sharpness"], "hand_area_ratio": area_ratio},
            traits=["Move your hand closer — fill more of the frame"],
            overlay_jpeg_bytes=None,
        )

    # Compute edge/line texture on the cropped hand ROI for less background noise.
    roi = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    edge_count = int(np.sum(edges > 0))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=30,
        maxLineGap=8,
    )
    line_count = int(0 if lines is None else len(lines))

    pixels = float(max(1, edges.shape[0] * edges.shape[1]))
    density_score = float(edge_count / pixels)

    geometric = _extract_geometric_features(hand_landmarks)

    features: dict[str, Any] = {
        **geometric,
        "mean_gray": qm["mean_gray"],
        "sharpness": qm["sharpness"],
        "hand_area_ratio": area_ratio,
        "roi": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "edge_count": edge_count,
        "line_count": line_count,
        "density_score": density_score,
        "image_size": {"width": int(w), "height": int(h)},
    }

    traits = _traits_from_features(features)

    overlay_bytes: Optional[bytes] = None
    if return_overlay:
        # MediaPipe Tasks drawing expects RGB.
        overlay_rgb = cv2.cvtColor(bgr.copy(), cv2.COLOR_BGR2RGB)
        mp_drawing = mp.tasks.vision.drawing_utils
        mp_styles = mp.tasks.vision.drawing_styles
        mp_hands = mp.tasks.vision.HandLandmarksConnections

        try:
            mp_drawing.draw_landmarks(
                overlay_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
        except Exception:
            pass

        overlay = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        # Draw ROI bbox.
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (92, 107, 192), 2)

        ok_enc, buffer = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok_enc:
            overlay_bytes = buffer.tobytes()

    return PalmCvResult(
        status="success",
        features=features,
        traits=traits,
        overlay_jpeg_bytes=overlay_bytes,
    )
