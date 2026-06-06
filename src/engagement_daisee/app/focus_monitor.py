from __future__ import annotations

import argparse
import collections
import math
import time
from dataclasses import dataclass
from typing import Deque, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    raise RuntimeError("mediapipe is required for webcam focus monitor") from exc


# FaceMesh landmark indices used for robust, lightweight proxies.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
NOSE_TIP = 1
CHIN = 152
UPPER_LIP = 13
LOWER_LIP = 14


@dataclass
class MonitorConfig:
    calibration_seconds: float = 30.0
    smoothing_alpha: float = 0.20
    distracted_enter: float = 0.62
    distracted_exit: float = 0.48
    dwell_seconds: float = 1.2
    perclos_window_seconds: float = 60.0
    blink_ear_threshold: float = 0.21


class FocusStateMachine:
    def __init__(self, cfg: MonitorConfig) -> None:
        self.cfg = cfg
        self.state = "focused"
        self._pending_since: Optional[float] = None

    def update(self, distracted_score: float, now: float) -> str:
        if self.state == "focused":
            if distracted_score >= self.cfg.distracted_enter:
                if self._pending_since is None:
                    self._pending_since = now
                elif now - self._pending_since >= self.cfg.dwell_seconds:
                    self.state = "distracted"
                    self._pending_since = None
            else:
                self._pending_since = None
        else:
            if distracted_score <= self.cfg.distracted_exit:
                if self._pending_since is None:
                    self._pending_since = now
                elif now - self._pending_since >= self.cfg.dwell_seconds:
                    self.state = "focused"
                    self._pending_since = None
            else:
                self._pending_since = None
        return self.state


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) + 1e-6)


def _eye_aspect_ratio(pts: np.ndarray, idxs: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idxs]
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4))


def _normed(points: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.copy(points)
    out[:, 0] *= width
    out[:, 1] *= height
    out[:, 2] *= width
    return out


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class FocusMonitor:
    def __init__(self, cfg: MonitorConfig) -> None:
        self.cfg = cfg
        self.started = time.time()
        self.state_machine = FocusStateMachine(cfg)
        self.smooth_score = 0.5
        self.calib_values: list[np.ndarray] = []
        self.calib_mean: Optional[np.ndarray] = None
        self.calib_std: Optional[np.ndarray] = None
        self.eye_closed_hist: Deque[tuple[float, bool]] = collections.deque()

    def _extract_raw(self, pts: np.ndarray, w: int, h: int) -> np.ndarray:
        px = _normed(pts, w, h)

        left_ear = _eye_aspect_ratio(px, LEFT_EYE)
        right_ear = _eye_aspect_ratio(px, RIGHT_EYE)
        ear = 0.5 * (left_ear + right_ear)

        # Iris offset from eye center -> gaze proxy.
        l_iris = np.mean(px[LEFT_IRIS, :2], axis=0)
        l_eye_center = 0.5 * (px[33, :2] + px[133, :2])
        r_iris = np.mean(px[RIGHT_IRIS, :2], axis=0)
        r_eye_center = 0.5 * (px[362, :2] + px[263, :2])
        gaze_offset = 0.5 * (
            np.linalg.norm(l_iris - l_eye_center) / (_dist(px[33, :2], px[133, :2]))
            + np.linalg.norm(r_iris - r_eye_center) / (_dist(px[362, :2], px[263, :2]))
        )

        # Head pose proxy: nose/chin vector deviation from vertical center line.
        nose = px[NOSE_TIP, :2]
        chin = px[CHIN, :2]
        v = chin - nose
        head_tilt = abs(math.atan2(v[0], v[1] + 1e-6))  # yaw-ish proxy in radians

        # Mouth dynamics proxy for expression activity.
        lip_open = _dist(px[UPPER_LIP, :2], px[LOWER_LIP, :2]) / (_dist(px[33, :2], px[263, :2]))

        return np.array([ear, gaze_offset, head_tilt, lip_open], dtype=np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.calib_mean is None or self.calib_std is None:
            return x
        return (x - self.calib_mean) / (self.calib_std + 1e-6)

    def update(self, x: np.ndarray) -> tuple[float, float, float, str]:
        now = time.time()

        if now - self.started < self.cfg.calibration_seconds:
            self.calib_values.append(x)
            return 0.5, 0.0, 0.0, "calibrating"

        if self.calib_mean is None and self.calib_values:
            arr = np.stack(self.calib_values, axis=0)
            self.calib_mean = arr.mean(axis=0)
            self.calib_std = arr.std(axis=0)

        nx = self._normalize(x)
        ear, gaze_off, head_tilt, lip_open = x

        eye_closed = ear < self.cfg.blink_ear_threshold
        self.eye_closed_hist.append((now, eye_closed))
        while self.eye_closed_hist and now - self.eye_closed_hist[0][0] > self.cfg.perclos_window_seconds:
            self.eye_closed_hist.popleft()
        perclos = sum(int(flag) for _, flag in self.eye_closed_hist) / max(1, len(self.eye_closed_hist))

        # Weighted fusion score in [0,1]: larger = more distracted.
        score = (
            0.34 * _clip01((gaze_off - 0.22) / 0.35)
            + 0.24 * _clip01((head_tilt - 0.20) / 0.50)
            + 0.26 * _clip01((perclos - 0.15) / 0.45)
            + 0.16 * _clip01((abs(nx[3]) - 0.6) / 1.8)
        )

        self.smooth_score = self.cfg.smoothing_alpha * score + (1.0 - self.cfg.smoothing_alpha) * self.smooth_score
        state = self.state_machine.update(self.smooth_score, now)
        return float(self.smooth_score), float(perclos), float(ear), state


def run_webcam(camera_id: int, cfg: MonitorConfig) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam id={camera_id}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    mon = FocusMonitor(cfg)
    print("[focus-monitor] started | press q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        label = "no-face"
        color = (0, 140, 255)
        score = 0.5
        perclos = 0.0
        ear = 0.0

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            raw = mon._extract_raw(pts, w, h)
            score, perclos, ear, state = mon.update(raw)
            label = state
            color = (0, 220, 0) if state in ("focused", "calibrating") else (0, 0, 255)

        cv2.putText(frame, f"state: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"distracted_score: {score:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(frame, f"perclos: {perclos:.2f} ear: {ear:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

        if label == "distracted":
            cv2.putText(frame, "WARNING: Please refocus", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Focus Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime webcam focus monitor with multi-signal fusion.")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--calibration-seconds", type=float, default=30.0)
    p.add_argument("--distracted-enter", type=float, default=0.62)
    p.add_argument("--distracted-exit", type=float, default=0.48)
    p.add_argument("--dwell-seconds", type=float, default=1.2)
    p.add_argument("--smoothing-alpha", type=float, default=0.20)
    p.add_argument("--blink-ear-threshold", type=float, default=0.21)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MonitorConfig(
        calibration_seconds=args.calibration_seconds,
        distracted_enter=args.distracted_enter,
        distracted_exit=args.distracted_exit,
        dwell_seconds=args.dwell_seconds,
        smoothing_alpha=args.smoothing_alpha,
        blink_ear_threshold=args.blink_ear_threshold,
    )
    run_webcam(camera_id=args.camera_id, cfg=cfg)


if __name__ == "__main__":
    main()
