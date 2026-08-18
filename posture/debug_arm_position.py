"""Arm-position (open/partial/closed) detection, calibrated per-user/per-camera.

Why this exists: the original posture module's arm_position classifier was trained on a public
Kaggle dataset (Confidence Detection Dataset) using a reconstructed wrist_shoulder_ratio formula
that was never verified against the dataset's real formula (only 4 of the 15 features could be
cross-checked exactly, and this wasn't one of them -- see hri-posture-recognition-new/README.md).
Live-tested against a real webcam, the mismatch is confirmed: the dataset's "Closed Arms" class
has ratio mean 0.51 (range 0.01-1.11), but live testing showed the ratio never drops below ~0.9
no matter how closed the arms actually are -- a scale mismatch, not just a wrong threshold.

Fix: instead of trusting the original dataset's scale, this script calibrates thresholds directly
against YOUR camera and YOUR body by asking you to show fully-open and fully-closed arms once at
startup, then splits that live-measured range into three even bands (Closed / Partially Open /
Open). This replaces only the arm_position decision -- head_direction, posture, and the overall
confidence_label still use the original trained classifiers, since only arm_position was shown to
be broken.

Usage: python debug_arm_position.py   (run from this folder; press q to quit)
Controls during calibration: hold arms fully OPEN and press 'o', then hold arms fully CLOSED and
press 'c'. Press 'r' any time during the main loop to redo calibration. Press 'q' to quit.
"""
import csv
import json
import os
import sys
import time
from collections import deque

import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(REPO_ROOT, 'results', 'posture_classifier.pkl')
DATA_PATH = os.path.join(REPO_ROOT, 'local_data', 'confidence_features.csv')
LOG_PATH = os.path.join(REPO_ROOT, 'results', 'debug_arm_position_predictions.csv')
CALIB_PATH = os.path.join(REPO_ROOT, 'results', 'arm_calibration.json')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

NUMERIC_COLS = [
    'eye_shoulder_y_ratio', 'shoulder_y_diff', 'wrist_distance_x', 'wrist_shoulder_ratio',
    'nose_eye_center_offset_x', 'shoulder_span', 'hip_shoulder_y_diff', 'body_lean_x',
    'shoulder_center_x', 'hip_center_x', 'spine_angle', 'eye_distance', 'head_tilt_angle',
    'eye_distance_ratio', 'shoulder_slope',
]

CALIB_SAMPLE_WINDOW = 15  # frames averaged on each calibration keypress, to reduce landmark jitter


def extract_features_v2(lm):
    """Same reconstruction as pose_module.ipynb cell 22 / live_webcam_demo.py in
    hri-posture-recognition-new."""
    l_sh, r_sh = lm[11], lm[12]
    l_hip, r_hip = lm[23], lm[24]
    l_wr, r_wr = lm[15], lm[16]
    l_eye, r_eye = lm[2], lm[5]
    nose = lm[0]

    shoulder_center_x = (l_sh.x + r_sh.x) / 2
    shoulder_center_y = (l_sh.y + r_sh.y) / 2
    hip_center_x = (l_hip.x + r_hip.x) / 2
    hip_center_y = (l_hip.y + r_hip.y) / 2
    eye_center_x = (l_eye.x + r_eye.x) / 2
    eye_center_y = (l_eye.y + r_eye.y) / 2

    shoulder_span = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y) + 1e-6
    wrist_distance_x = abs(l_wr.x - r_wr.x)
    wrist_shoulder_ratio = wrist_distance_x / shoulder_span

    eye_distance = np.hypot(l_eye.x - r_eye.x, l_eye.y - r_eye.y)
    eye_distance_ratio = eye_distance / shoulder_span

    nose_eye_center_offset_x = nose.x - eye_center_x

    shoulder_y_diff = abs(l_sh.y - r_sh.y)
    shoulder_slope = shoulder_y_diff

    hip_shoulder_y_diff = hip_center_y - shoulder_center_y
    body_lean_x = shoulder_center_x - hip_center_x

    spine_angle = 180 - np.degrees(np.arctan2(hip_shoulder_y_diff, body_lean_x + 1e-9))
    eye_shoulder_y_ratio = (eye_center_y - shoulder_center_y) / shoulder_span
    head_tilt_angle = np.degrees(np.arctan2(l_eye.y - r_eye.y, l_eye.x - r_eye.x + 1e-9))

    return {
        'eye_shoulder_y_ratio': eye_shoulder_y_ratio,
        'shoulder_y_diff': shoulder_y_diff,
        'wrist_distance_x': wrist_distance_x,
        'wrist_shoulder_ratio': wrist_shoulder_ratio,
        'nose_eye_center_offset_x': nose_eye_center_offset_x,
        'shoulder_span': shoulder_span,
        'hip_shoulder_y_diff': hip_shoulder_y_diff,
        'body_lean_x': body_lean_x,
        'shoulder_center_x': shoulder_center_x,
        'hip_center_x': hip_center_x,
        'spine_angle': spine_angle,
        'eye_distance': eye_distance,
        'head_tilt_angle': head_tilt_angle,
        'eye_distance_ratio': eye_distance_ratio,
        'shoulder_slope': shoulder_slope,
    }


def train_aux_models(df):
    """head_direction and posture aux classifiers -- unchanged, no evidence these are broken.
    arm_position is intentionally NOT included here anymore; see calibrated_arm_position() below."""
    aux_models = {}
    for target in ['head_direction', 'posture']:
        aux_le = LabelEncoder()
        aux_y = aux_le.fit_transform(df[target])
        aux_clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        aux_clf.fit(df[NUMERIC_COLS], aux_y)
        aux_models[target] = (aux_clf, aux_le)
    return aux_models


def calibrated_arm_position(ratio, closed_ratio, open_ratio):
    """Three even bands across the user's own calibrated [closed, open] range, instead of
    trusting the original dataset's (empirically wrong-scale) thresholds."""
    lo = closed_ratio + (open_ratio - closed_ratio) / 3
    hi = closed_ratio + 2 * (open_ratio - closed_ratio) / 3
    if ratio <= lo:
        return 'Closed Arms'
    elif ratio <= hi:
        return 'Partially Open'
    return 'Open Arms'


def draw_fitted_text(frame, text, origin, max_width, base_scale, color=(0, 255, 0), min_scale=0.3):
    scale = base_scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    while scale > min_scale:
        (text_w, _), _ = cv2.getTextSize(text, font, scale, 2)
        if text_w <= max_width:
            break
        scale -= 0.05
    cv2.putText(frame, text, origin, font, scale, color, 2)


def get_live_crop(cap, pose, target_shoulder_frac=0.4, max_warmup_frames=90):
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for _ in range(max_warmup_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            l_sh, r_sh = lm[11], lm[12]
            l_hip, r_hip = lm[23], lm[24]
            nose = lm[0]
            shoulder_px_width = abs(l_sh.x - r_sh.x) * frame_w
            if shoulder_px_width < 1:
                continue
            crop_w = shoulder_px_width / target_shoulder_frac

            cx = (l_sh.x + r_sh.x) / 2 * frame_w
            head_top_px = min(nose.y, l_sh.y, r_sh.y) * frame_h
            hip_bottom_px = max(l_hip.y, r_hip.y) * frame_h
            headroom = 0.15 * crop_w
            top_y = head_top_px - headroom
            crop_h = (hip_bottom_px - top_y) + 0.1 * crop_w

            x0 = int(max(0, cx - crop_w / 2))
            x1 = int(min(frame_w, cx + crop_w / 2))
            y0 = int(max(0, top_y))
            y1 = int(min(frame_h, top_y + crop_h))
            if x1 - x0 > 10 and y1 - y0 > 10:
                return x0, y0, x1, y1
    return 0, 0, frame_w, frame_h


def main():
    print('Loading posture model + aux classifiers...')
    bundle = joblib.load(MODEL_PATH)
    clf, le, feature_cols = bundle['model'], bundle['label_encoder'], bundle['feature_cols']
    df = pd.read_csv(DATA_PATH)
    aux_models = train_aux_models(df)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Could not open webcam (index 0).', file=sys.stderr)
        sys.exit(1)

    calib_pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
    print('Calibrating crop box from your current position, hold still...')
    x0, y0, x1, y1 = get_live_crop(cap, calib_pose)
    calib_pose.close()
    print(f'crop box: x[{x0}:{x1}] y[{y0}:{y1}]')

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_rows = []
    frame_idx = 0
    t0 = time.time()

    ratio_buffer = deque(maxlen=CALIB_SAMPLE_WINDOW)
    closed_ratio, open_ratio = None, None
    calibrating = True

    if os.path.isfile(CALIB_PATH):
        with open(CALIB_PATH) as f:
            saved = json.load(f)
        closed_ratio, open_ratio = saved['closed_ratio'], saved['open_ratio']
        calibrating = False
        print(f'Loaded saved calibration from {CALIB_PATH}: closed={closed_ratio:.3f}  '
              f"open={open_ratio:.3f}. Press 'r' to redo. Press 'q' to quit.")
    else:
        print("Calibration: hold arms fully OPEN, press 'o'. Then hold fully CLOSED, press 'c'.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        crop = frame[y0:y1, x0:x1]
        live_ratio = None
        feats = None

        if crop.size > 0:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                feats = extract_features_v2(lm)
                live_ratio = feats['wrist_shoulder_ratio']
                ratio_buffer.append(live_ratio)
                mp_drawing.draw_landmarks(crop, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                frame[y0:y1, x0:x1] = crop

        frame_w = frame.shape[1]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 1)
        cv2.rectangle(frame, (0, 0), (frame_w, 96), (0, 0, 0), -1)

        key = cv2.waitKey(1) & 0xFF

        if calibrating:
            ratio_txt = f"{live_ratio:.3f}" if live_ratio is not None else "no pose"
            draw_fitted_text(frame, f"CALIBRATING  live wrist_shoulder_ratio={ratio_txt}",
                              (8, 22), frame_w - 16, 0.55, (0, 200, 255))
            draw_fitted_text(frame, "Hold arms fully OPEN, press 'o'" if open_ratio is None
                              else f"open_ratio captured = {open_ratio:.3f}",
                              (8, 44), frame_w - 16, 0.5,
                              (0, 255, 0) if open_ratio is None else (150, 150, 150))
            draw_fitted_text(frame, "Then hold fully CLOSED, press 'c'" if closed_ratio is None
                              else f"closed_ratio captured = {closed_ratio:.3f}",
                              (8, 66), frame_w - 16, 0.5,
                              (0, 255, 0) if closed_ratio is None else (150, 150, 150))

            if key == ord('o') and ratio_buffer:
                open_ratio = float(np.mean(ratio_buffer))
                print(f'open_ratio calibrated: {open_ratio:.3f}')
            elif key == ord('c') and ratio_buffer:
                closed_ratio = float(np.mean(ratio_buffer))
                print(f'closed_ratio calibrated: {closed_ratio:.3f}')

            if open_ratio is not None and closed_ratio is not None:
                if open_ratio <= closed_ratio:
                    print('WARNING: open_ratio <= closed_ratio, recalibrate -- press o/c again.')
                    open_ratio, closed_ratio = None, None
                else:
                    calibrating = False
                    with open(CALIB_PATH, 'w') as f:
                        json.dump({'closed_ratio': closed_ratio, 'open_ratio': open_ratio}, f)
                    print(f'Calibration done and saved to {CALIB_PATH}: closed={closed_ratio:.3f}  '
                          f'open={open_ratio:.3f}. '
                          "Press 'r' any time to redo. Press 'q' to quit.")
        else:
            label_line1, label_line2 = 'no person detected in crop', ''
            if feats is not None:
                row = pd.DataFrame([feats])[NUMERIC_COLS]
                for target, (aux_clf, aux_le) in aux_models.items():
                    pred = aux_clf.predict(row)[0]
                    feats[target] = aux_le.inverse_transform([pred])[0]
                feats['arm_position'] = calibrated_arm_position(live_ratio, closed_ratio, open_ratio)

                cat_df = pd.DataFrame([{k: feats[k] for k in ['head_direction', 'arm_position', 'posture']}]).astype(str)
                X_live = pd.concat([row, pd.get_dummies(cat_df)], axis=1).reindex(columns=feature_cols, fill_value=0)
                pred_id = clf.predict(X_live)[0]
                confidence_label = le.inverse_transform([pred_id])[0]

                label_line1 = confidence_label
                label_line2 = (f"arms={feats['arm_position']} (ratio={live_ratio:.3f})  "
                                f"head={feats['head_direction']}  posture={feats['posture']}")

                log_rows.append({
                    'frame': frame_idx, 't': round(time.time() - t0, 2),
                    'confidence_label': confidence_label, 'arm_position': feats['arm_position'],
                    'wrist_shoulder_ratio': live_ratio,
                    'closed_ratio_calib': closed_ratio, 'open_ratio_calib': open_ratio,
                    'head_direction': feats['head_direction'], 'posture': feats['posture'],
                })

            draw_fitted_text(frame, label_line1, (8, 22), frame_w - 16, 0.65)
            if label_line2:
                draw_fitted_text(frame, label_line2, (8, 44), frame_w - 16, 0.5)
            draw_fitted_text(frame, f"calibrated: closed<={closed_ratio + (open_ratio - closed_ratio) / 3:.2f}  "
                                     f"open>={closed_ratio + 2 * (open_ratio - closed_ratio) / 3:.2f}  "
                                     "('r' to redo)",
                              (8, 88), frame_w - 16, 0.42, (0, 200, 255))

            if key == ord('r'):
                calibrating = True
                open_ratio, closed_ratio = None, None
                print("Recalibrating: hold arms fully OPEN, press 'o'. Then fully CLOSED, press 'c'.")

        cv2.imshow('Arm-position debug - live webcam (q to quit)', frame)
        frame_idx += 1

        if key == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    if log_rows:
        with open(LOG_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        counts = pd.DataFrame(log_rows)['arm_position'].value_counts().to_dict()
        print(f'\nSaved {len(log_rows)} predicted frames to {LOG_PATH}')
        print('arm_position distribution:', counts)
    else:
        print('\nNo frames were logged (calibration may not have completed).')


if __name__ == '__main__':
    main()
