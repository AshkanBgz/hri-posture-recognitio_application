"""Exports the posture pipeline's RandomForest classifiers to ONNX for on-device mobile inference.

Three models (arm_position is NOT included -- it was replaced by the per-user ratio calibration
in debug_arm_position.py after the original RF-based arm_position classifier was found to not
transfer to live webcam input; see that script's docstring):
  1. confidence_classifier.onnx  -- the main model (25 features: 15 numeric + one-hot
     head_direction/arm_position/posture) -> Confident/Neutral/Low
  2. head_direction_classifier.onnx -- aux model (15 numeric features) -> head direction category
  3. posture_classifier_aux.onnx    -- aux model (15 numeric features) -> Upright/Slouched/Stiff

The aux classifiers were never persisted to disk in the original project (only retrained in-memory
each time a demo script ran -- see train_aux_models() in combined_demo.py/live_webcam_demo.py), so
this script retrains them once here before exporting.

Usage: python export_onnx.py
"""
import os

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(REPO_ROOT, 'results', 'posture_classifier.pkl')
DATA_PATH = os.path.join(REPO_ROOT, 'local_data', 'confidence_features.csv')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')

NUMERIC_COLS = [
    'eye_shoulder_y_ratio', 'shoulder_y_diff', 'wrist_distance_x', 'wrist_shoulder_ratio',
    'nose_eye_center_offset_x', 'shoulder_span', 'hip_shoulder_y_diff', 'body_lean_x',
    'shoulder_center_x', 'hip_center_x', 'spine_angle', 'eye_distance', 'head_tilt_angle',
    'eye_distance_ratio', 'shoulder_slope',
]


def export_and_verify(sk_model, X_sample, name, extra_meta=None):
    onnx_path = os.path.join(RESULTS_DIR, f'{name}.onnx')
    onnx_model = to_onnx(
        sk_model, initial_types=[('input', FloatTensorType([None, X_sample.shape[1]]))],
        target_opset=17, options={id(sk_model): {'zipmap': False}},
    )
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    sk_pred = sk_model.predict(X_sample)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_pred = sess.run(None, {'input': X_sample.astype(np.float32)})[0]

    agree = (sk_pred == onnx_pred).mean()
    print(f'{name}: exported to {onnx_path}  (sklearn/ONNX prediction agreement on sample: {agree:.4f})')
    assert agree > 0.99, f'{name}: ONNX export mismatch, do not ship this model'
    if extra_meta:
        print(f'  {extra_meta}')
    return onnx_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('Loading main confidence classifier + dataset...')
    bundle = joblib.load(MODEL_PATH)
    clf, le, feature_cols = bundle['model'], bundle['label_encoder'], bundle['feature_cols']
    df = pd.read_csv(DATA_PATH)

    print('Retraining aux classifiers (head_direction, posture) -- these were only ever kept '
          'in-memory in the original demo scripts...')
    aux_encoders = {}
    for target in ['head_direction', 'posture']:
        aux_le = LabelEncoder()
        aux_y = aux_le.fit_transform(df[target])
        aux_clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        aux_clf.fit(df[NUMERIC_COLS], aux_y)
        aux_encoders[target] = aux_le

        X_sample = df[NUMERIC_COLS].sample(n=min(500, len(df)), random_state=1).values.astype(np.float32)
        export_and_verify(aux_clf, X_sample, f'{target}_classifier_aux',
                           extra_meta=f'classes: {dict(enumerate(aux_le.classes_))}')

    print('Exporting main confidence classifier (25 features: 15 numeric + one-hot categoricals)...')
    # Build a representative sample of the full 25-dim feature vector by re-deriving it the same
    # way the live demos do: numeric cols as-is + one-hot of the three categorical columns,
    # reindexed to the exact training feature_cols order (missing dummy columns filled with 0).
    cat_df = df[['head_direction', 'arm_position', 'posture']].astype(str)
    X_full = pd.concat([df[NUMERIC_COLS], pd.get_dummies(cat_df)], axis=1).reindex(
        columns=feature_cols, fill_value=0)
    X_sample = X_full.sample(n=min(500, len(X_full)), random_state=1).values.astype(np.float32)
    export_and_verify(clf, X_sample, 'confidence_classifier',
                       extra_meta=f'classes: {dict(enumerate(le.classes_))}  feature_cols order saved separately')

    # feature_cols order matters for building the live 25-dim input vector on-device -- save it
    # so the Dart code builds the one-hot vector in the exact same column order.
    import json
    meta_path = os.path.join(RESULTS_DIR, 'confidence_classifier_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'feature_cols': feature_cols,
            'confidence_classes': list(le.classes_),
            'head_direction_classes': list(aux_encoders['head_direction'].classes_),
            'posture_classes': list(aux_encoders['posture'].classes_),
            'numeric_cols': NUMERIC_COLS,
        }, f, indent=2)
    print(f'Saved feature/class metadata to {meta_path}')


if __name__ == '__main__':
    main()
