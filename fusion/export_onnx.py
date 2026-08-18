"""Exports the trained fusion MLP (results/fusion_mlp.pkl) to ONNX for on-device mobile inference.

Usage: python export_onnx.py
"""
import json
import os

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(REPO_ROOT, 'results', 'fusion_mlp.pkl')
ONNX_PATH = os.path.join(REPO_ROOT, 'results', 'fusion_mlp.onnx')
META_PATH = os.path.join(REPO_ROOT, 'results', 'fusion_mlp_meta.json')


def main():
    bundle = joblib.load(MODEL_PATH)
    clf, le = bundle['model'], bundle['label_encoder']
    n_features = len(bundle['face_classes']) + len(bundle['posture_classes'])

    onnx_model = to_onnx(
        clf, initial_types=[('input', FloatTensorType([None, n_features]))],
        target_opset=17, options={id(clf): {'zipmap': False}},
    )
    with open(ONNX_PATH, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f'Exported to {ONNX_PATH}')

    rng = np.random.default_rng(0)
    face_probs = rng.dirichlet(np.full(len(bundle['face_classes']), 1.0), size=200)
    posture_probs = rng.dirichlet(np.full(len(bundle['posture_classes']), 1.0), size=200)
    X_sample = np.hstack([face_probs, posture_probs]).astype(np.float32)

    sk_pred = clf.predict(X_sample)
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    onnx_pred = sess.run(None, {'input': X_sample})[0]
    agree = (sk_pred == onnx_pred).mean()
    print(f'sklearn/ONNX prediction agreement on random sample: {agree:.4f}')
    assert agree > 0.99, 'ONNX export mismatch -- do not ship this model'

    with open(META_PATH, 'w') as f:
        json.dump({
            'face_classes': bundle['face_classes'],
            'posture_classes': bundle['posture_classes'],
            'fused_labels': list(le.classes_),
            'input_order': 'concat(face_probs, posture_probs)',
        }, f, indent=2)
    print(f'Saved metadata to {META_PATH}')
    print('fused_labels (index -> name):', dict(enumerate(le.classes_)))


if __name__ == '__main__':
    main()
