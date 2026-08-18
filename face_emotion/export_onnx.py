"""Exports the trained face-emotion CNN (results/face_emotion_classifier.pt) to ONNX, so it can
run on-device in the Flutter app via an ONNX Runtime plugin instead of needing PyTorch on mobile.

Usage: python export_onnx.py
"""
import os

import numpy as np
import onnxruntime as ort
import torch

from model import EmotionCNN

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(REPO_ROOT, 'results', 'face_emotion_classifier.pt')
ONNX_PATH = os.path.join(REPO_ROOT, 'results', 'face_emotion_classifier.onnx')


def main():
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    class_names = ckpt['class_names']
    model = EmotionCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    dummy_input = torch.randn(1, 3, 96, 96)
    torch.onnx.export(
        model, dummy_input, ONNX_PATH,
        input_names=['image'], output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17, dynamo=False,
    )
    print(f'Exported to {ONNX_PATH}')
    print('Class order (index -> name):', dict(enumerate(class_names)))

    # Sanity check: PyTorch and ONNX Runtime must agree on the same random input, otherwise the
    # export is silently wrong and everything built on top of it (mobile inference) would be too.
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {'image': dummy_input.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f'Max abs diff between PyTorch and ONNX Runtime outputs: {max_diff:.2e}')
    assert max_diff < 1e-4, 'ONNX export mismatch -- do not use this model on mobile until fixed'
    print('Sanity check passed: ONNX output matches PyTorch output.')


if __name__ == '__main__':
    main()
