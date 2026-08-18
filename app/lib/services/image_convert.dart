// Converts a raw camera frame (YUV420, the format Android's `camera` plugin streams) into an
// `image` package RGB Image, so we can crop the detected face box and resize it to the 96x96
// input the face-emotion CNN expects. Android-only for now (this app doesn't target iOS -- no
// Mac available to build/test that side, see project notes).
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

img.Image convertYuv420ToImage(CameraImage cameraImage) {
  final width = cameraImage.width;
  final height = cameraImage.height;
  final yPlane = cameraImage.planes[0];
  final uPlane = cameraImage.planes[1];
  final vPlane = cameraImage.planes[2];

  final image = img.Image(width: width, height: height);

  final yRowStride = yPlane.bytesPerRow;
  final uvRowStride = uPlane.bytesPerRow;
  final uvPixelStride = uPlane.bytesPerPixel ?? 1;

  for (int row = 0; row < height; row++) {
    final yRowOffset = row * yRowStride;
    final uvRowOffset = (row ~/ 2) * uvRowStride;
    for (int col = 0; col < width; col++) {
      final yIndex = yRowOffset + col;
      final uvIndex = uvRowOffset + (col ~/ 2) * uvPixelStride;

      final y = yPlane.bytes[yIndex];
      final u = uPlane.bytes[uvIndex] - 128;
      final v = vPlane.bytes[uvIndex] - 128;

      // Standard YUV -> RGB (BT.601), clamped to a byte.
      int r = (y + 1.402 * v).round();
      int g = (y - 0.344136 * u - 0.714136 * v).round();
      int b = (y + 1.772 * u).round();
      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      image.setPixelRgb(col, row, r, g, b);
    }
  }
  return image;
}

/// Rotates [image] clockwise by [rotationDegrees] (0/90/180/270 -- same value passed as
/// `sensorOrientation` to build the InputImage for ML Kit). ML Kit's face/pose detectors apply
/// this rotation internally and return bounding boxes/landmarks in the *rotated, upright*
/// image's coordinate space -- but convertYuv420ToImage() above produces the raw, unrotated
/// sensor buffer. Without this step, a face box from ML Kit gets used to crop the wrong region
/// of the unrotated image (e.g. background instead of the face), which is exactly the kind of
/// bug that makes a classifier fall back to a "safe" default class (Neutral) on garbage input.
img.Image applySensorRotation(img.Image image, int rotationDegrees) {
  if (rotationDegrees == 0) return image;
  return img.copyRotate(image, angle: rotationDegrees);
}

/// Crops [box] (pixel coords, already clamped by the caller to the image bounds), resizes to
/// [targetSize] x [targetSize], and returns a CHW float32 tensor normalized with the given
/// per-channel mean/std -- matches face_emotion/train_face_emotion.ipynb's eval_tf
/// (`transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])`).
Float32List cropResizeNormalizeChw(
  img.Image source, {
  required int x,
  required int y,
  required int w,
  required int h,
  required int targetSize,
  List<double> mean = const [0.5, 0.5, 0.5],
  List<double> std = const [0.5, 0.5, 0.5],
}) {
  final cropped = img.copyCrop(source, x: x, y: y, width: w, height: h);
  final resized = img.copyResize(cropped, width: targetSize, height: targetSize);

  final chw = Float32List(3 * targetSize * targetSize);
  final plane = targetSize * targetSize;
  for (int py = 0; py < targetSize; py++) {
    for (int px = 0; px < targetSize; px++) {
      final pixel = resized.getPixel(px, py);
      final idx = py * targetSize + px;
      chw[idx] = ((pixel.r / 255.0) - mean[0]) / std[0];
      chw[plane + idx] = ((pixel.g / 255.0) - mean[1]) / std[1];
      chw[2 * plane + idx] = ((pixel.b / 255.0) - mean[2]) / std[2];
    }
  }
  return chw;
}
