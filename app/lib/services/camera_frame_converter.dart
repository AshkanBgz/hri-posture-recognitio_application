// Converts a `camera` package CameraImage (YUV_420_888 on Android) into the NV21-format
// InputImage that google_mlkit needs. NV21 (not raw YUV_420_888) is used because it's the
// format ML Kit's Android SDK supports most reliably across versions -- passing YUV_420_888
// planes directly is not consistently supported. Android-only (no iOS build/test available on
// this machine -- see project notes).
import 'dart:typed_data';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';

InputImage? cameraImageToInputImage(CameraImage image, int sensorOrientation) {
  final rotation =
      InputImageRotationValue.fromRawValue(sensorOrientation) ?? InputImageRotation.rotation0deg;

  final nv21Bytes = _yuv420ToNv21(image);

  return InputImage.fromBytes(
    bytes: nv21Bytes,
    metadata: InputImageMetadata(
      size: Size(image.width.toDouble(), image.height.toDouble()),
      rotation: rotation,
      format: InputImageFormat.nv21,
      bytesPerRow: image.width,
    ),
  );
}

Uint8List _yuv420ToNv21(CameraImage image) {
  final width = image.width;
  final height = image.height;
  final yPlane = image.planes[0];
  final uPlane = image.planes[1];
  final vPlane = image.planes[2];

  final ySize = width * height;
  final uvSize = width * height ~/ 2;
  final nv21 = Uint8List(ySize + uvSize);

  // Y plane: copy row by row in case bytesPerRow has padding beyond `width`.
  int offset = 0;
  for (int row = 0; row < height; row++) {
    final rowStart = row * yPlane.bytesPerRow;
    nv21.setRange(offset, offset + width, yPlane.bytes, rowStart);
    offset += width;
  }

  // NV21 interleaves V,U (in that order) at half resolution. U/V planes from the camera plugin
  // may themselves already be interleaved (pixelStride == 2, sharing one buffer) or separate
  // (pixelStride == 1) depending on device -- handle both.
  final uvPixelStride = uPlane.bytesPerPixel ?? 1;
  final uvRowStride = uPlane.bytesPerRow;
  for (int row = 0; row < height ~/ 2; row++) {
    for (int col = 0; col < width ~/ 2; col++) {
      final uvIndex = row * uvRowStride + col * uvPixelStride;
      nv21[offset++] = vPlane.bytes[uvIndex];
      nv21[offset++] = uPlane.bytes[uvIndex];
    }
  }

  return nv21;
}
