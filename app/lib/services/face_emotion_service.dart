import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'image_convert.dart';
import 'math_utils.dart';
import 'onnx_models.dart';

class FaceEmotionResult {
  final String label;
  final List<double> probabilities;
  final List<String> classes;

  FaceEmotionResult(this.label, this.probabilities, this.classes);
}

class FaceEmotionService {
  final OnnxModels models;
  final FaceDetector _detector = FaceDetector(
    options: FaceDetectorOptions(performanceMode: FaceDetectorMode.fast),
  );

  FaceEmotionService(this.models);

  /// Detects the largest face in [inputImage], crops it out of the already-converted
  /// [decodedFrame] (see image_convert.dart), and classifies its emotion.
  /// Returns null if no face was found.
  Future<FaceEmotionResult?> process(InputImage inputImage, img.Image decodedFrame) async {
    final faces = await _detector.processImage(inputImage);
    if (faces.isEmpty) return null;

    faces.sort((a, b) =>
        (b.boundingBox.width * b.boundingBox.height)
            .compareTo(a.boundingBox.width * a.boundingBox.height));
    final box = faces.first.boundingBox;

    final x = box.left.round().clamp(0, decodedFrame.width - 1);
    final y = box.top.round().clamp(0, decodedFrame.height - 1);
    final w = box.width.round().clamp(1, decodedFrame.width - x);
    final h = box.height.round().clamp(1, decodedFrame.height - y);

    final chw = cropResizeNormalizeChw(decodedFrame, x: x, y: y, w: w, h: h, targetSize: 96);
    final logits = models.runFaceEmotionLogits(chw);
    final probs = softmax(logits);

    var bestIdx = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    return FaceEmotionResult(models.faceClasses[bestIdx], probs, models.faceClasses);
  }

  void dispose() => _detector.close();
}
