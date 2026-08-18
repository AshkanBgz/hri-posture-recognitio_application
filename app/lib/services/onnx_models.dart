// Loads the 5 ONNX models (exported from the project's PyTorch/sklearn models -- see
// face_emotion/export_onnx.py, posture/export_onnx.py, fusion/export_onnx.py) as ONNX Runtime
// sessions, and exposes one typed run method per model so the rest of the app never touches
// onnxruntime's low-level FFI API directly.
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

class ClassifierResult {
  final String label;
  final List<double> probabilities;
  final List<String> classes;

  ClassifierResult(this.label, this.probabilities, this.classes);

  double get confidence => probabilities.reduce((a, b) => a > b ? a : b);
}

class OnnxModels {
  static const _assetDir = 'assets/models';

  late OrtSession _faceEmotion;
  late OrtSession _headDirection;
  late OrtSession _postureAux;
  late OrtSession _confidence;
  late OrtSession _fusion;

  late List<String> faceClasses;
  late List<String> headDirectionClasses;
  late List<String> postureClasses;
  late List<String> confidenceClasses;
  late List<String> confidenceFeatureCols;
  late List<String> confidenceNumericCols;
  late List<String> fusedLabels;

  /// The posture-class order the fusion model was actually trained on (train_fusion.py's
  /// POSTURE_CLASSES = ['Confident','Neutral','Low']) -- NOT the same order as
  /// [confidenceClasses], which is alphabetical (sklearn LabelEncoder default: Confident/Low/
  /// Neutral). Always reorder confidence_classifier's probability output into this order (see
  /// [reorderConfidenceProbsForFusion]) before concatenating with face probs for the fusion model.
  late List<String> fusionPostureClassOrder;

  bool _loaded = false;
  bool get isLoaded => _loaded;

  Future<void> loadAll() async {
    OrtEnv.instance.init();

    final options = OrtSessionOptions();
    _faceEmotion = await _loadSession('face_emotion_classifier.onnx', options);
    _headDirection = await _loadSession('head_direction_classifier_aux.onnx', options);
    _postureAux = await _loadSession('posture_classifier_aux.onnx', options);
    _confidence = await _loadSession('confidence_classifier.onnx', options);
    _fusion = await _loadSession('fusion_mlp.onnx', options);

    final confMeta = await _loadJson('confidence_classifier_meta.json');
    confidenceFeatureCols = List<String>.from(confMeta['feature_cols']);
    confidenceNumericCols = List<String>.from(confMeta['numeric_cols']);
    confidenceClasses = List<String>.from(confMeta['confidence_classes']);
    headDirectionClasses = List<String>.from(confMeta['head_direction_classes']);
    postureClasses = List<String>.from(confMeta['posture_classes']);

    final fusionMeta = await _loadJson('fusion_mlp_meta.json');
    faceClasses = List<String>.from(fusionMeta['face_classes']);
    fusedLabels = List<String>.from(fusionMeta['fused_labels']);
    fusionPostureClassOrder = List<String>.from(fusionMeta['posture_classes']);

    _loaded = true;
  }

  /// Reorders confidence_classifier's [probs] (in [confidenceClasses] order) into
  /// [fusionPostureClassOrder], the order the fusion model actually expects.
  List<double> reorderConfidenceProbsForFusion(List<double> probs) {
    return fusionPostureClassOrder
        .map((cls) => probs[confidenceClasses.indexOf(cls)])
        .toList();
  }

  Future<OrtSession> _loadSession(String fileName, OrtSessionOptions options) async {
    final bytes = await rootBundle.load('$_assetDir/$fileName');
    return OrtSession.fromBuffer(bytes.buffer.asUint8List(), options);
  }

  Future<Map<String, dynamic>> _loadJson(String fileName) async {
    final raw = await rootBundle.loadString('$_assetDir/$fileName');
    return jsonDecode(raw) as Map<String, dynamic>;
  }

  /// Runs a session with a single float input tensor of the given [shape] and returns
  /// (label index, probabilities) from the standard skl2onnx classifier output pair
  /// ("label" int64 + "probabilities" float[N,classes]).
  (int, List<double>) _runClassifier(
      OrtSession session, Float32List input, List<int> shape) {
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, shape);
    final runOptions = OrtRunOptions();
    final outputs = session.run(runOptions, {'input': inputTensor});
    final labelIdx = (outputs[0]?.value as List).first as int;
    final probs = (outputs[1]?.value as List).first as List;
    inputTensor.release();
    runOptions.release();
    for (final o in outputs) {
      o?.release();
    }
    return (labelIdx, probs.map((e) => (e as num).toDouble()).toList());
  }

  /// Face-emotion CNN -- image must already be a [3,96,96] CHW float tensor, normalized with
  /// mean=0.5/std=0.5 per channel (same preprocessing as face_emotion/train_face_emotion.ipynb).
  /// Returns raw logits (softmax applied by the caller, see FaceEmotionService).
  List<double> runFaceEmotionLogits(Float32List chwImage) {
    final inputTensor =
        OrtValueTensor.createTensorWithDataList(chwImage, [1, 3, 96, 96]);
    final runOptions = OrtRunOptions();
    final outputs = _faceEmotion.run(runOptions, {'image': inputTensor});
    final logits = (outputs[0]?.value as List).first as List;
    inputTensor.release();
    runOptions.release();
    for (final o in outputs) {
      o?.release();
    }
    return logits.map((e) => (e as num).toDouble()).toList();
  }

  ClassifierResult runHeadDirection(Float32List numeric15) {
    final (idx, probs) = _runClassifier(_headDirection, numeric15, [1, 15]);
    return ClassifierResult(headDirectionClasses[idx], probs, headDirectionClasses);
  }

  ClassifierResult runPostureAux(Float32List numeric15) {
    final (idx, probs) = _runClassifier(_postureAux, numeric15, [1, 15]);
    return ClassifierResult(postureClasses[idx], probs, postureClasses);
  }

  ClassifierResult runConfidence(Float32List features25) {
    final (idx, probs) = _runClassifier(_confidence, features25, [1, 25]);
    return ClassifierResult(confidenceClasses[idx], probs, confidenceClasses);
  }

  ClassifierResult runFusion(Float32List concatProbs11) {
    final (idx, probs) = _runClassifier(_fusion, concatProbs11, [1, 11]);
    return ClassifierResult(fusedLabels[idx], probs, fusedLabels);
  }
}
