import 'dart:typed_data';

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import 'arm_calibration.dart';
import 'feature_extraction.dart';
import 'onnx_models.dart';

class PostureResult {
  final String confidenceLabel;
  final List<double> confidenceProbs; // order = models.confidenceClasses
  final String armPosition;
  final String headDirection;
  final String postureLabel;

  PostureResult(this.confidenceLabel, this.confidenceProbs, this.armPosition,
      this.headDirection, this.postureLabel);
}

class PostureService {
  final OnnxModels models;
  final PoseDetector _detector =
      PoseDetector(options: PoseDetectorOptions(mode: PoseDetectionMode.stream));

  ArmCalibration? calibration;
  final RatioAverager calibrationAverager = RatioAverager();
  bool get isCalibrated => calibration != null;

  PostureService(this.models);

  Future<void> loadSavedCalibration() async {
    calibration = await ArmCalibration.load();
  }

  Future<void> saveCalibrationSample({required bool isOpen}) async {
    final avg = calibrationAverager.average;
    if (avg == null) return;
    if (isOpen) {
      _pendingOpenRatio = avg;
    } else {
      _pendingClosedRatio = avg;
    }
    if (_pendingOpenRatio != null && _pendingClosedRatio != null) {
      if (_pendingOpenRatio! <= _pendingClosedRatio!) {
        // Invalid calibration (open should read higher than closed) -- caller should prompt retry.
        _pendingOpenRatio = null;
        _pendingClosedRatio = null;
        return;
      }
      calibration = ArmCalibration(_pendingClosedRatio!, _pendingOpenRatio!);
      await calibration!.save();
      _pendingOpenRatio = null;
      _pendingClosedRatio = null;
    }
  }

  double? _pendingOpenRatio;
  double? _pendingClosedRatio;

  void resetCalibration() {
    calibration = null;
    _pendingOpenRatio = null;
    _pendingClosedRatio = null;
    calibrationAverager.clear();
  }

  /// Runs pose detection + feature extraction on [inputImage]. During calibration (no saved
  /// [calibration] yet), only feeds the live ratio into [calibrationAverager] and returns null
  /// (caller should read `calibrationAverager.average` for the on-screen readout). Once
  /// calibrated, runs the full posture pipeline and returns a result.
  Future<PostureResult?> process(
      InputImage inputImage, int imageWidth, int imageHeight) async {
    final poses = await _detector.processImage(inputImage);
    if (poses.isEmpty) return null;

    final features = extractPostureFeatures(poses.first.landmarks, imageWidth, imageHeight);
    if (features == null) return null;

    final ratio = features['wrist_shoulder_ratio']!;
    calibrationAverager.add(ratio);

    final calib = calibration;
    if (calib == null) return null;

    final numeric15 = Float32List.fromList(
        numericFeatureOrder.map((name) => features[name]!).toList());

    final headDirection = models.runHeadDirection(numeric15);
    final postureAux = models.runPostureAux(numeric15);
    final armPosition = calib.classify(ratio);

    final oneHot = <String, double>{
      'head_direction_${headDirection.label}': 1.0,
      'arm_position_$armPosition': 1.0,
      'posture_${postureAux.label}': 1.0,
    };

    final features25 = Float32List.fromList(models.confidenceFeatureCols.map((col) {
      if (models.confidenceNumericCols.contains(col)) return features[col]!;
      return oneHot[col] ?? 0.0;
    }).toList());

    final confidence = models.runConfidence(features25);

    return PostureResult(confidence.label, confidence.probabilities, armPosition,
        headDirection.label, postureAux.label);
  }

  void dispose() => _detector.close();
}
