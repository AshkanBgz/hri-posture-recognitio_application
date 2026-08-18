// Dart port of extract_features_v2() from hri-posture-recognition-new/demo/live_webcam_demo.py.
// Must stay numerically equivalent to that Python function -- it's what the posture RF models
// were trained on. ML Kit's PoseLandmark gives x/y in *pixel* coordinates (image frame), unlike
// MediaPipe's Python API which gives normalized [0,1] coordinates directly -- so every landmark
// here is normalized by image width/height first, to reproduce MediaPipe's coordinate space
// before computing the same ratios/angles.
import 'dart:math' as math;

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

const numericFeatureOrder = [
  'eye_shoulder_y_ratio',
  'shoulder_y_diff',
  'wrist_distance_x',
  'wrist_shoulder_ratio',
  'nose_eye_center_offset_x',
  'shoulder_span',
  'hip_shoulder_y_diff',
  'body_lean_x',
  'shoulder_center_x',
  'hip_center_x',
  'spine_angle',
  'eye_distance',
  'head_tilt_angle',
  'eye_distance_ratio',
  'shoulder_slope',
];

class _Point {
  final double x, y;
  const _Point(this.x, this.y);
}

_Point _norm(PoseLandmark lm, int imageWidth, int imageHeight) =>
    _Point(lm.x / imageWidth, lm.y / imageHeight);

/// Returns null if any required landmark is missing from [landmarks] (e.g. person not fully
/// in frame) -- callers should skip the frame rather than feed a partial/garbage feature vector.
Map<String, double>? extractPostureFeatures(
    Map<PoseLandmarkType, PoseLandmark> landmarks, int imageWidth, int imageHeight) {
  final required = [
    PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder,
    PoseLandmarkType.leftHip, PoseLandmarkType.rightHip,
    PoseLandmarkType.leftWrist, PoseLandmarkType.rightWrist,
    PoseLandmarkType.leftEye, PoseLandmarkType.rightEye,
    PoseLandmarkType.nose,
  ];
  for (final t in required) {
    if (!landmarks.containsKey(t)) return null;
  }

  final lSh = _norm(landmarks[PoseLandmarkType.leftShoulder]!, imageWidth, imageHeight);
  final rSh = _norm(landmarks[PoseLandmarkType.rightShoulder]!, imageWidth, imageHeight);
  final lHip = _norm(landmarks[PoseLandmarkType.leftHip]!, imageWidth, imageHeight);
  final rHip = _norm(landmarks[PoseLandmarkType.rightHip]!, imageWidth, imageHeight);
  final lWr = _norm(landmarks[PoseLandmarkType.leftWrist]!, imageWidth, imageHeight);
  final rWr = _norm(landmarks[PoseLandmarkType.rightWrist]!, imageWidth, imageHeight);
  final lEye = _norm(landmarks[PoseLandmarkType.leftEye]!, imageWidth, imageHeight);
  final rEye = _norm(landmarks[PoseLandmarkType.rightEye]!, imageWidth, imageHeight);
  final nose = _norm(landmarks[PoseLandmarkType.nose]!, imageWidth, imageHeight);

  final shoulderCenterX = (lSh.x + rSh.x) / 2;
  final shoulderCenterY = (lSh.y + rSh.y) / 2;
  final hipCenterX = (lHip.x + rHip.x) / 2;
  final hipCenterY = (lHip.y + rHip.y) / 2;
  final eyeCenterX = (lEye.x + rEye.x) / 2;
  final eyeCenterY = (lEye.y + rEye.y) / 2;

  final shoulderSpan = math.sqrt(math.pow(lSh.x - rSh.x, 2) + math.pow(lSh.y - rSh.y, 2)) + 1e-6;
  final wristDistanceX = (lWr.x - rWr.x).abs();
  final wristShoulderRatio = wristDistanceX / shoulderSpan;

  final eyeDistance = math.sqrt(math.pow(lEye.x - rEye.x, 2) + math.pow(lEye.y - rEye.y, 2));
  final eyeDistanceRatio = eyeDistance / shoulderSpan;

  final noseEyeCenterOffsetX = nose.x - eyeCenterX;

  final shoulderYDiff = (lSh.y - rSh.y).abs();
  final shoulderSlope = shoulderYDiff;

  final hipShoulderYDiff = hipCenterY - shoulderCenterY;
  final bodyLeanX = shoulderCenterX - hipCenterX;

  final spineAngle =
      180 - _degrees(math.atan2(hipShoulderYDiff, bodyLeanX + 1e-9));
  final eyeShoulderYRatio = (eyeCenterY - shoulderCenterY) / shoulderSpan;
  final headTiltAngle = _degrees(math.atan2(lEye.y - rEye.y, lEye.x - rEye.x + 1e-9));

  return {
    'eye_shoulder_y_ratio': eyeShoulderYRatio,
    'shoulder_y_diff': shoulderYDiff,
    'wrist_distance_x': wristDistanceX,
    'wrist_shoulder_ratio': wristShoulderRatio,
    'nose_eye_center_offset_x': noseEyeCenterOffsetX,
    'shoulder_span': shoulderSpan,
    'hip_shoulder_y_diff': hipShoulderYDiff,
    'body_lean_x': bodyLeanX,
    'shoulder_center_x': shoulderCenterX,
    'hip_center_x': hipCenterX,
    'spine_angle': spineAngle,
    'eye_distance': eyeDistance,
    'head_tilt_angle': headTiltAngle,
    'eye_distance_ratio': eyeDistanceRatio,
    'shoulder_slope': shoulderSlope,
  };
}

double _degrees(double radians) => radians * 180 / math.pi;
