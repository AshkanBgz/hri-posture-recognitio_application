// Per-user/per-camera arm_position calibration -- same fix as posture/debug_arm_position.py.
// The dataset's arm_position classifier doesn't transfer to live camera input (confirmed by
// testing: dataset "Closed Arms" mean wrist_shoulder_ratio=0.51, but live testing showed the
// ratio never drops below ~1.4-1.6 no matter how closed the arms actually are -- a scale
// mismatch, not a threshold problem). Fix: ask the user to show fully-open and fully-closed arms
// once, split their own live range into three even bands, and persist it to disk.
import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

class ArmCalibration {
  final double closedRatio;
  final double openRatio;

  const ArmCalibration(this.closedRatio, this.openRatio);

  String classify(double ratio) {
    final lo = closedRatio + (openRatio - closedRatio) / 3;
    final hi = closedRatio + 2 * (openRatio - closedRatio) / 3;
    if (ratio <= lo) return 'Closed Arms';
    if (ratio <= hi) return 'Partially Open';
    return 'Open Arms';
  }

  static Future<File> _file() async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/arm_calibration.json');
  }

  static Future<ArmCalibration?> load() async {
    final file = await _file();
    if (!await file.exists()) return null;
    final json = jsonDecode(await file.readAsString());
    return ArmCalibration(
      (json['closed_ratio'] as num).toDouble(),
      (json['open_ratio'] as num).toDouble(),
    );
  }

  Future<void> save() async {
    final file = await _file();
    await file.writeAsString(jsonEncode({
      'closed_ratio': closedRatio,
      'open_ratio': openRatio,
    }));
  }
}

/// Averages a rolling window of live ratio samples during calibration, same as the
/// CALIB_SAMPLE_WINDOW average in debug_arm_position.py -- reduces landmark jitter noise.
class RatioAverager {
  final List<double> _samples = [];
  final int windowSize;

  RatioAverager({this.windowSize = 15});

  void add(double ratio) {
    _samples.add(ratio);
    if (_samples.length > windowSize) _samples.removeAt(0);
  }

  double? get average =>
      _samples.isEmpty ? null : _samples.reduce((a, b) => a + b) / _samples.length;

  void clear() => _samples.clear();
}
