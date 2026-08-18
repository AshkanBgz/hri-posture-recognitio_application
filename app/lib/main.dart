import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'services/camera_frame_converter.dart';
import 'services/face_emotion_service.dart';
import 'services/fusion_service.dart';
import 'services/image_convert.dart';
import 'services/onnx_models.dart';
import 'services/posture_service.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const HriApp());
}

class HriApp extends StatelessWidget {
  const HriApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'The Unfiltered - Social Cue Analysis',
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  final OnnxModels _models = OnnxModels();
  FaceEmotionService? _faceService;
  PostureService? _postureService;
  FusionService? _fusionService;

  String _status = 'Loading models...';
  bool _busy = false;

  // Live results, updated as frames come in.
  String _faceLine = '';
  String _postureLine = '';
  String _fusedLine = '';
  String _responseLine = '';
  double? _liveRatio;

  // Raw per-frame fused-state predictions are noisy (flip almost every frame). Smooth the
  // *displayed* state by taking the majority vote over a short rolling window, so the UI only
  // switches once a new state is consistently observed rather than on single-frame blips.
  static const int _fusedSmoothingWindow = 8;
  final List<String> _recentFusedStates = [];

  String _smoothedFusedState(String newState) {
    _recentFusedStates.add(newState);
    if (_recentFusedStates.length > _fusedSmoothingWindow) {
      _recentFusedStates.removeAt(0);
    }
    final counts = <String, int>{};
    for (final s in _recentFusedStates) {
      counts[s] = (counts[s] ?? 0) + 1;
    }
    var best = _recentFusedStates.first;
    var bestCount = 0;
    for (final entry in counts.entries) {
      if (entry.value > bestCount) {
        best = entry.key;
        bestCount = entry.value;
      }
    }
    return best;
  }

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await _models.loadAll();
    _faceService = FaceEmotionService(_models);
    _postureService = PostureService(_models);
    _fusionService = FusionService(_models);
    await _postureService!.loadSavedCalibration();

    if (cameras.isEmpty) {
      setState(() => _status = 'No camera found on this device.');
      return;
    }
    final front = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );
    final controller = CameraController(
      front,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    try {
      await controller.initialize();
      if (!mounted) return;
      setState(() {
        _controller = controller;
        _status = _postureService!.isCalibrated
            ? 'Running.'
            : 'Calibration needed -- hold arms fully OPEN, tap "Mark Open".';
      });
      controller.startImageStream(_onFrame);
    } catch (e) {
      setState(() => _status = 'Camera init failed: $e');
    }
  }

  void _onFrame(CameraImage image) {
    if (_busy) return;
    _busy = true;
    _processFrame(image).whenComplete(() => _busy = false);
  }

  Future<void> _processFrame(CameraImage image) async {
    final controller = _controller;
    if (controller == null) return;

    final rotation = controller.description.sensorOrientation;
    final inputImage = cameraImageToInputImage(image, rotation);
    if (inputImage == null) return;

    // ML Kit's face/pose detectors internally rotate the buffer by `rotation` and return
    // boxes/landmarks in that *rotated, upright* coordinate space -- so anything we compare
    // them against (crop source dimensions, normalization width/height) must use the rotated
    // dimensions too, not the raw sensor buffer's. Width/height swap when rotation is 90/270.
    final rotated90or270 = rotation == 90 || rotation == 270;
    final rotatedWidth = rotated90or270 ? image.height : image.width;
    final rotatedHeight = rotated90or270 ? image.width : image.height;

    final postureService = _postureService!;

    if (!postureService.isCalibrated) {
      // process() returns null while calibrating (by design) -- it still runs pose detection
      // and feeds calibrationAverager as a side effect, which is all we need here.
      await postureService.process(inputImage, rotatedWidth, rotatedHeight);
      if (mounted) {
        setState(() {
          _liveRatio = postureService.calibrationAverager.average;
        });
      }
      return;
    }

    final postureResult = await postureService.process(inputImage, rotatedWidth, rotatedHeight);

    final decodedFrame = applySensorRotation(convertYuv420ToImage(image), rotation);
    final faceResult = await _faceService!.process(inputImage, decodedFrame);

    if (!mounted) return;
    setState(() {
      if (faceResult != null) {
        _faceLine = 'Face: ${faceResult.label} '
            '(${(faceResult.probabilities.reduce((a, b) => a > b ? a : b) * 100).toStringAsFixed(0)}%)';
      } else {
        _faceLine = 'Face: no face detected';
      }
      if (postureResult != null) {
        _postureLine = 'Posture: ${postureResult.confidenceLabel} '
            '(arms=${postureResult.armPosition}, head=${postureResult.headDirection})';
      } else {
        _postureLine = 'Posture: no person detected';
      }
      if (faceResult != null && postureResult != null) {
        final fused = _fusionService!.fuse(faceResult, postureResult);
        final smoothedState = _smoothedFusedState(fused.state);
        _fusedLine = 'Combined: $smoothedState';
        _responseLine = _fusionService!.responseFor(smoothedState);
      } else {
        _fusedLine = 'Combined: waiting for both signals';
        _responseLine = '';
      }
    });
  }

  Future<void> _markCalibration({required bool isOpen}) async {
    await _postureService!.saveCalibrationSample(isOpen: isOpen);
    if (!mounted) return;
    setState(() {
      _status = _postureService!.isCalibrated
          ? 'Calibrated -- running.'
          : isOpen
              ? 'Open captured. Now hold arms fully CLOSED, tap "Mark Closed".'
              : 'Closed captured. Now hold arms fully OPEN, tap "Mark Open".';
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    _faceService?.dispose();
    _postureService?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    final calibrated = _postureService?.isCalibrated ?? false;

    return Scaffold(
      appBar: AppBar(title: const Text('The Unfiltered')),
      body: Column(
        children: [
          Expanded(
            child: controller != null && controller.value.isInitialized
                ? CameraPreview(controller)
                : Center(child: Text(_status)),
          ),
          if (controller != null && !calibrated)
            Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  Text(_liveRatio == null
                      ? 'Detecting pose...'
                      : 'Live ratio: ${_liveRatio!.toStringAsFixed(3)}'),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      ElevatedButton(
                        onPressed: () => _markCalibration(isOpen: true),
                        child: const Text('Mark Open'),
                      ),
                      const SizedBox(width: 16),
                      ElevatedButton(
                        onPressed: () => _markCalibration(isOpen: false),
                        child: const Text('Mark Closed'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            color: Colors.black87,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(_status, style: const TextStyle(color: Colors.grey, fontSize: 12)),
                if (calibrated) ...[
                  Text(_faceLine, style: const TextStyle(color: Colors.orangeAccent, fontSize: 13)),
                  Text(_postureLine, style: const TextStyle(color: Colors.greenAccent, fontSize: 13)),
                  Text(_fusedLine,
                      style: const TextStyle(
                          color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold)),
                  if (_responseLine.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 4),
                      child: Text(_responseLine,
                          style: const TextStyle(color: Colors.tealAccent, fontSize: 13)),
                    ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
