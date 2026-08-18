import 'dart:typed_data';

import 'face_emotion_service.dart';
import 'onnx_models.dart';
import 'posture_service.dart';

class FusedResult {
  final String state;
  final String responseText;

  FusedResult(this.state, this.responseText);
}

// Response generation -- required by the project proposal ("the system will generate
// appropriate responses that could be used by a social robot or virtual assistant"), not
// optional polish. One short scripted line per fused state; a real robot would swap this for
// speech/gesture output, but the mapping itself is the deliverable here.
const _responseText = {
  'Positive & Engaged': "You seem happy and engaged -- great to see!",
  'Positive, Moderate Engagement': "You seem in good spirits.",
  'Positive but Reserved (smiling, closed body language)':
      "You're smiling, but your posture looks a little closed off -- everything okay?",
  'Calm & Confident': "You seem calm and confident right now.",
  'Neutral': "You seem to be in a neutral state.",
  'Neutral but Withdrawn': "You seem a bit withdrawn -- is everything alright?",
  'Negative but Assertive (mixed signal)':
      "Your expression and posture are giving mixed signals -- want to talk about it?",
  'Mildly Negative': "You seem a little down.",
  'Negative & Disengaged': "You seem disengaged -- let me know if you'd like to pause.",
};

class FusionService {
  final OnnxModels models;

  FusionService(this.models);

  FusedResult fuse(FaceEmotionResult face, PostureResult posture) {
    final postureProbsReordered = models.reorderConfidenceProbsForFusion(posture.confidenceProbs);
    final input = Float32List.fromList([...face.probabilities, ...postureProbsReordered]);
    final result = models.runFusion(input);
    return FusedResult(result.label, responseFor(result.label));
  }

  String responseFor(String state) => _responseText[state] ?? '...';
}
