# hri-posture-recognitio_app

This is the individual extension of my group project (The Unfiltered — Social Cue Analysis for
Human-Robot Interaction). The group-phase work (posture classifier, live webcam demo, first fusion
demo with my teammate's face-emotion model) lives in a separate repo,
[hri-posture-recognition](https://github.com/AshkanBgz/hri-posture-recognition), and was presented
on 28 July 2026. Everything in this repo was built afterwards, on my own, as an individual extension:
I replaced the old hand-written fusion logic with an actual trained fusion model, trained my own
face-emotion CNN, and turned the whole thing into a working on-device Android app.

## What's in here
- `posture/` — the posture/confidence classifier carried over from the group phase, plus a real bug
  I found and fixed: the aux `arm_position` classifier's scale didn't match live webcam data at all,
  so I replaced it with a per-user calibration step instead of retraining (`debug_arm_position.py`).
- `face_emotion/` — my own face-emotion model, a small CNN trained from scratch on real AffectNet
  data (not reused from my teammate's YOLO model). Two attempts to improve it with a pretrained
  backbone made things worse and were reverted — see the commit history and the reflection in my
  individual report for why.
- `fusion/` — a real trained fusion model (`train_fusion.py`), replacing the hand-written lookup
  table from the group phase. I rejected reusing the lookup table a second time on purpose; see the
  individual report for the reasoning.
- `app/` — the Flutter Android app. All three models above are exported to ONNX and run entirely
  on-device (no server), with the camera feed driving posture + face-emotion + fusion inference and
  a response-text output for each combined social state.

Raw training data isn't committed here (see `.gitignore`) — the face-emotion dataset is
[AffectNet (YOLO format) on Kaggle](https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format),
and the posture dataset is the same [Confidence Detection Dataset](https://www.kaggle.com/datasets/muhammadkhubaibahmad/confidence-detection-dataset)
used in the group-phase repo.

## Running it
**Posture / face-emotion / fusion training** — each has its own `train*.py` (or notebook, for
face-emotion) and `export_onnx.py` under its folder. These were run on Google Colab (GPU) for the
face-emotion CNN, and locally for posture/fusion.

**The app** — `app/` is a normal Flutter project:
```
cd app
flutter pub get
flutter build apk --debug
```
Needs Flutter 3.x + Android SDK 34+ + JDK 17. The exported ONNX models are already bundled under
`app/assets/models/`, so no extra download is needed to build and run it.

## Results
- Face-emotion CNN: 69.5% test accuracy, macro F1 68.4% (weakest on Sad/Neutral, a commonly-confused
  pair).
- Fusion MLP: 93% held-out accuracy on synthetic training data, 100% agreement with the old
  hand-written lookup table on fully-confident one-hot inputs (sanity check).
- All 5 models (posture main + 2 aux classifiers, face-emotion CNN, fusion MLP) verified with 100%
  prediction agreement between the original PyTorch/scikit-learn models and their ONNX exports.
- App builds and runs on-device; tested live on a real Android phone.

## A couple of real bugs worth knowing about
- The posture model's ONNX output class order (alphabetical: Confident/Low/Neutral) didn't match
  what the fusion model was trained to expect (Confident/Neutral/Low). Fixed by reordering the
  probabilities before they reach the fusion model — otherwise it would have silently received
  scrambled inputs. General lesson: check that every model in a chain agrees on class order, not
  just class names.
- The fused state flickered frame-to-frame during live testing, so I added rolling majority-vote
  smoothing over recent predictions rather than trusting each single frame.
