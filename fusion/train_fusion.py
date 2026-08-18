"""Trains a real fusion model (not a hand-written lookup table) that combines the face-emotion
CNN's 8-class probability output with the posture module's 3-class confidence probability output
into one social-state label.

Why not a lookup table: a hand-written valence x engagement table (9 cells) is exactly what an
earlier version of this project used (hri-posture-recognition-new/demo/combined_demo.py) -- simple
and interpretable, but it only ever looks at the *argmax* class of each modality, throwing away
how confident each model actually was. Two inputs that are both "60% happy" and "95% happy" get
treated identically as long as happy wins.

Why synthetic data instead of a real labeled dataset: there's no public dataset that pairs face
emotion + posture confidence with a ground-truth combined social-state label (this is also called
out as a known limitation in the old combined_demo.py). So instead:
  1. Sample random face-emotion probability vectors (8-dim, Dirichlet) and posture-confidence
     probability vectors (3-dim, Dirichlet) covering both "peaky" (confident) and "diffuse"
     (uncertain) distributions.
  2. Label each synthetic pair using the same valence x engagement rule the old table used --
     this keeps the *meaning* of each combined-state class grounded in the project's original
     reasoned design, not arbitrary.
  3. Train a small MLP directly on the continuous probability vectors (not the discretised
     valence/engagement bins). The MLP still reproduces the rule where classes are unambiguous,
     but because it sees the full probability vector, it can interpolate smoothly near class
     boundaries and generalize to probability combinations the hand-written table never
     enumerated -- e.g. "50% happy / 50% sad" isn't just forced into whichever wins by 1%.

Usage: python train_fusion.py
"""
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
MODEL_PATH = os.path.join(RESULTS_DIR, 'fusion_mlp.pkl')

RNG = np.random.default_rng(42)
N_SAMPLES = 40_000

# Must match hri-multimodal-app/face_emotion/dataset.py CLASS_NAMES order exactly --
# this fusion model is built for the project's own CNN, not the teammate's YOLO model
# (different class set/order).
FACE_CLASSES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
POSTURE_CLASSES = ['Confident', 'Neutral', 'Low']

EMOTION_VALENCE = {
    'Happy': 'positive', 'Surprise': 'positive',
    'Neutral': 'neutral',
    'Anger': 'negative', 'Contempt': 'negative', 'Disgust': 'negative', 'Fear': 'negative',
    'Sad': 'negative',
}
CONFIDENCE_ENGAGEMENT = {'Confident': 'high', 'Neutral': 'medium', 'Low': 'low'}

# Same 9-cell reasoning as the old hand-written table -- reused here as the *labelling rule* for
# synthetic data, not as the thing making live predictions.
FUSION_TABLE = {
    ('positive', 'high'): 'Positive & Engaged',
    ('positive', 'medium'): 'Positive, Moderate Engagement',
    ('positive', 'low'): 'Positive but Reserved (smiling, closed body language)',
    ('neutral', 'high'): 'Calm & Confident',
    ('neutral', 'medium'): 'Neutral',
    ('neutral', 'low'): 'Neutral but Withdrawn',
    ('negative', 'high'): 'Negative but Assertive (mixed signal)',
    ('negative', 'medium'): 'Mildly Negative',
    ('negative', 'low'): 'Negative & Disengaged',
}
FUSED_LABELS = sorted(set(FUSION_TABLE.values()))


def sample_prob_vectors(n, k, rng):
    """Dirichlet sampling across a spread of concentration parameters, so the synthetic data
    covers both confident (peaky, alpha small) and uncertain (diffuse, alpha large) predictions --
    real softmax outputs from a trained classifier look like both depending on the input."""
    alphas = rng.choice([0.3, 0.8, 1.5, 3.0, 6.0], size=n)
    return np.array([rng.dirichlet(np.full(k, a)) for a in alphas])


def make_synthetic_dataset(n, rng):
    face_probs = sample_prob_vectors(n, len(FACE_CLASSES), rng)
    posture_probs = sample_prob_vectors(n, len(POSTURE_CLASSES), rng)

    labels = []
    for fp, pp in zip(face_probs, posture_probs):
        face_class = FACE_CLASSES[int(np.argmax(fp))]
        posture_class = POSTURE_CLASSES[int(np.argmax(pp))]
        valence = EMOTION_VALENCE[face_class]
        engagement = CONFIDENCE_ENGAGEMENT[posture_class]
        labels.append(FUSION_TABLE[(valence, engagement)])

    X = np.hstack([face_probs, posture_probs])
    y = np.array(labels)
    return X, y


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f'Generating {N_SAMPLES} synthetic (face_probs, posture_probs) -> fused_state samples...')
    X, y = make_synthetic_dataset(N_SAMPLES, RNG)
    print('Class distribution:', {lbl: int((y == lbl).sum()) for lbl in FUSED_LABELS})

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation='relu', alpha=1e-4,
        max_iter=500, random_state=42, early_stopping=True,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\nHeld-out synthetic test accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Sanity check: on unambiguous, fully-confident one-hot inputs, the MLP should reproduce
    # the original rule table exactly -- if not, something's wrong with training.
    print('Sanity check against original rule table (one-hot inputs):')
    mismatches = 0
    for face_class in FACE_CLASSES:
        for posture_class in POSTURE_CLASSES:
            fp = np.zeros(len(FACE_CLASSES))
            fp[FACE_CLASSES.index(face_class)] = 1.0
            pp = np.zeros(len(POSTURE_CLASSES))
            pp[POSTURE_CLASSES.index(posture_class)] = 1.0
            x = np.hstack([fp, pp]).reshape(1, -1)
            pred = le.inverse_transform(clf.predict(x))[0]
            expected = FUSION_TABLE[(EMOTION_VALENCE[face_class], CONFIDENCE_ENGAGEMENT[posture_class])]
            if pred != expected:
                mismatches += 1
                print(f'  MISMATCH: face={face_class} posture={posture_class} '
                      f'-> predicted={pred!r} expected={expected!r}')
    if mismatches == 0:
        print('  All one-hot combinations match the original rule table.')
    else:
        print(f'  {mismatches} mismatches out of {len(FACE_CLASSES) * len(POSTURE_CLASSES)} -- '
              'consider more training iterations or a larger hidden layer.')

    joblib.dump({
        'model': clf,
        'label_encoder': le,
        'face_classes': FACE_CLASSES,
        'posture_classes': POSTURE_CLASSES,
        'fused_labels': FUSED_LABELS,
    }, MODEL_PATH)
    print(f'\nSaved fusion model to {MODEL_PATH}')


if __name__ == '__main__':
    main()
