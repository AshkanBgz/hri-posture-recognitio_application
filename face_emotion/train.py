"""Train the face-emotion CNN on the real AffectNet YOLO-format mirror (fatihkgg/affectnet-yolo-format).

Class imbalance here is mild (~1.75x, see conversation), so no oversampling/undersampling --
just a class-weighted loss on the full real dataset. Standard light augmentation (flip/rotation/
color jitter) is applied to train only, for generalization, not for balancing.

Usage: python train.py
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AffectNetYolo, CLASS_NAMES
from model import EmotionCNN

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO_ROOT, 'data_raw_yolo', 'YOLO_format')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
CKPT_PATH = os.path.join(RESULTS_DIR, 'face_emotion_classifier.pt')

DEVICE = torch.device('cpu')
BATCH_SIZE = 64
EPOCHS = 15
PATIENCE = 5
LR = 1e-3

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, macro_f1, all_labels, all_preds


def plot_confusion(labels, preds, path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds = AffectNetYolo(DATA_ROOT, 'train', transform=train_tf)
    val_ds = AffectNetYolo(DATA_ROOT, 'valid', transform=eval_tf)
    test_ds = AffectNetYolo(DATA_ROOT, 'test', transform=eval_tf)
    print(f'train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}')

    counts = train_ds.class_counts()
    total = sum(counts)
    weights = torch.tensor([total / (len(CLASS_NAMES) * c) for c in counts], dtype=torch.float32)
    print('class counts:', dict(zip(CLASS_NAMES, counts)))
    print('class weights:', dict(zip(CLASS_NAMES, [round(w, 2) for w in weights.tolist()])))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = EmotionCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_f1 = -1.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ds)

        val_acc, val_f1, _, _ = evaluate(model, val_loader)
        scheduler.step(val_f1)
        dt = time.time() - t0
        print(f'epoch {epoch:02d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  '
              f'val_macro_f1={val_f1:.4f}  ({dt:.0f}s)')

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save({'model_state': model.state_dict(), 'class_names': CLASS_NAMES}, CKPT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f'Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)')
                break

    print('\nLoading best checkpoint for final test evaluation...')
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_acc, test_f1, labels, preds = evaluate(model, test_loader)
    print(f'TEST  acc={test_acc:.4f}  macro_f1={test_f1:.4f}')

    per_class_f1 = f1_score(labels, preds, average=None)
    for name, f1 in zip(CLASS_NAMES, per_class_f1):
        print(f'  {name:<10} f1={f1:.3f}')

    plot_confusion(labels, preds, os.path.join(RESULTS_DIR, 'face_emotion_confusion_matrix.png'))
    print(f'\nSaved model to {CKPT_PATH}')
    print(f'Saved confusion matrix to {os.path.join(RESULTS_DIR, "face_emotion_confusion_matrix.png")}')


if __name__ == '__main__':
    main()
