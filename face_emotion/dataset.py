"""Dataset loader for the AffectNet YOLO-format mirror (fatihkgg/affectnet-yolo-format).

Each image has a matching YOLO label .txt with a single line `class cx cy w h`. The box is
essentially the whole 96x96 image (the dataset is one pre-cropped face per image), so this is
really a classification label wearing a detection format -- we only need the class id.
"""
import os

from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


class AffectNetYolo(Dataset):
    def __init__(self, root, split, transform=None):
        img_dir = os.path.join(root, split, 'images')
        label_dir = os.path.join(root, split, 'labels')
        self.transform = transform
        self.samples = []
        for fname in os.listdir(img_dir):
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(label_dir, stem + '.txt')
            if not os.path.isfile(label_path):
                continue
            with open(label_path) as fh:
                line = fh.readline().strip()
            if not line:
                continue
            cls = int(line.split()[0])
            self.samples.append((os.path.join(img_dir, fname), cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, cls

    def class_counts(self):
        counts = [0] * len(CLASS_NAMES)
        for _, cls in self.samples:
            counts[cls] += 1
        return counts
