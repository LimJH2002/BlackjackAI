import os

import torch
from PIL import Image
from torchvision.transforms import functional as F


class CardDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(img_dir)))
        self.annotations = list(sorted(os.listdir(annotation_dir)))

        # Verify matching files
        if len(self.imgs) != len(self.annotations):
            print("WARNING: Number of images and annotations don't match!")

    def __getitem__(self, idx):
        try:
            # Load image
            img_path = os.path.join(self.img_dir, self.imgs[idx])
            img = Image.open(img_path).convert("RGB")

            # Load annotations
            annotation_path = os.path.join(self.annotation_dir, self.annotations[idx])
            boxes = []
            labels = []

            with open(annotation_path, "r") as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    labels.append(int(values[0]))
                    boxes.append(values[1:])

            # Convert to tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}

            if self.transforms:
                img = self.transforms(img)

            return img, target

        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            raise e

    def __len__(self):
        return len(self.imgs)
