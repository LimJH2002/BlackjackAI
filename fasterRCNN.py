import ast  # For loading previous training data
import os  # For reading previously saved model
import time
import json  # For parsing COCO annotations

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        """
        Args:
            images_dir (str): Path to the directory containing images.
            annotations_file (str): Path to the COCO annotations JSON file.
            transforms: Transforms to apply to images and annotations.
        """
        super().__init__()
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Load annotations
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)
        
        # Create image-to-annotation mapping
        self.image_data = {img["id"]: img for img in self.annotations["images"]}
        self.annotations_by_image = {}
        for ann in self.annotations["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # Map categories to IDs
        self.category_map = {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
        self.image_ids = list(self.image_data.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.image_data[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        annotations = self.annotations_by_image.get(img_id, [])
        boxes = []
        labels = []
        for ann in annotations:
            x_min, y_min, width, height = ann["bbox"]
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])
        
        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target


def get_model(num_classes):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def evaluate(model, data_loader, device, transform):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [transform(img).to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(imgs)

            for target, prediction in zip(targets, predictions):
                boxes_pred = prediction['boxes']
                scores_pred = prediction['scores']
                boxes_gt = target["boxes"]

                for t, p in zip(boxes_gt, boxes_pred):
                    iou = compute_iou(t, p)
                    if iou > 0.5:
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    num_classes = 52
    batch_size = 16

    train_dataset = COCODataset(
        "./data/train/images",
        "./data/train/annotations_coco.json",
        transforms=None,
    )
    val_dataset = COCODataset(
        "./data/valid/images",
        "./data/valid/annotations_coco.json",
        transforms=None,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda")
    model = get_model(num_classes)
    # Freeze the backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Fine-tune the ROI heads
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    # Freeze BatchNorm layers for stability
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()

    # Define optimizer for trainable parameters only
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001, 
        weight_decay=0.0001
    )
    model.to(device)

    PATH = './mobilenet_card_detector.pth'
    
    # Define optimizer and transform
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    transform = transforms.ToTensor()
    num_epochs = 5

    # Load pre-trained model and optimizer state if available
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Training on {device}")
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")

    # Training loop
    total_start_time = time.time()

    for epoch in range(num_epochs):
        # Load previous results if they exist
        losses = []
        accs = []
        if os.path.exists("mobilenet_results.txt"):
            with open("mobilenet_results.txt", mode="r") as file:
                lines = file.readlines()
                losses = ast.literal_eval(lines[0][len("Training losses: ") :].strip())
                accs = ast.literal_eval(lines[1][len("Testing accuracies: ") :].strip())

        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        batch_count = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for batch_idx, (imgs, targets) in enumerate(train_data_loader):
            batch_start_time = time.time()

            imgs = [transform(img).to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(imgs, targets)
            current_loss = sum(loss for loss in loss_dict.values())
            current_loss.backward()
            optimizer.step()

            epoch_loss += current_loss.item()
            batch_count += 1

            if (batch_idx + 1) % 5 == 0:
                avg_loss = epoch_loss / batch_count
                batch_time = time.time() - batch_start_time
                print(
                    f"Batch {batch_idx + 1}/{len(train_data_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {batch_time:.1f}s"
                )

        # Calculate metrics
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float("inf")
        losses.append(avg_epoch_loss)

        # Evaluate
        accuracy = evaluate(model, val_data_loader, device, transform)
        accs.append(accuracy * 100)

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Loss: {avg_epoch_loss:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Time: {epoch_time:.1f}s")

        # Save results and model every epoch
        with open("mobilenet_results.txt", mode="w") as file:
            file.write("Training losses: " + str(losses) + "\n")
            file.write("Testing accuracies: " + str(accs) + "\n")
        
        torch.save(
            {
                "epoch": len(losses) + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "accuracy": accuracy,
            },
            PATH,
        )

    # Training complete
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(accs)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig("mobilenet_results.png")
    plt.close()
