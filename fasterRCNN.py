import json  # For parsing COCO annotations
import os  # For reading previously saved model
import ast # for loading the previous training data
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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
        self.category_map = {
            cat["id"]: cat["name"] for cat in self.annotations["categories"]
        }
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


def evaluate(
    model, data_loader, device, transform, iou_threshold=0.5, confidence_threshold=0.5
):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [transform(img).to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(imgs)

            for target, pred in zip(targets, predictions):
                # Get ground truth boxes and labels
                boxes_gt = target["boxes"]
                labels_gt = target["labels"]

                # Get predicted boxes, scores, and labels
                boxes_pred = pred["boxes"]
                scores_pred = pred["scores"]
                labels_pred = pred["labels"]

                # Filter predictions by confidence threshold
                mask = scores_pred >= confidence_threshold
                boxes_pred = boxes_pred[mask]
                labels_pred = labels_pred[mask]

                # Keep track of which gt boxes have been matched
                gt_matched = torch.zeros(len(boxes_gt), dtype=torch.bool)

                # For each predicted box
                for i in range(len(boxes_pred)):
                    pred_box = boxes_pred[i]
                    pred_label = labels_pred[i]

                    # Calculate IoU with all gt boxes
                    ious = torch.zeros(len(boxes_gt))
                    for j, gt_box in enumerate(boxes_gt):
                        ious[j] = compute_iou(pred_box, gt_box)

                    # Find best matching gt box
                    if len(ious) > 0:
                        best_match_idx = torch.argmax(ious).item()
                        best_match_iou = ious[best_match_idx]

                        # If IoU is good enough and classes match and gt box hasn't been matched
                        if (
                            best_match_iou >= iou_threshold
                            and pred_label == labels_gt[best_match_idx]
                            and not gt_matched[best_match_idx]
                        ):
                            correct += 1
                            gt_matched[best_match_idx] = True

                # Add total ground truth boxes to total
                total += len(boxes_gt)

    mAP = correct / total if total > 0 else 0
    return mAP


def collate_fn(batch):
    return tuple(zip(*batch))


def save_metrics(
    train_losses, val_accuracies, test_accuracy, filename="model_metrics.json"
):
    metrics = {
        "training_losses": train_losses,
        "validation_accuracies": val_accuracies,
        "test_accuracy": test_accuracy,
    }
    with open(filename, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    num_classes = 52
    batch_size = 16
    num_epochs = 15

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
    test_dataset = COCODataset(
        "./data/test/images",
        "./data/test/annotations_coco.json",
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
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
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

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.0001,
    )
    transform = transforms.ToTensor()

    model.to(device)

    PATH = "./mobilenet_card_detector.pth"
    BEST_MODEL_PATH = "./mobilenet_card_detector_best.pth"

    # Define optimizer and transform
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.0001,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2,
        verbose=True,
        min_lr=1e-7,
    )

    transform = transforms.ToTensor()

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
        train_losses = []
        val_accuracies = []
        if os.path.exists("mobilenet_results.txt"):
            with open("mobilenet_results.txt", mode="r") as file:
                lines = file.readlines()
                train_losses = ast.literal_eval(lines[0][len("Training losses: ") :].strip())
                val_accuracies = ast.literal_eval(lines[1][len("Testing accuracies: ") :].strip())

        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        batch_count = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # Training Phase
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

        # Calculate epoch metrics
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float("inf")
        train_losses.append(avg_epoch_loss)

        # Validation phase
        val_accuracy = evaluate(model, val_data_loader, device, transform)
        val_accuracies.append(val_accuracy * 100)

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}")

        best_val_accuracy = max(val_accuracies)/100
        if val_accuracy >= best_val_accuracy:
            print("Saving the best model...")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                },
                BEST_MODEL_PATH,
            )

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_epoch_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Time: {epoch_time:.1f}s")

        # Save results and model every epoch
        with open("mobilenet_results.txt", mode="w") as file:
            file.write("Training losses: " + str(train_losses) + "\n")
            file.write("Testing accuracies: " + str(val_accuracies) + "\n")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_epoch_loss,
                "val_accuracy": val_accuracy,
            },
            PATH,
        )

        # Plot results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 3, 2)
        plt.plot(val_accuracies)
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")

        plt.tight_layout()
        plt.savefig("training_results.png")

    # Training complete
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    if os.path.exists(BEST_MODEL_PATH):
        best_checkpoint = torch.load(BEST_MODEL_PATH)
        model.load_state_dict(best_checkpoint["model_state_dict"])

    print("\nPerforming final test evaluation...")
    test_accuracy = evaluate(model, test_data_loader, device, transform)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    plt.close()
