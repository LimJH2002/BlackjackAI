import ast  # for loading the previous training data
import os  # for reading previously saved model
import time

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from CardDataset import CardDataset


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
                for t, p in zip(target["boxes"], prediction["boxes"]):
                    iou = compute_iou(t, p)
                    if iou > 0.5:
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    # Initialize dataset and model
    num_classes = 52
    batch_size = 16

    train_dataset = CardDataset("./data/train/images", "./data/train/labels_pascal")
    val_dataset = CardDataset("./data/valid/images", "./data/valid/labels_pascal")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # Parallel data loading
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

    device = torch.device("cpu")
    model = get_model(num_classes).to(device)

    # Load previous results if they exist
    losses = []
    accs = []
    if os.path.exists("mobilenet_results.txt"):
        with open("mobilenet_results.txt", mode="r") as file:
            lines = file.readlines()
            losses = ast.literal_eval(lines[0][len("Training losses: ") :].strip())
            accs = ast.literal_eval(lines[1][len("Testing accuracies: ") :].strip())

    # Define optimizer and transform
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    transform = transforms.ToTensor()
    num_epochs = 5

    print(f"Training on CPU with {torch.get_num_threads()} threads")
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")

    # Training loop
    total_start_time = time.time()

    for epoch in range(num_epochs):
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

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "accuracy": accuracy,
            },
            f"mobilenet_checkpoint_epoch_{epoch+1}.pth",
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

    # Save results
    with open("mobilenet_results.txt", mode="w") as file:
        file.write("Training losses: " + str(losses) + "\n")
        file.write("Testing accuracies: " + str(accs) + "\n")

    # Save model
    torch.save(model.state_dict(), "mobilenet_card_detector.pth")
