import torchvision
import torch
import os # for reading previously saved model
import ast # for loading the previous training data
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # pre-trained faster RCNN model
from CardDataset import CardDataset

def get_model(num_classes):
    # Load a pretrained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the classifier head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def evaluate(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't compute gradients during evaluation
        for imgs, targets in data_loader:
            imgs = [transform(img).to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Perform inference
            predictions = model(imgs)
            
            # Here, we'll assume that you want to count how many boxes have IoU > 0.5
            for target, prediction in zip(targets, predictions):
                # Check the number of correct predictions (IoU > 0.5)
                for t, p in zip(target['boxes'], prediction['boxes']):
                    iou = compute_iou(t, p)  # Define your IoU computation
                    if iou > 0.5:  # IoU threshold of 0.5
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

def compute_iou(box1, box2):
    # Compute Intersection over Union (IoU) between two boxes
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute intersection
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    # Compute IoU
    return inter_area / union_area if union_area > 0 else 0

if __name__ == '__main__':

    # get the previous training losses and accuracies
    losses = []
    accs = []
    if os.path.exists("results.txt"):
        # read the lines from previous results
        with open("results.txt", mode='r') as file:
            lines = file.readlines()
            
            # assuming the first line is losses and the second line is accuracies
            losses = ast.literal_eval(lines[0][len("Training losses: "):].strip())
            accs = ast.literal_eval(lines[1][len("Testing accuracies: "):].strip())


    # To do: Re-initialize the model architecture
    # model = get_model(num_classes)  # Ensure the architecture matches the one used during training
    # model.load_state_dict(torch.load("faster_rcnn_card_detector.pth"))
    # model.to(device)


    # Initialize dataset, model, and data loader
    num_classes = 52  # 52 card classes
    train_dataset = CardDataset('./data/train/images', './data/train/labels_pascal')
    val_dataset = CardDataset('./data/valid/images', './data/valid/labels_pascal')

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes).to(device)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for imgs, targets in train_data_loader:
            transform = transforms.ToTensor()
            imgs = [transform(img).to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Training step
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Calculate testing accuracy
        accuracy = evaluate(model, val_data_loader, device)
        accs.append(accuracy * 100)
        losses.append(losses.item())
        print(f"Epoch {epoch+1}, Loss: {losses.item()}, Validation Accuracy: {accuracy * 100:.2f}%")

    # plot losses and accuracies
    plt.plot(losses)
    plt.title("Training loss")
    plt.savefig("training_loss.png")
    plt.clf() # clears the graph

    plt.plot(accs)
    plt.title("Testing Accuracy")
    plt.savefig("testing_accuracy.png")

    # save losses and accuracies
    with open("results.txt", mode='w') as file:
        file.write("Training losses: " + str(losses) + "\n")
        file.write("Testing accuracies: " + str(accs) + "\n")

    # save the model
    torch.save(model.state_dict(), "faster_rcnn_card_detector.pth")
