from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score

# CNN part, code copied from https://github.com/hiroonwijekoon/pytorch-cnn-playing-cards-classifier
class CardClassifierCNN(nn.Module):
    def __init__(self, num_classes=53):
        super(CardClassifierCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU() # Activation functions
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjust input size based on image size
        self.relu3 = nn.ReLU() # Activation functions
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self,x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 128 * 32 * 32)  # Adjust based on input image size
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) # no activation function in the end
        return x

PATH = './trained_model.pth'
model = CardClassifierCNN()
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()
true_labels = []
predicted_labels = []

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

batch_size = 32
test_folder = './dataset/test/'
test_dataset = ImageFolder(test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# get CNN accuracy
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)  # Get the class index with the highest probability
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predictions.cpu().numpy())






# YOLO part
yolo_model = YOLO('./models/yolov8s-batchsize(-1)-epoch150/runs/detect/train/weights/best.pt')
yolo_model.to(device)
true_labels_YOLO = []
pred_labels_YOLO = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # size [3,128,128] -> [3,320,320]
        images = F.interpolate(images, size=(320, 320), mode='bilinear', align_corners=False)

        # run YOLO
        results = yolo_model(images, stream=False)
        
        predictions = []
        for r in results:
            if len(r.boxes) > 0:
                predictions.append(int(r.boxes[0].cls[0]))
            else:
                predictions.append(-1)  # no detection

        # Append results
        true_labels_YOLO.extend(labels.cpu().tolist())
        pred_labels_YOLO.extend(predictions)

print("\n\n\n\n\n")
# Calculate CNN accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

# Calculate YOLO accuracy
accuracy = accuracy_score(true_labels_YOLO, pred_labels_YOLO)
print(f"YOLO Accuracy: {accuracy * 100:.2f}%\n")