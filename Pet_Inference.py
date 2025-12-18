import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


# The Net class must match V3 exactly
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def predict_my_pet(img_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    net = Net().to(device)
    net.load_state_dict(torch.load('./cifar_net_v3.pth'))
    net.eval()

    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Prep the image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        img = Image.open(img_name).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = net(img_t)
            # Convert outputs to probabilities (0% to 100%)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        print(f"\n--- Inference Result ---")
        print(f"Image: {img_name}")
        print(f"AI Guess: {classes[predicted[0]]}")
        print(f"Confidence: {confidence.item() * 100:.2f}%")
        print(f"------------------------\n")

    except FileNotFoundError:
        print(f"Error: Could not find '{img_name}'. Make sure it's in the same folder as this script!")


if __name__ == '__main__':
    # Change 'pet.jpg' to the actual filename of your pet's photo
    predict_my_pet('C:/Users/johnp/PycharmProjects/AI_Vision/PicsToIdentify/rufus3.jpg')