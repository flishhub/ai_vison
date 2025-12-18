import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ==========================================
# 1. DEEPER NEURAL NETWORK (V3)
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Layer 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Layer 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)  # Prevents memorization
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Input: 32x32 -> Conv -> Pool: 16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 16x16 -> Conv -> Pool: 8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 8x8 -> Conv -> Pool: 4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n--- AI Vision V3: The Accuracy Push ---")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")

    # --- STEP B: Data Augmentation ---
    # This makes the model "robust" by showing it flipped/shifted versions of the data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 64  # Better for deeper models

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- STEP C: Initialize & Optimize ---
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    # Using Adam optimizer: It's "smarter" than SGD for deep networks
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # --- STEP D: Training Loop ---
    epochs = 15
    for epoch in range(epochs):
        net.train()  # Set to training mode (enables Dropout/BatchNorm)
        running_loss = 0.0
        pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                pbar.set_postfix({'loss': f'{running_loss / (i + 1):.4f}'})

    # --- STEP E: Final Accuracy ---
    net.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nFinal Accuracy: {100 * correct // total}%')
    torch.save(net.state_dict(), './cifar_net_v3.pth')


if __name__ == '__main__':
    main()