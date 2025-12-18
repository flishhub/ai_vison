import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm  # Professional progress bar


# ==========================================
# 1. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==========================================
# 2. MAIN EXECUTION ENGINE
# ==========================================
def main():
    # --- STEP A: Accelerated Computing Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n--- AI Vision V2: 10-Epoch Sprint ---")
    print(f"Hardware: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"-------------------------------------\n")

    # --- STEP B: Data Pipeline ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Increased batch size to 32 to saturate the GPU more effectively
    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # --- STEP C: Initialize Model & Optimizer ---
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # --- STEP D: Optimized Training Loop ---
    epochs = 25
    print(f"Starting {epochs} training epochs...")

    for epoch in range(epochs):
        running_loss = 0.0
        # Progress Bar setup

        pbar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update the progress bar with live Loss data
            if i % 100 == 99:
                pbar.set_postfix({'loss': f'{running_loss / (i + 1):.4f}'})

    print('\nTraining Complete! ✅')

    # --- STEP E: Save the "Brain" ---
    PATH = './cifar_net_v2.pth'
    torch.save(net.state_dict(), PATH)
    print(f"Model saved to {PATH}")

    # --- STEP F: Evaluation ---
    print("\nCalculating Final Accuracy...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = 100 * correct // total
    print(f'Total Accuracy on 10,000 test images: {final_acc}%')

    if final_acc > 60:
        print("Excellent! Your model is now significantly more accurate.")


# ==========================================
# 3. ENTRY POINT
# ==========================================
if __name__ == '__main__':
    main()