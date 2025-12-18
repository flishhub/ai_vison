import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests


def main():
    # 1. Setup hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the Pro Model (ResNet-50)
    # This downloads the 'weights' (the brain) from PyTorch the first time
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device)
    model.eval()

    # 3. Get the 1,000 label names
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    categories = requests.get(labels_url).text.splitlines()

    # 4. Pro-level image prep
    # Unlike our 32x32 model, this uses 224x224 for much higher detail
    preprocess = weights.transforms()

    # 5. Load Rufus!
    img_path = r'C:/Users/johnp/PycharmProjects/AI_Vision/PicsToIdentify/rufus3.jpg'

    try:
        img = Image.open(img_path).convert('RGB')
        batch = preprocess(img).unsqueeze(0).to(device)

        # 6. Run the math
        with torch.no_grad():
            prediction = model(batch).squeeze(0)
            confidences = torch.nn.functional.softmax(prediction, dim=0)

            # Get the top 3 guesses
            top3_prob, top3_idx = torch.topk(confidences, 3)

        print(f"\n--- PRO RESULTS FOR RUFUS ---")
        for i in range(3):
            print(f"{i + 1}. {categories[top3_idx[i]]}: {top3_prob[i].item() * 100:.2f}%")
        print(f"-----------------------------\n")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()