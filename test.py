import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model.ResNet18 import ResNet, BasicBlock
from labels import class_labels  # {0: "Speed limit (20km/h)", ...}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GTSRBTestDataset(Dataset):
    """
    Dataset class specifically for loading the official GTSRB test set.
    Reads Test.csv, crops the Roi region, and applies preprocessing.
    """

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]

        # Join image path (Path in CSV is usually 'Test/00000.png' or '00000.png')
        img_path = os.path.join(self.root_dir, os.path.basename(str(row["Path"])))
        image = Image.open(img_path).convert("RGB")

        # Core step: crop ROI (keep only the traffic sign, remove background)
        x1, y1, x2, y2 = int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])
        image = image.crop((x1, y1, x2, y2))

        label = int(row["ClassId"])

        if self.transform:
            image = self.transform(image)

        return image, label


def build_idx_to_class_id(checkpoint: dict, num_classes: int = 43, use_imagefolder_order: bool = True) -> dict:
    """
    Build the mapping "model output class index -> true integer ClassId (0~42)".
    Solves the label misalignment caused by ImageFolder sorting in string lexicographical order ('0', '1', '10', ..., '2', ...).
    """
    # 1. Prefer reading class_to_idx / class_labels saved during training from the checkpoint
    for key in ("class_to_idx", "class_labels"):
        saved = checkpoint.get(key)
        if isinstance(saved, dict) and len(saved) == num_classes:
            first_k = next(iter(saved.keys()))
            first_v = next(iter(saved.values()))
            # Format: {'0': 0, '1': 1, '10': 2, ...}
            if isinstance(first_k, str) and first_k.isdigit():
                return {int(v): int(k) for k, v in saved.items()}
            # Format: {0: '0', 1: '1', 2: '10', ...}
            if isinstance(first_v, str) and first_v.isdigit():
                return {int(k): int(v) for k, v in saved.items()}

    # 2. If checkpoint did not save folder number mapping, and training used ImageFolder('0'..'42')
    if use_imagefolder_order:
        folder_names = sorted(str(i) for i in range(num_classes))
        return {idx: int(name) for idx, name in enumerate(folder_names)}

    # 3. Otherwise keep one-to-one correspondence
    return {i: i for i in range(num_classes)}


def evaluate_test_set(
    model_path="gtsrb_resnet18.pth",
    test_root="gtsrb/Test",
    csv_path="gtsrb/Test.csv",
    batch_size=64,
    use_imagefolder_order=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_path):
        print(f"Error: Model weight file not found: {model_path}")
        return

    # 1. Load model and training configuration (compatible with PyTorch 2.6+ and older versions)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = int(checkpoint.get("num_classes", 43))
        input_size = int(checkpoint.get("input_size", 64))
        mean = checkpoint.get("normalize_mean", IMAGENET_MEAN)
        std = checkpoint.get("normalize_std", IMAGENET_STD)
    else:
        state_dict = checkpoint
        checkpoint = {}
        num_classes = 43
        input_size = 64
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully. (input_size={input_size}, num_classes={num_classes})")
    print(f"Normalize mean={mean}, std={std}")

    # Build mapping from model output index to true ClassId
    idx_to_class_id = build_idx_to_class_id(
        checkpoint, num_classes=num_classes, use_imagefolder_order=use_imagefolder_order
    )
    print("Label mapping preview (model_idx -> true_ClassId):", {k: idx_to_class_id[k] for k in range(min(15, num_classes))})

    # 2. Define test set preprocessing (consistent with training validation set)
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 3. Load dataset
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please ensure the downloaded GTSRB dataset contains Test.csv.")
        return

    test_dataset = GTSRBTestDataset(root_dir=test_root, csv_file=csv_path, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test set size: {len(test_dataset)} images")

    # 4. Start inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Map model output index back to true integer ClassId (0~42)
            mapped_preds = [idx_to_class_id[int(p)] for p in predicted.cpu().tolist()]
            all_preds.extend(mapped_preds)
            all_labels.extend(labels.numpy().tolist())

            if (i + 1) % 20 == 0:
                print(f"Processed {i * batch_size + len(images)} / {len(test_dataset)} images...")

    # 5. Generate classification_report
    target_names = [class_labels[i] for i in range(num_classes)]
    all_class_ids = list(range(num_classes))

    print("\n" + "=" * 80)
    print("Classification Report on GTSRB Test Set:")
    print("=" * 80)

    report = classification_report(
        all_labels,
        all_preds,
        labels=all_class_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    with open("result/test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Report saved to test_report.txt")

    # 6. Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=all_class_ids)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - GTSRB Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig("result/confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    evaluate_test_set()