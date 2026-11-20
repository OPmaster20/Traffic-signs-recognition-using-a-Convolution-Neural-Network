import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import dmg
import predictor
from model.ResNet18 import ResNet,BasicBlock
from labels import class_labels as cls
from sklearn.metrics import classification_report
import os
import pandas as pd
from PIL import Image
from dmg.ShowImage import DI

# download dataset
def download():
    # Download local path
    output_path = './gtsrb'
    # Download latest version
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
                                      output_dir=output_path)
    print("Path to dataset files:", path)

# train model
def train():
    # Cuda loading (training on the GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # training times
    epochs = 10
    # Image Preprocessing
    transform = transforms.Compose([
        # resize images
        transforms.Resize((64, 64)),
        # rotation 15 degree
        #transforms.RandomRotation(15),
        # Randomly adjust the brightness and contrast to simulate different lighting conditions
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # transfer to tensor
        transforms.ToTensor(),
        # Feature scaling
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # train dataset
    train_dataset = datasets.ImageFolder(root="gtsrb/Train", transform=transform)
    # mini batch loading - each time divided into 64 and randomly shuffle
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8,pin_memory=True)

    # output some information about train set
    print("number of classes:", len(train_dataset.classes))
    print("classes label:", train_dataset.classes)
    print("class_to_idx:", train_dataset.class_to_idx)

    # Resnet 18
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=43)
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()
    # update parameters
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # record loss
    epoch_losses = []
    epoch_accs = []

    model.train()

    # mini batch
    for epoch in range(epochs):
        running_loss = 0.0

        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total

        epoch_losses.append(avg_loss)
        epoch_accs.append(acc)

        print(f"Epoch {epoch + 1}/{epochs}, Loss={avg_loss:.4f}, Acc={acc:.2f}%")

    print(f"Final Train Acc={acc:.2f}%")
    dmg.plot_metrics(epoch_losses, epoch_accs)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_labels": cls,
        "num_classes": 43,
        "input_size": 64,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    }, "gtsrb_resnet18.pth")
    print("Model saved to gtsrb_resnet18.pth")


if __name__ == "__main__":
     #dmg.img_dataset()
     #DI().show_random_images()
     train()
     #download()
