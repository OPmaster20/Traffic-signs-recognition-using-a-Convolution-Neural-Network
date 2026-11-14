import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.ResNet18 import ResNet,BasicBlock
from sklearn.metrics import classification_report
import os
import pandas as pd
from PIL import Image
from ShowImg.ShowImage import DI

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
    # Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image Preprocessing
    transform = transforms.Compose([
        # resize images
        transforms.Resize((224, 224)),
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


    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=43)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    # mini batch
    for epoch in range(1):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    DI().show_random_images()
    #train()