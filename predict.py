# encoding: utf-8

"""
CheXNet: Training and Prediction for Chest X-ray Disease Detection
Made by ShootingStar2608: dohongphuc260805@gmail.com
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from PIL import Image
import argparse

# Định nghĩa các hằng số
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TRAIN_IMAGE_LIST = './ChestX-ray14/labels/train_list.txt'
VAL_IMAGE_LIST = './ChestX-ray14/labels/val_list.txt'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 16
NUM_EPOCHS = 5  # Số epoch huấn luyện, có thể tăng nếu cần
LEARNING_RATE = 0.001

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.strip().split()
                image_name = items[0]
                label = list(map(float, items[1:1+N_CLASSES]))
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        image_name = os.path.join(self.data_dir, self.image_names[index])
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

class DenseNet121(nn.Module):
    """Model modified."""
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Khởi tạo mô hình
    model = DenseNet121(N_CLASSES).to(device)
    model = torch.nn.DataParallel(model).to(device)

    # Định nghĩa loss và optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Chuẩn bị dữ liệu
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=TRAIN_IMAGE_LIST, transform=train_transform)
    val_dataset = ChestXrayDataSet(data_dir=DATA_DIR, image_list_file=VAL_IMAGE_LIST, transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Huấn luyện mô hình
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss/len(val_loader):.4f}')

    # Lưu mô hình sau khi huấn luyện
    torch.save({'state_dict': model.state_dict()}, 'chexnet_trained.pth')
    print("Model saved as chexnet_trained.pth")

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Predicting on device: {device}")

    # Khởi tạo mô hình và tải trọng số đã huấn luyện
    model = DenseNet121(N_CLASSES).to(device)
    model = torch.nn.DataParallel(model).to(device)
    
    checkpoint = torch.load('chexnet_trained.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Tiền xử lý ảnh đầu vào
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    image = image.to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(image)
        probs = output.squeeze().cpu().numpy()

    # In kết quả
    print("\nPrediction Probabilities:")
    for i, (disease, prob) in enumerate(zip(CLASS_NAMES, probs)):
        print(f"{disease}: {prob:.3f}")

def main():
    parser = argparse.ArgumentParser(description="CheXNet: Train and Predict Diseases from Chest X-rays")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help="Mode: 'train' or 'predict'")
    parser.add_argument('--image', type=str, help="Path to the image for prediction (required if mode is 'predict')")
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image argument is required for predict mode")
            return
        if not os.path.exists(args.image):
            print(f"Error: Image path {args.image} does not exist")
            return
        predict_image(args.image)

if __name__ == '__main__':
    main()