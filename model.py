

# encoding: utf-8

"""
The main CheXNet model implementation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from read_data import ChestXrayDataSet
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Định nghĩa các hằng số
CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 16  # Giảm batch size để tiết kiệm bộ nhớ

def main():
    # Kiểm tra xem có GPU không
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Khởi tạo và tải mô hình
    model = DenseNet121(N_CLASSES).to(device)
    model = torch.nn.DataParallel(model).to(device)

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH, map_location=device)
        
        # Sửa tên các key trong state_dict để khớp với mô hình
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("norm.1", "norm1").replace("norm.2", "norm2").replace("conv.1", "conv1").replace("conv.2", "conv2")
            new_state_dict[new_key] = value
        
        # Tải state_dict đã sửa
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
        return

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    print(f"Dataset loaded with {len(test_dataset)} images")
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Khởi tạo tensor ground truth và dự đoán
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)

    # Chuyển sang chế độ đánh giá
    model.eval()

    # Sử dụng torch.no_grad() để tiết kiệm bộ nhớ và tăng tốc
    with torch.no_grad():
        for i, (inp, target) in enumerate(test_loader):
            print(f"Processing batch {i+1}/{len(test_loader)}")
            target = target.to(device)
            gt = torch.cat((gt, target), 0)

            bs, n_crops, c, h, w = inp.size()
            inp = inp.view(-1, c, h, w).to(device)
            output = model(inp)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean), 0)

    # Tính AUROC
    print("Computing AUROCs...")
    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.3f}'.format(CLASS_NAMES[i], AUROCs[i]))

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores."""
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

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

if __name__ == '__main__':
    main()