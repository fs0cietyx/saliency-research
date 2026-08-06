import os
import sys
import shutil
import datetime
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import exp
import warnings
import scipy.stats

# Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------------------------------------
# 0. HELPER: SSIM & HYBRID LOSS COMPONENTS
# -----------------------------------------------------------------------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim_func(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2; C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class HybridLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(HybridLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        pred_sigmoid = torch.sigmoid(pred)
        
        if self.window.device != pred.device: self.window = self.window.to(pred.device)
        ssim_val = ssim_func(pred_sigmoid, target, self.window, self.window_size, self.channel, self.size_average)
        ssim_loss = 1 - ssim_val
        
        inter = (pred_sigmoid * target).sum(dim=(2, 3))
        union = (pred_sigmoid + target).sum(dim=(2, 3)) - inter
        iou = (inter + 1) / (union + 1)
        iou_loss = 1 - iou.mean()
        
        return bce_loss + ssim_loss + iou_loss

# -----------------------------------------------------------------------------------------
# 1. ENVIRONMENT & DATA PREP
# -----------------------------------------------------------------------------------------
def setup_environment():
    print(" Setting up Environment...")
    os.makedirs('out', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    if not os.path.exists('data/DUTS-TR'):
        print(" Downloading DUTS Dataset...")
        os.system('curl -sL http://saliencydetection.net/duts/download/DUTS-TR.zip -o data/DUTS-TR.zip')
        os.system('curl -sL http://saliencydetection.net/duts/download/DUTS-TE.zip -o data/DUTS-TE.zip')
        print(" Extracting Data...")
        os.system('unzip -q data/DUTS-TR.zip -d data/')
        os.system('unzip -q data/DUTS-TE.zip -d data/')
        
        tr_root = 'data/DUTS-TR'
        te_root = 'data/DUTS-TE'
        
        if os.path.exists(f'{tr_root}/DUTS-TR-Image'):
            os.system(f'mv {tr_root}/DUTS-TR-Image/* {tr_root}/image/ 2>/dev/null || mkdir -p {tr_root}/image && mv {tr_root}/DUTS-TR-Image/* {tr_root}/image/')
            os.system(f'mv {tr_root}/DUTS-TR-Mask/* {tr_root}/mask/ 2>/dev/null || mkdir -p {tr_root}/mask && mv {tr_root}/DUTS-TR-Mask/* {tr_root}/mask/')
        
        if os.path.exists(f'{te_root}/DUTS-TE-Image'):
            os.system(f'mv {te_root}/DUTS-TE-Image/* {te_root}/image/ 2>/dev/null || mkdir -p {te_root}/image && mv {te_root}/DUTS-TE-Image/* {te_root}/image/')
            os.system(f'mv {te_root}/DUTS-TE-Mask/* {te_root}/mask/ 2>/dev/null || mkdir -p {te_root}/mask && mv {te_root}/DUTS-TE-Mask/* {te_root}/mask/')

        for path in [tr_root, te_root]:
            imgs = [f.split('.')[0] for f in os.listdir(os.path.join(path, 'image')) if f.endswith('.jpg')]
            with open(os.path.join(path, 'train.txt' if 'TR' in path else 'test.txt'), 'w') as f:
                f.write('\n'.join(imgs))
    print(" Environment Ready.")

class DutsDataset(Dataset):
    def __init__(self, root, mode='train', augment=True):
        self.root = root
        self.mode = mode
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        with open(os.path.join(root, 'train.txt' if mode=='train' else 'test.txt'), 'r') as lines:
            self.samples = [line.strip() for line in lines]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(os.path.join(self.root, 'image', name+'.jpg'))[:,:,::-1]
        mask  = cv2.imread(os.path.join(self.root, 'mask', name+'.png'), 0)
        
        if self.mode == 'train' and self.augment:
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                angle = random.randint(-10, 10)
                M = cv2.getRotationMatrix2D((128, 128), angle, 1)
                image = cv2.warpAffine(image, M, (256, 256))
                mask = cv2.warpAffine(mask, M, (256, 256), flags=cv2.INTER_NEAREST)
        else:
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        mask = mask.astype(np.float32) / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

class TestData(Dataset):
    def __init__(self, root):
        self.root = root
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        with open(os.path.join(root, 'test.txt'), 'r') as lines:
            self.samples = [line.strip() for line in lines]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        name  = self.samples[idx]
        image_path = os.path.join(self.root, 'image', name+'.jpg')
        image = cv2.imread(image_path)[:,:,::-1]
        shape = image.shape[:2]
        
        image = cv2.resize(image, (256, 256))
        # COMPLETED THE MISSING PART
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, name, shape

# -----------------------------------------------------------------------------------------
# 2. MODEL
# -----------------------------------------------------------------------------------------
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.layer1 = self.backbone.layer1 
        self.layer2 = self.backbone.layer2 
        self.layer3 = self.backbone.layer3 
        self.layer4 = self.backbone.layer4 

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout(0.5), nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout(0.5), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout(0.5), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.layer0(x); x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        x = self.up1(x4)
        if x.size() != x3.size(): x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1); x = self.conv1(x)
        x = self.up2(x)
        if x.size() != x2.size(): x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1); x = self.conv2(x)
        x = self.up3(x)
        if x.size() != x1.size(): x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1); x = self.conv3(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.final(x)
        return x

def calculate_uncertainty(probs):
    epsilon = 1e-8
    entropy = -probs * torch.log(probs + epsilon) - (1 - probs) * torch.log(1 - probs + epsilon)
    return entropy

# -----------------------------------------------------------------------------------------
# 3. TRAINING LOOP
# -----------------------------------------------------------------------------------------
def train_model():
    setup_environment()
    EPOCHS = 15
    BATCH_SIZE = 16
    LR = 1e-4
    
    train_ds = DutsDataset(root='data/DUTS-TR', mode='train')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    print(f"Data Loaded: {len(train_ds)} training images")
    
    model = ResNetUNet(num_classes=1).to(device)
    criterion = HybridLoss().to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR, weight_decay=1e-5)
    
    print(" Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (imgs, masks) in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.5f}")
        
        if (epoch+1) % 4 == 0 or (epoch+1) == EPOCHS:
            torch.save(model.state_dict(), f'out/resnet_unet_epoch_{epoch+1}.pth')
            
    torch.save(model.state_dict(), 'out/resnet_unet_final.pth')
    print("Training Complete.")

# -----------------------------------------------------------------------------------------
# 4. EVALUATION LOOP
# -----------------------------------------------------------------------------------------
def test_model():
    print(" Starting Testing...")
    test_ds = TestData(root='data/DUTS-TE')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = ResNetUNet(num_classes=1).to(device)
    if os.path.exists('out/resnet_unet_final.pth'):
        model.load_state_dict(torch.load('out/resnet_unet_final.pth', map_location=device))
        print("Model weights loaded.")
    else:
        print("Warning: Model weights not found. Please train the model first.")
        return
        
    model.eval()

    os.makedirs('out/predictions', exist_ok=True)
    os.makedirs('out/uncertainty', exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        for i, (img, name, shape) in pbar:
            img = img.to(device)
            pred = model(img)
            pred = torch.sigmoid(pred)
            
            # Convert prediction to numpy
            pred_np = pred.squeeze().cpu().numpy()
            
            # Calculate uncertainty
            uncertainty = calculate_uncertainty(pred)
            uncertainty_np = uncertainty.squeeze().cpu().numpy()
            
            h, w = shape[0].item(), shape[1].item()
            
            # Resize and save prediction
            pred_resized = cv2.resize(pred_np, (w, h))
            pred_img = (pred_resized * 255).astype(np.uint8)
            cv2.imwrite(f'out/predictions/{name[0]}.png', pred_img)
            
            # Resize and save uncertainty map (normalized for visualization)
            unc_resized = cv2.resize(uncertainty_np, (w, h))
            unc_img = (unc_resized / np.max(unc_resized) * 255).astype(np.uint8)
            cv2.applyColorMap(unc_img, cv2.COLORMAP_JET)
            cv2.imwrite(f'out/uncertainty/{name[0]}_unc.png', unc_img)
            
    print("Testing Complete. Predictions and Uncertainty maps saved in 'out/'")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_model()
    else:
        train_model()
        test_model()
