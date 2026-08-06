import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import resample
from math import exp

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'BATCH_SIZE': 16,
    'EPOCHS': 32,
    'PHASE1_RES': 224, 'PHASE1_END': 10,
    'PHASE2_RES': 288, 'PHASE2_END': 20,
    'PHASE3_RES': 352, 'PHASE3_END': 32,
    'LR_BACKBONE': 1e-4,
    'LR_HEAD': 1e-3,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'TRAIN_IMG': 'data/DUTS-TR/image',
    'TRAIN_MASK': 'data/DUTS-TR/mask',
    'TEST_IMG': 'data/DUTS-TE/image',
    'TEST_MASK': 'data/DUTS-TE/mask'
}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

seed_everything()

# ==========================================
# 2. HYBRID LOSS (From Model 1)
# ==========================================
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

# ==========================================
# 3. DATASET (From Model 2)
# ==========================================
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
            if not os.path.exists(os.path.join(path, 'image')): continue
            imgs = [f.split('.')[0] for f in os.listdir(os.path.join(path, 'image')) if f.endswith('.jpg')]
            with open(os.path.join(path, 'train.txt' if 'TR' in path else 'test.txt'), 'w') as f:
                f.write('\n'.join(imgs))
    print(" Environment Ready.")

class SODDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True, resolution=352):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = [f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.is_train = is_train
        self.resolution = resolution
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def set_resolution(self, res):
        self.resolution = res

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, f"{img_id}.jpg")
       
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.resolution, self.resolution), resample=Image.BILINEAR)
        mask = mask.resize((self.resolution, self.resolution), resample=Image.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = image.rotate(angle, resample=Image.BILINEAR)
                mask = mask.rotate(angle, resample=Image.NEAREST)

        img_t = transforms.ToTensor()(image)
        mask_t = transforms.ToTensor()(mask)
        img_t = self.normalize(img_t)
       
        return img_t, mask_t

# ==========================================
# 4. ARCHITECTURE (Minimalist FPN from Model 2)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SimpleRefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
       
        self.lat4 = nn.Conv2d(2048, 64, 1)
        self.lat3 = nn.Conv2d(1024, 64, 1)
        self.lat2 = nn.Conv2d(512, 64, 1)
        self.lat1 = nn.Conv2d(256, 64, 1)

        self.smooth4 = ConvBlock(64, 64)
        self.smooth3 = ConvBlock(64, 64)
        self.smooth2 = ConvBlock(64, 64)
        self.smooth1 = ConvBlock(64, 64)

        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        input_shape = x.shape[2:]
       
        x0 = self.resnet.conv1(x); x0 = self.resnet.bn1(x0); x0 = self.resnet.relu(x0); x0 = self.resnet.maxpool(x0)
        c1 = self.layer1(x0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.smooth4(self.lat4(c4))
        p4_up = F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=True)
        p3 = self.smooth3(self.lat3(c3) + p4_up)
       
        p3_up = F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=True)
        p2 = self.smooth2(self.lat2(c2) + p3_up)
       
        p2_up = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=True)
        p1 = self.smooth1(self.lat1(c1) + p2_up)

        pred = self.out(p1)
        pred = F.interpolate(pred, size=input_shape, mode='bilinear', align_corners=True)
       
        return pred

# ==========================================
# 5. METRICS ENGINE
# ==========================================
class Evaluator:
    def __init__(self):
        self.mae_list = []
        self.f1_list = []
        self.beta2 = 0.3

    def update(self, pred, gt):
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        gt = gt.squeeze().cpu().numpy()

        self.mae_list.append(np.mean(np.abs(pred - gt)))

        thresh = 2 * pred.mean()
        if thresh > 1: thresh = 1
       
        pred_bin = pred > thresh
        gt_bin = gt > 0.5
       
        tp = (pred_bin & gt_bin).sum()
        prec = tp / (pred_bin.sum() + 1e-8)
        rec = tp / (gt_bin.sum() + 1e-8)
       
        f_score = (1 + self.beta2) * prec * rec / (self.beta2 * prec + rec + 1e-8)
        self.f1_list.append(f_score)

    def get_results(self):
        mae = np.array(self.mae_list)
        f1 = np.array(self.f1_list)

        def get_ci(data):
            means = [np.mean(resample(data)) for _ in range(1000)]
            return np.mean(data), np.percentile(means, 2.5), np.percentile(means, 97.5)

        m_mu, m_lo, m_hi = get_ci(mae)
        f_mu, f_lo, f_hi = get_ci(f1)
       
        return {
            "MAE": (m_mu, m_lo, m_hi),
            "F1": (f_mu, f_lo, f_hi)
        }

# ==========================================
# 6. TRAINING LOOP (PSAR + Hybrid Loss)
# ==========================================
def train_progressive():
    curr_res = CONFIG['PHASE1_RES']
    train_ds = SODDataset(CONFIG['TRAIN_IMG'], CONFIG['TRAIN_MASK'], is_train=True, resolution=curr_res)
    train_dl = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)
   
    model = SimpleRefineNet().to(CONFIG['DEVICE'])
    criterion = HybridLoss().to(CONFIG['DEVICE'])
   
    optimizer = torch.optim.AdamW([
        {'params': model.resnet.parameters(), 'lr': CONFIG['LR_BACKBONE']},
        {'params': model.lat4.parameters(), 'lr': CONFIG['LR_HEAD']},
        {'params': model.lat3.parameters(), 'lr': CONFIG['LR_HEAD']},
        {'params': model.lat2.parameters(), 'lr': CONFIG['LR_HEAD']},
        {'params': model.lat1.parameters(), 'lr': CONFIG['LR_HEAD']},
        {'params': model.smooth4.parameters(), 'lr': CONFIG['LR_HEAD']},
        {'params': model.smooth3.parameters(), 'lr': CONFIG['LR_HEAD']},
        {'params': model.smooth2.parameters(), 'lr': CONFIG['LR_HEAD']},
        {'params': model.smooth1.parameters(), 'lr': CONFIG['LR_HEAD']},
        {'params': model.out.parameters(), 'lr': CONFIG['LR_HEAD']},
    ], weight_decay=1e-4)
   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'])

    print(f"🚀 STARTING Model 3: Progressive Curriculum + Hybrid Loss")
    print(f"Initial Resolution: {curr_res}x{curr_res}")

    for epoch in range(CONFIG['EPOCHS']):
        new_res = None
        if epoch == CONFIG['PHASE1_END']: new_res = CONFIG['PHASE2_RES']
        elif epoch == CONFIG['PHASE2_END']: new_res = CONFIG['PHASE3_RES']
       
        if new_res:
            print(f"\n⚡ CURRICULUM STEP: Upgrading Resolution {curr_res} -> {new_res}")
            curr_res = new_res
            train_ds.set_resolution(curr_res)
            train_dl = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)

        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}/{CONFIG['EPOCHS']} [{curr_res}px]", leave=False)

        for img, mask in pbar:
            img, mask = img.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
           
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
       
    torch.save(model.state_dict(), "best_model3.pth")
    print("✅ Training Complete.")
    return model

def evaluate_standardized(model):
    print("\n📊 Evaluating Model 3 on DUTS-TE (Standard 352x352)...")
    test_ds = SODDataset(CONFIG['TEST_IMG'], CONFIG['TEST_MASK'], is_train=False, resolution=352)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
   
    model.eval()
    evaluator = Evaluator()

    with torch.no_grad():
        import cv2
        os.makedirs('out/preds_model3', exist_ok=True)
        for i, (img, mask) in enumerate(tqdm(test_dl)):
            img, mask = img.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
            pred = model(img)
            evaluator.update(pred, mask)
            
            pred_np = torch.sigmoid(pred[0]).squeeze().cpu().numpy()
            img_name = test_ds.ids[i] + ".png"
            cv2.imwrite(os.path.join('out/preds_model3', img_name), (pred_np * 255).astype(np.uint8))

    metrics = evaluator.get_results()
   
    print("\n" + "="*50)
    print("🏆 MODEL 3 (HYBRID LOSS + PSAR) RESULTS (95% CI)")
    print("="*50)
    print(f"Algorithm    | MAE (Lower is better) | Adp F-Measure (Higher is better)")
    print(f"-------------|-----------------------|------------------------------")
    print(f"F3Net (Ref)  | 0.035                 | 0.840")
    print(f"Model 3      | {metrics['MAE'][0]:.4f} ± {(metrics['MAE'][2]-metrics['MAE'][1])/2:.4f} | {metrics['F1'][0]:.4f} ± {(metrics['F1'][2]-metrics['F1'][1])/2:.4f}")
    print("="*50)

if __name__ == "__main__":
    setup_environment()
    if os.path.exists("best_model3.pth"):
        print("Found existing trained Model 3. Loading for evaluation...")
        model = SimpleRefineNet().to(CONFIG['DEVICE'])
        model.load_state_dict(torch.load("best_model3.pth", map_location=CONFIG['DEVICE']))
        evaluate_standardized(model)
    else:
        model = train_progressive()
        evaluate_standardized(model)
