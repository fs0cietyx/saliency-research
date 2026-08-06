import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm  # Changed from tqdm.notebook to regular tqdm for terminal use
import matplotlib.pyplot as plt
from sklearn.utils import resample

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'BATCH_SIZE': 16, # Good balance for T4 GPU
    'EPOCHS': 32,
    # --- NOVELTY: PROGRESSIVE CURRICULUM ---
    'PHASE1_RES': 224, 'PHASE1_END': 10,  # Learn Global Context
    'PHASE2_RES': 288, 'PHASE2_END': 20,  # Learn Structure
    'PHASE3_RES': 352, 'PHASE3_END': 32,  # Refine Boundaries (Standard Benchmark)
    # ---------------------------------------
    'LR_BACKBONE': 1e-4,
    'LR_HEAD': 1e-3,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    # MODIFIED PATHS to match the local data directory structure set up in Model 1
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
# 2. DATASET (Dynamic Resolution)
# ==========================================
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
        # Handle png/jpg mismatch in dataset
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, f"{img_id}.jpg")
       
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Dynamic Resize
        image = image.resize((self.resolution, self.resolution), resample=Image.BILINEAR)
        mask = mask.resize((self.resolution, self.resolution), resample=Image.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                # Random rotation
                angle = random.uniform(-10, 10)
                image = image.rotate(angle, resample=Image.BILINEAR)
                mask = mask.rotate(angle, resample=Image.NEAREST)

        img_t = transforms.ToTensor()(image)
        mask_t = transforms.ToTensor()(mask)
       
        img_t = self.normalize(img_t)
       
        return img_t, mask_t

# ==========================================
# 3. ARCHITECTURE: Minimalist FPN
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
        # Backbone: ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.layer1 = self.resnet.layer1 # 256
        self.layer2 = self.resnet.layer2 # 512
        self.layer3 = self.resnet.layer3 # 1024
        self.layer4 = self.resnet.layer4 # 2048
       
        # Lateral Connections (1x1 Conv to reduce channels to 64)
        self.lat4 = nn.Conv2d(2048, 64, 1)
        self.lat3 = nn.Conv2d(1024, 64, 1)
        self.lat2 = nn.Conv2d(512, 64, 1)
        self.lat1 = nn.Conv2d(256, 64, 1)

        # Smooth Layers (3x3 Conv to mix features)
        self.smooth4 = ConvBlock(64, 64)
        self.smooth3 = ConvBlock(64, 64)
        self.smooth2 = ConvBlock(64, 64)
        self.smooth1 = ConvBlock(64, 64)

        # Final Prediction
        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        input_shape = x.shape[2:] # capture H, W
       
        # Encoder
        x0 = self.resnet.conv1(x); x0 = self.resnet.bn1(x0); x0 = self.resnet.relu(x0); x0 = self.resnet.maxpool(x0)
        c1 = self.layer1(x0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Decoder (Top-Down)
        p4 = self.smooth4(self.lat4(c4))
       
        p4_up = F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=True)
        p3 = self.smooth3(self.lat3(c3) + p4_up)
       
        p3_up = F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=True)
        p2 = self.smooth2(self.lat2(c2) + p3_up)
       
        p2_up = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=True)
        p1 = self.smooth1(self.lat1(c1) + p2_up)

        # Final Prediction
        pred = self.out(p1)
        # CRITICAL FIX: Ensure output matches input size exactly
        pred = F.interpolate(pred, size=input_shape, mode='bilinear', align_corners=True)
       
        return pred

# ==========================================
# 4. LOSS (Structure Aware - Same as F3Net)
# ==========================================
def structure_loss(pred, mask):
    # BCE (Pixel-level)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
   
    # IoU (Global structure)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
   
    return wbce + wiou.mean()

# ==========================================
# 5. METRICS ENGINE (Bootstrapped CI)
# ==========================================
class Evaluator:
    def __init__(self):
        self.mae_list = []
        self.f1_list = []
        self.beta2 = 0.3

    def update(self, pred, gt):
        # Convert logits to probability
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        gt = gt.squeeze().cpu().numpy()

        # MAE
        self.mae_list.append(np.mean(np.abs(pred - gt)))

        # Adaptive F-measure (AdpF)
        # Threshold = 2 * Mean Saliency
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

        # Bootstrapping for 95% CI
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
# 6. TRAINING (Progressive Curriculum)
# ==========================================
def train_progressive():
    # Initial Resolution (Phase 1)
    curr_res = CONFIG['PHASE1_RES']
    train_ds = SODDataset(CONFIG['TRAIN_IMG'], CONFIG['TRAIN_MASK'], is_train=True, resolution=curr_res)
    train_dl = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)
   
    model = SimpleRefineNet().to(CONFIG['DEVICE'])
   
    # Differential Learning Rates
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

    print(f"🚀 STARTING Progressive Structure-Aware Refinement (PSAR)")
    print(f"Initial Resolution: {curr_res}x{curr_res}")

    for epoch in range(CONFIG['EPOCHS']):
        # --- CURRICULUM LOGIC ---
        new_res = None
        if epoch == CONFIG['PHASE1_END']: new_res = CONFIG['PHASE2_RES']
        elif epoch == CONFIG['PHASE2_END']: new_res = CONFIG['PHASE3_RES']
       
        if new_res:
            print(f"\n⚡ CURRICULUM STEP: Upgrading Resolution {curr_res} -> {new_res}")
            curr_res = new_res
            train_ds.set_resolution(curr_res)
            train_dl = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)
        # ------------------------

        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}/{CONFIG['EPOCHS']} [{curr_res}px]", leave=False)

        for img, mask in pbar:
            img, mask = img.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
           
            optimizer.zero_grad()
            pred = model(img)
           
            # Loss
            loss = structure_loss(pred, mask)
           
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
       
    torch.save(model.state_dict(), "best_psar_model.pth")
    print("✅ Training Complete.")
    return model

# ==========================================
# 7. EVALUATION
# ==========================================
def evaluate_standardized(model):
    print("\n📊 Evaluating on DUTS-TE (Standard 352x352)...")
    # F3Net/BASNet standards use 352x352 for testing
    test_ds = SODDataset(CONFIG['TEST_IMG'], CONFIG['TEST_MASK'], is_train=False, resolution=352)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
   
    model.eval()
    evaluator = Evaluator()
    visuals = []

    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(test_dl)):
            img, mask = img.to(CONFIG['DEVICE']), mask.to(CONFIG['DEVICE'])
            pred = model(img)
            evaluator.update(pred, mask)

            if i < 4: # Save for visualization
                # Denormalize
                inv_norm = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225])
                img_vis = inv_norm(img[0]).permute(1,2,0).cpu().numpy()
                img_vis = np.clip(img_vis, 0, 1)
                gt_vis = mask[0].squeeze().cpu().numpy()
                pred_vis = torch.sigmoid(pred[0]).squeeze().cpu().numpy()
                visuals.append((img_vis, gt_vis, pred_vis))

    metrics = evaluator.get_results()
   
    # OUTPUT FORMATTED FOR PAPER
    print("\n" + "="*50)
    print("🏆 COMPARATIVE RESULTS (95% CI)")
    print("="*50)
    print(f"Algorithm    | MAE (Lower is better) | Adp F-Measure (Higher is better)")
    print(f"-------------|-----------------------|------------------------------")
    print(f"F3Net (Ref)  | 0.035                 | 0.840")
    print(f"Model 2 (PSAR)  | {metrics['MAE'][0]:.4f} ± {(metrics['MAE'][2]-metrics['MAE'][1])/2:.4f} | {metrics['F1'][0]:.4f} ± {(metrics['F1'][2]-metrics['F1'][1])/2:.4f}")
    print("="*50)
    print(f"Full MAE CI: [{metrics['MAE'][1]:.4f}, {metrics['MAE'][2]:.4f}]")
    print(f"Full F1  CI: [{metrics['F1'][1]:.4f}, {metrics['F1'][2]:.4f}]")

    # Plot
    os.makedirs("output/visuals", exist_ok=True)
    plt.figure(figsize=(12, 10))
    for i, (img, gt, pred) in enumerate(visuals):
        plt.subplot(4, 3, i*3+1); plt.imshow(img); plt.axis('off');
        if i==0: plt.title("Input")
        plt.subplot(4, 3, i*3+2); plt.imshow(gt, cmap='gray'); plt.axis('off');
        if i==0: plt.title("Ground Truth")
        plt.subplot(4, 3, i*3+3); plt.imshow(pred, cmap='gray'); plt.axis('off');
        if i==0: plt.title("Model 2 (PSAR)")
    plt.tight_layout()
    plt.savefig("output/visuals/final_paper_results.png")
    print("Saved visual results to output/visuals/final_paper_results.png")
    plt.show()

if __name__ == "__main__":
    # If a trained model exists, just load and evaluate, else train.
    if os.path.exists("best_psar_model.pth"):
        print("Found existing trained model. Loading for evaluation...")
        model = SimpleRefineNet().to(CONFIG['DEVICE'])
        model.load_state_dict(torch.load("best_psar_model.pth", map_location=CONFIG['DEVICE']))
        evaluate_standardized(model)
    else:
        model = train_progressive()
        evaluate_standardized(model)
