import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore")

# Highly recommended for standard S-measure and E-measure
try:
    from py_sod_metrics import Smeasure, Emeasure, Fmeasure, MAE
    HAS_PY_SOD = True
except ImportError:
    HAS_PY_SOD = False
    print("⚠️  py_sod_metrics not found. Run: !pip install pysodmetrics")

class PaperEvaluator:
    def __init__(self, gt_dir, pred_dirs, model_names):
        """
        gt_dir: path to DUTS-TE/mask
        pred_dirs: list of paths to model predictions (e.g., ['preds_m1', 'preds_m2', 'preds_m3'])
        model_names: list of names (e.g., ['Model 1 (Hybrid)', 'Model 2 (PSAR)', 'Model 3 (Hybrid+PSAR)'])
        """
        self.gt_dir = gt_dir
        self.pred_dirs = pred_dirs
        self.model_names = model_names
        self.img_names = [f for f in os.listdir(gt_dir) if f.endswith('.png') or f.endswith('.jpg')]
        
        # Storage for metrics
        self.results = {name: {'mae': [], 'f1': [], 'tpr_curve': np.zeros(255), 'fpr_curve': np.zeros(255)} for name in model_names}
        self.beta2 = 0.3

    def compute_metrics(self):
        print(f"Evaluating {len(self.img_names)} images across {len(self.model_names)} models...")
        
        # Initialize py_sod_metrics if available
        if HAS_PY_SOD:
            sod_metrics = {name: {'S': Smeasure(), 'E': Emeasure(), 'F': Fmeasure(), 'M': MAE()} for name in self.model_names}

        for img_name in tqdm(self.img_names, desc="Processing Images"):
            gt_path = os.path.join(self.gt_dir, img_name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is None: continue
            
            gt_norm = (gt / 255.0).astype(np.float32)
            gt_bin = gt_norm > 0.5
            
            for d_idx, pred_dir in enumerate(self.pred_dirs):
                m_name = self.model_names[d_idx]
                pred_path = os.path.join(pred_dir, img_name)
                
                # If model prediction missing, skip (for partial evaluations)
                if not os.path.exists(pred_path): 
                    # Try png instead of jpg or vice versa
                    alt_ext = '.png' if img_name.endswith('.jpg') else '.jpg'
                    pred_path = os.path.join(pred_dir, img_name.split('.')[0] + alt_ext)
                    if not os.path.exists(pred_path):
                        continue
                        
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                pred_norm = (pred / 255.0).astype(np.float32)

                # 1. Custom MAE & F1 (Per Image for Bootstrapping CI)
                mae = np.mean(np.abs(pred_norm - gt_norm))
                self.results[m_name]['mae'].append(mae)

                thresh = 2 * pred_norm.mean()
                thresh = min(thresh, 1.0)
                pred_bin = pred_norm > thresh
                
                tp = (pred_bin & gt_bin).sum()
                prec = tp / (pred_bin.sum() + 1e-8)
                rec = tp / (gt_bin.sum() + 1e-8)
                f_score = (1 + self.beta2) * prec * rec / (self.beta2 * prec + rec + 1e-8)
                self.results[m_name]['f1'].append(f_score)

                # 2. ROC Curve Data (TPR and FPR across 255 thresholds)
                for t in range(255):
                    t_norm = t / 255.0
                    p_b = pred_norm >= t_norm
                    
                    TP = (p_b & gt_bin).sum()
                    FP = (p_b & ~gt_bin).sum()
                    FN = (~p_b & gt_bin).sum()
                    TN = (~p_b & ~gt_bin).sum()
                    
                    self.results[m_name]['tpr_curve'][t] += TP / (TP + FN + 1e-8)
                    self.results[m_name]['fpr_curve'][t] += FP / (FP + TN + 1e-8)

                # 3. Official PySODMetrics (S, E, maxF)
                if HAS_PY_SOD:
                    sod_metrics[m_name]['S'].step(pred, gt)
                    sod_metrics[m_name]['E'].step(pred, gt)
                    sod_metrics[m_name]['F'].step(pred, gt)
                    sod_metrics[m_name]['M'].step(pred, gt)

        # Normalize ROC curves
        total_imgs = len(self.img_names)
        for name in self.model_names:
            self.results[name]['tpr_curve'] /= total_imgs
            self.results[name]['fpr_curve'] /= total_imgs

        # Generate Final Dictionary
        final_stats = {}
        for name in self.model_names:
            mae_arr = np.array(self.results[name]['mae'])
            f1_arr = np.array(self.results[name]['f1'])
            
            if len(mae_arr) == 0: continue

            # Bootstrapping for 95% Confidence Interval
            def get_ci(data):
                means = [np.mean(resample(data)) for _ in range(1000)]
                return np.mean(data), np.percentile(means, 2.5), np.percentile(means, 97.5)

            m_mu, m_lo, m_hi = get_ci(mae_arr)
            f_mu, f_lo, f_hi = get_ci(f1_arr)
            
            stats = {
                'MAE_CI': (m_mu, m_lo, m_hi),
                'F1_CI': (f_mu, f_lo, f_hi)
            }

            if HAS_PY_SOD:
                stats['S_measure'] = sod_metrics[name]['S'].get_results()['sm']
                stats['E_measure'] = sod_metrics[name]['E'].get_results()['em']['curve'].max()
                stats['MaxF'] = sod_metrics[name]['F'].get_results()['fm']['curve'].max()
            
            final_stats[name] = stats

        return final_stats

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        colors = ['#FF4B4B', '#4B4BFF', '#00C853']
        styles = ['--', '-.', '-']
        
        for idx, name in enumerate(self.model_names):
            if np.sum(self.results[name]['tpr_curve']) == 0: continue
            
            fpr = self.results[name]['fpr_curve']
            tpr = self.results[name]['tpr_curve']
            
            plt.plot(fpr, tpr, label=name, color=colors[idx%len(colors)], 
                     linestyle=styles[idx%len(styles)], linewidth=2.5)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=14)
        plt.ylabel('True Positive Rate (TPR)', fontsize=14)
        plt.title('ROC Curves: Influence of Loss Functions', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/ROC_Comparison_Curves.png', dpi=300)
        print("✅ Saved ROC curves to output/ROC_Comparison_Curves.png")
        plt.show()

if __name__ == "__main__":
    print("="*60)
    print(" SOD PAPER EVALUATION SUITE ")
    print("="*60)
    
    # Define directories where prediction images are saved. 
    # (You need to run test() in each script and save predictions to these folders)
    GT_DIR = 'data/DUTS-TE/mask'
    PRED_DIRS = ['out/preds_model1', 'out/preds_model2', 'out/preds_model3']
    MODELS = ['Model 1 (Hybrid Loss)', 'Model 2 (PSAR)', 'Model 3 (Hybrid+PSAR)']
    
    # Check if directories exist
    valid_dirs = []
    valid_models = []
    for d, m in zip(PRED_DIRS, MODELS):
        if os.path.exists(d) and len(os.listdir(d)) > 0:
            valid_dirs.append(d)
            valid_models.append(m)
        else:
            print(f"Skipping {m} - Directory '{d}' not found or empty.")
            
    if len(valid_dirs) == 0:
        print("Error: No prediction directories found. Please run inference on your models first and save masks to the folders listed in the script.")
        exit()
        
    evaluator = PaperEvaluator(GT_DIR, valid_dirs, valid_models)
    results = evaluator.compute_metrics()
    
    print("\n" + "="*80)
    print(" FINAL RESEARCH PAPER METRICS ")
    print("="*80)
    for name, stats in results.items():
        print(f"\n[{name}]")
        print(f"  MAE (95% CI):   {stats['MAE_CI'][0]:.4f}  [ {stats['MAE_CI'][1]:.4f} , {stats['MAE_CI'][2]:.4f} ]")
        print(f"  AdpF (95% CI):  {stats['F1_CI'][0]:.4f}  [ {stats['F1_CI'][1]:.4f} , {stats['F1_CI'][2]:.4f} ]")
        if 'S_measure' in stats:
            print(f"  S-Measure (Sα): {stats['S_measure']:.4f}")
            print(f"  E-Measure (Eφ): {stats['E_measure']:.4f}")
            print(f"  Max F-Measure:  {stats['MaxF']:.4f}")
            
    print("="*80)
    evaluator.plot_roc_curves()
