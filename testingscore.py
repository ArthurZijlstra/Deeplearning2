import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from medpy.metric.binary import dc, hd
from scipy.stats import pearsonr
import warnings

# We negeren kleine waarschuwingen voor een schone output
warnings.filterwarnings('ignore')

# --- CONFIGURATIE ---
test_dir = "./originele_data/testing/testing/"
pred_dir = "./predictions"
classes = {1: "Rechter Ventrikel (RV)", 2: "Myocardium (MYO)", 3: "Linker Ventrikel (LV)"}

def parse_info_cfg(cfg_path):
    info = {}
    with open(cfg_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':')
                info[key.strip()] = val.strip()
    return int(info.get('ED', 1)), int(info.get('ES', 1))

def compute_class_metrics(gt_arr, pred_arr, class_id, voxel_size):
    gt_seg = (gt_arr == class_id).astype(int)
    pred_seg = (pred_arr == class_id).astype(int)
    
    if gt_seg.sum() == 0 or pred_seg.sum() == 0:
        return 0.0, np.nan, 0.0, 0.0
    
    dice = dc(gt_seg, pred_seg)
    # Hausdorff afstand in mm
    try:
        hausdorff = hd(pred_seg, gt_seg, voxelspacing=voxel_size)
    except:
        hausdorff = np.nan
        
    vol_gt = gt_seg.sum() * np.prod(voxel_size) / 1000.0
    vol_pred = pred_seg.sum() * np.prod(voxel_size) / 1000.0
    
    return dice, hausdorff, vol_gt, vol_pred

# Verzamelbak voor alle resultaten
all_results = []

print("🔍 Analyseert 50 patiënten voor RV, MYO en LV...")

patient_folders = sorted(glob.glob(os.path.join(test_dir, "patient*")))

for folder in patient_folders:
    pid = os.path.basename(folder)
    cfg_path = os.path.join(folder, "Info.cfg")
    if not os.path.exists(cfg_path): continue
        
    ed_f, es_f = parse_info_cfg(cfg_path)
    
    paths = {
        'gt_ed': os.path.join(folder, f"{pid}_frame{ed_f:02d}_gt.nii.gz"),
        'gt_es': os.path.join(folder, f"{pid}_frame{es_f:02d}_gt.nii.gz"),
        'pr_ed': os.path.join(pred_dir, f"{pid}_frame{ed_f:02d}_gt.nii.gz"),
        'pr_es': os.path.join(pred_dir, f"{pid}_frame{es_f:02d}_gt.nii.gz")
    }

    if not all(os.path.exists(p) for p in paths.values()): continue
        
    # Laden van data
    nii_ed = nib.load(paths['gt_ed'])
    gt_ed = nii_ed.get_fdata()
    vox_size = nii_ed.header.get_zooms()
    pr_ed = nib.load(paths['pr_ed']).get_fdata()
    
    gt_es = nib.load(paths['gt_es']).get_fdata()
    pr_es = nib.load(paths['pr_es']).get_fdata()
    
    # Bereken metrics voor ELKE klasse
    for cid, cname in classes.items():
        d_ed, h_ed, v_gt_ed, v_pr_ed = compute_class_metrics(gt_ed, pr_ed, cid, vox_size)
        d_es, h_es, v_gt_es, v_pr_es = compute_class_metrics(gt_es, pr_es, cid, vox_size)
        
        # EF berekening (Alleen relevant voor ventrikels 1 en 3, maar we doen het voor de consistentie)
        ef_gt = ((v_gt_ed - v_gt_es) / v_gt_ed * 100) if v_gt_ed > 0 else 0
        ef_pr = ((v_pr_ed - v_pr_es) / v_pr_ed * 100) if v_pr_ed > 0 else 0
        
        all_results.append({
            'Patient': pid, 'ClassID': cid, 'ClassName': cname,
            'Dice_ED': d_ed, 'Dice_ES': d_es,
            'HD_ED': h_ed, 'HD_ES': h_es,
            'Vol_GT_ED': v_gt_ed, 'Vol_PR_ED': v_pr_ed,
            'EF_GT': ef_gt, 'EF_PR': ef_pr
        })

df_all = pd.DataFrame(all_results)

# --- UITDRAAI PER KLASSE ---
for cid in [3, 1, 2]: # We beginnen met LV, dan RV, dan MYO
    df_c = df_all[df_all['ClassID'] == cid]
    name = classes[cid]
    
    # Verschillen berekenen
    df_c['Vol_Diff'] = df_c['Vol_PR_ED'] - df_c['Vol_GT_ED']
    df_c['EF_Diff'] = df_c['EF_PR'] - df_c['EF_GT']
    
    print(f"\n" + "█"*60)
    print(f" STATISTIEKEN VOOR: {name.upper()}")
    print("█" + "-"*58 + "█")
    
    print(f"{'Mean DICE ED':<25}: {df_c['Dice_ED'].mean():.4f}")
    print(f"{'Mean DICE ES':<25}: {df_c['Dice_ES'].mean():.4f}")
    print(f"{'Mean Hausdorff ED':<25}: {df_c['HD_ED'].dropna().mean():.2f} mm")
    print(f"{'Mean Hausdorff ES':<25}: {df_c['HD_ES'].dropna().mean():.2f} mm")
    
    # Correlaties
    c_ef, _ = pearsonr(df_c['EF_PR'], df_c['EF_GT']) if cid != 2 else (0,0)
    c_vol, _ = pearsonr(df_c['Vol_PR_ED'], df_c['Vol_GT_ED'])
    
    print(f"{'-'*30}")
    if cid != 2: # EF is niet echt een ding voor het Myocardium
        print(f"{'EF correlation':<25}: {c_ef:.4f}")
        print(f"{'EF bias':<25}: {df_c['EF_Diff'].mean():.2f} %")
        print(f"{'EF std':<25}: {df_c['EF_Diff'].std():.2f} %")
        print(f"{'-'*30}")
        
    print(f"{'Volume ED correlation':<25}: {c_vol:.4f}")
    print(f"{'Volume ED bias':<25}: {df_c['Vol_Diff'].mean():.2f} ml")
    print(f"{'Volume ED std':<25}: {df_c['Vol_Diff'].std():.2f} ml")
