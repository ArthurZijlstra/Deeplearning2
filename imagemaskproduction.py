import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

warnings.filterwarnings('ignore')

# --- MAPPEN CONFIGURATIE ---
test_dir = "./originele_data/testing/testing/"
pred_dir = "./predictions"

# We pakken de eerste 3 voorspellingen uit de map
pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))[:3]

# Vaste kleuren voor de organen: 0=Zwart (Achtergrond), 1=Blauw (RV), 2=Groen (MYO), 3=Rood (LV)
cmap = ListedColormap(['none', 'blue', 'lime', 'red'])

# Maak een figuur aan (Rijen = Patiënten, Kolommen = Origineel / Arts / Model)
fig, axes = plt.subplots(len(pred_files), 3, figsize=(15, 5 * len(pred_files)))

for i, pred_path in enumerate(pred_files):
    filename = os.path.basename(pred_path)  # Bijv: patient101_frame01_gt.nii.gz
    patient_id = filename.split('_')[0]
    
    # Bepaal de paden naar de originele MRI en de Ground Truth (Arts)
    gt_path = os.path.join(test_dir, patient_id, filename)
    orig_img_name = filename.replace("_gt.nii.gz", ".nii.gz")
    orig_path = os.path.join(test_dir, patient_id, orig_img_name)
    
    # Laad de 3D volumes
    orig_vol = nib.load(orig_path).get_fdata()
    gt_vol = nib.load(gt_path).get_fdata()
    pred_vol = nib.load(pred_path).get_fdata()
    
    # SLIM TRUCJE: Zoek de Z-slice waar het hart het grootst is in de Ground Truth
    pixels_per_slice = (gt_vol > 0).sum(axis=(0, 1))
    best_slice_idx = np.argmax(pixels_per_slice)
    
    # Isoleer dat specifieke 2D plakje
    orig_slice = orig_vol[:, :, best_slice_idx]
    gt_slice = gt_vol[:, :, best_slice_idx]
    pred_slice = pred_vol[:, :, best_slice_idx]
    
    # -- KOLOM 1: Originele MRI (Kaal) --
    axes[i, 0].imshow(orig_slice, cmap='gray')
    axes[i, 0].set_title(f"{patient_id} - MRI (Slice {best_slice_idx})", fontsize=14)
    axes[i, 0].axis('off')
    
    # -- KOLOM 2: Ground Truth (Arts) --
    axes[i, 1].imshow(orig_slice, cmap='gray')
    # Maak alle '0' (achtergrond) pixels volledig doorzichtig
    gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
    axes[i, 1].imshow(gt_masked, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
    axes[i, 1].set_title("Arts (Ground Truth)", fontsize=14)
    axes[i, 1].axis('off')
    
    # -- KOLOM 3: Jouw Model (Predictie) --
    axes[i, 2].imshow(orig_slice, cmap='gray')
    pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
    axes[i, 2].imshow(pred_masked, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
    axes[i, 2].set_title("Jouw Model", fontsize=14)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()
