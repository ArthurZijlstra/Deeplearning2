import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
from tqdm import tqdm


import random
import math


# Zorg dat Python de MedSAM2 map kent
sys.path.append(os.path.abspath('./MedSAM2/MedSAM2'))

from sam2.build_sam import build_sam2
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# ==========================================
# 0. ACDC METRICS BEREKENEN (Dice & HD95)
# ==========================================
def compute_3d_hd95(pred_vol, target_vol, num_classes=4, voxel_spacing=(1.0, 1.0, 1.0)):
    """Razendsnelle HD95 die alleen de randen (contours) van de 3D volumes vergelijkt."""
    hd95_scores = {}
    
    for c in range(1, num_classes):
        p = (pred_vol == c)
        t = (target_vol == c)
        
        if t.sum() == 0 or p.sum() == 0:
            hd95_scores[c] = np.nan
            continue
            
        # Extraheer alleen de contouren/randen voor gigantische snelheidswinst!
        p_edges = p ^ binary_erosion(p)
        t_edges = t ^ binary_erosion(t)
        
        coords_pred = np.argwhere(p_edges) * voxel_spacing
        coords_target = np.argwhere(t_edges) * voxel_spacing
        
        if len(coords_pred) == 0: coords_pred = np.argwhere(p) * voxel_spacing
        if len(coords_target) == 0: coords_target = np.argwhere(t) * voxel_spacing
        
        tree_pred = cKDTree(coords_pred)
        tree_target = cKDTree(coords_target)
        
        dist1, _ = tree_pred.query(coords_target)
        dist2, _ = tree_target.query(coords_pred)
        
        hd95_scores[c] = max(np.percentile(dist1, 95), np.percentile(dist2, 95))
        
    return hd95_scores

def compute_3d_dice(pred_vol, target_vol, num_classes=4):
    """Berekent de Dice score over het gehele 3D volume."""
    dice_scores = {}
    
    for c in range(1, num_classes):
        p = (pred_vol == c)
        t = (target_vol == c)
        
        # Als de klasse niet voorkomt in de ground truth én niet in de predictie
        if t.sum() == 0 and p.sum() == 0:
            continue
        # Als de klasse wél zou moeten bestaan, maar compleet is gemist (of andersom)
        elif t.sum() == 0 or p.sum() == 0:
            dice_scores[c] = 0.0
        else:
            intersection = np.logical_and(p, t).sum()
            union = p.sum() + t.sum()
            dice_scores[c] = (2. * intersection) / union
            
    return dice_scores

from scipy.ndimage import binary_erosion
def evaluate_batch_metrics(preds, targets, num_classes=4):
    """Geeft de Dice en HD95 per klasse (RV=1, Myo=2, LV=3) terug"""
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    batch_size = preds.shape[0]
    
    # Woordenboeken om de scores per klasse op te slaan
    dice_scores = {1: [], 2: [], 3: []}
    hd95_scores = {1: [], 2: [], 3: []}
    
    for b in range(batch_size):
        for c in range(1, num_classes):
            p = (preds[b] == c)
            t = (targets[b] == c)
            
            # Alleen berekenen als dit orgaan überhaupt op deze MRI-slice staat!
            if t.sum() > 0:
                intersection = np.logical_and(p, t).sum()
                union = p.sum() + t.sum()
                dice = (2. * intersection) / union if union > 0 else 0.0
                
                dice_scores[c].append(dice)
                hd95_scores[c].append(compute_hd95(p, t))
                
    return dice_scores, hd95_scores

# ==========================================
# 0.5 COMBINED LOSS FUNCTIE
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dice_loss = 0.0
        classes_counted = 0
        
        for c in range(1, self.num_classes): # Sla 0 (achtergrond) over
            prob_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]
            
            intersection = (prob_c * target_c).sum(dim=(1, 2))
            union = prob_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            
            dice_c = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_loss += (1.0 - dice_c).mean()
            classes_counted += 1
            
        return dice_loss / classes_counted

class CombinedLoss(nn.Module):
    # LET OP: Hier staat nu (self, device) in plaats van alleen (self)
    def __init__(self, device):
        super().__init__()
        
        # De Class Imbalance gewichten (Achtergrond, RV, Myo, LV)
        weights = torch.tensor([0.1, 1.0, 1.5, 1.0], dtype=torch.float32).to(device)
        
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss(num_classes=4)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)

# ==========================================
# 1. DATASET INLADEN (Met Native PyTorch Augmentatie!)
# ==========================================
class ACDCDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_training=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = (512, 512)  # Dit is wat SAM absoluut eist
        self.crop_size = (256, 256)    # Dit is het kleine stukje dat we uit de MRI knippen
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx])   
        mask = np.load(self.mask_paths[idx]) 
        patient_id = os.path.basename(self.image_paths[idx]).split('_')[0]

        # NIEUW: Normaliseer de pixelwaardes naar 0.0 - 1.0!
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        

        img = np.stack((img, img, img), axis=0)
        mask = np.expand_dims(mask, axis=0)

        img_t = torch.tensor(img, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        
        # 1. Native PyTorch Center Cropping (Nu 100% shape-proof!)
        # Pak altijd de LAATSTE twee getallen uit de shape (Hoogte en Breedte)
        h, w = img_t.shape[-2:] 
        ch, cw = self.crop_size

        # Stap A: Padding indien het origineel te klein is
        if h < ch or w < cw:
            pad_y = max(0, ch - h)
            pad_x = max(0, cw - w)
            padding = (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2)
            img_t = F.pad(img_t, padding, mode='constant', value=0.0)
            mask_t = F.pad(mask_t, padding, mode='constant', value=0.0)
            
            # Update H en W weer netjes
            h, w = img_t.shape[-2:]

        # Stap B: Knippen (Gooi de nutteloze randen en longen weg)
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        
        # Gebruik ... (ellipsis) om aan te geven: wat er ook vóór zit, knip alleen in de laatste 2 dimensies!
        img_t = img_t[..., y1:y1+ch, x1:x1+cw]
        mask_t = mask_t[..., y1:y1+ch, x1:x1+cw]
        
        
        # Stap C: Digitaal Inzoomen! Blaas het hart op naar 512x512 voor SAM
        img_t = F.interpolate(img_t.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        mask_t = F.interpolate(mask_t.unsqueeze(0), size=self.target_size, mode="nearest").squeeze(0)

        # 2. Data Augmentatie (Puur Wiskundig via PyTorch)
        if self.is_training:
            # A. Willekeurig horizontaal spiegelen (50% kans)
            if random.random() > 0.5:
                img_t = torch.flip(img_t, dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])

            # B. Willekeurig draaien en zoomen
            angle = random.uniform(-15, 15)
            scale = random.uniform(0.9, 1.1)

            # Reken graden om naar radialen
            angle_rad = angle * math.pi / 180.0
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Bouw de 2D "Affine" transformatiematrix
            theta = torch.tensor([
                [cos_a / scale, sin_a / scale, 0.0],
                [-sin_a / scale, cos_a / scale, 0.0]
            ], dtype=torch.float32).unsqueeze(0) # Shape: (1, 2, 3)

            # Creëer een vervormd "raster" op basis van de matrix
            grid = F.affine_grid(theta, size=(1, 3, self.target_size[0], self.target_size[1]), align_corners=False)

            # Trek de pixels via het raster naar hun nieuwe plek!
            # Beeld mag vloeiend (bilinear), masker moet keihard blijven (nearest)
            img_t = F.grid_sample(img_t.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(0)
            mask_t = F.grid_sample(mask_t.unsqueeze(0), grid, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0)

        return img_t, mask_t.long(), patient_id
# ==========================================
# 2. MODEL INSTELLEN & FREEZEN
# ==========================================
def setup_medsam2(checkpoint_path):
    GlobalHydra.instance().clear()
    config_dir = os.path.abspath("./") 
    
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # LET OP: We geven nu weer de NAAM door, en niet de 'cfg'.
        model = build_sam2("sam2.1_hiera_t512.yaml", checkpoint_path)
    
    for param in model.image_encoder.parameters():
        param.requires_grad = False
        
    for param in model.sam_mask_decoder.parameters():
        param.requires_grad = True
    for param in model.sam_prompt_encoder.parameters():
        param.requires_grad = True

    try:
        model.use_high_res_features_in_sam = False
        model.sam_mask_decoder.use_high_res_features = False
    except: pass
        
    return model
# ==========================================
# 3. TRAININGS LOOP MET K-FOLD CROSS VALIDATION
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Start training op: {device}")

    img_dir = "./processed_data/processed_data/images"
    mask_dir = "./processed_data/processed_data/masks"

    all_images = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
    all_masks = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))

    if len(all_images) == 0:
        print("Fout: Geen plaatjes gevonden!")
        return

    # --- PATIENT-LEVEL GROEPEREN ---
    # We extraheren het patiëntnummer uit de bestandsnaam (bijv. uit "patient001_frame01.npy")
    groups = [os.path.basename(p).split('_')[0] for p in all_images]
    
    gkf = GroupKFold(n_splits=5)
    sam_checkpoint = "./MedSAM2/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt"

    # We slaan de resultaten van alle 5 folds op voor het eindoordeel
    fold_results = []
    epochs_per_fold = 100 # Verander naar 20 als je héél veel tijd hebt!
    
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(all_images, all_masks, groups)):
        print(f"\n{'='*40}")
        print(f"🚀 START FOLD {fold + 1}/5")
        print(f"{'='*40}")

        train_imgs = [all_images[i] for i in train_idx]
        train_msks = [all_masks[i] for i in train_idx]
        val_imgs = [all_images[i] for i in val_idx]
        val_msks = [all_masks[i] for i in val_idx]

        # Geef True mee aan de training, en False aan de validatie!
        # Voeg num_workers=4 en pin_memory=True toe!
        train_loader = DataLoader(
            ACDCDataset(train_imgs, train_msks, is_training=True), 
            batch_size=4, 
            shuffle=True, 
            num_workers=4,        # 4 processen laden de data alvast in
            pin_memory=True,      # Zet data klaar naast de GPU
            drop_last=True        # Voorkomt gedoe met incomplete laatste batches
        )
        val_loader = DataLoader(
            ACDCDataset(val_imgs, val_msks, is_training=False), 
            batch_size=4, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )    
        

        # Reset model en optimizer compleet voor elke nieuwe fold
        model = setup_medsam2(sam_checkpoint).to(device)
        proj_layer = nn.Sequential(
            # Stap 1: Kijk naar buur-pixels (kernel_size=3) en maak 16 "denk-kanalen" aan
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            # Stap 2: Voeg niet-lineaire denkkracht toe
            nn.GELU(),
            # Stap 3: Normaliseer de waardes zodat training stabiel blijft
            nn.BatchNorm2d(16),
            # Stap 4: Vertaal de 16 denk-kanalen terug naar jouw 4 definitieve ACDC-klasses
            nn.Conv2d(16, 4, kernel_size=1)
        ).to(device)
        model.train()
        proj_layer.train()

        optimizer = optim.AdamW(
            [
                {"params": model.sam_mask_decoder.parameters()},
                {"params": model.sam_prompt_encoder.parameters()},
                {"params": proj_layer.parameters()}
            ], 
            lr=1e-3, weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_fold)
        
        criterion = CombinedLoss(device)
        
        # Nieuw: De AMP Scaler die de 16-bit en 32-bit getallen in balans houdt
        best_dice = 0.0
        # TRAINING LOOP
        for epoch in range(epochs_per_fold):
            epoch_loss = 0.0
            for batch_idx, (images, true_masks, _) in enumerate(tqdm(train_loader, desc=f"Fold {fold+1} - Epoch {epoch+1}")):
                images = images.to(device)
                true_masks = true_masks.squeeze(1).long().to(device) 

                # 1. Maak het geheugen van de vorige stap leeg
                optimizer.zero_grad(set_to_none=True)

               
                # 2. De 16-bit Snelweg (Nu met standaard float16)
                # NIEUW: Gebruik bfloat16
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_embeddings = model.image_encoder(images)
                    
                    sam_point_coords = torch.zeros(images.shape[0], 1, 2, device=device)
                    sam_point_labels = -torch.ones(images.shape[0], 1, dtype=torch.int32, device=device)
                    sparse_emb, dense_emb = model.sam_prompt_encoder(
                        points=(sam_point_coords, sam_point_labels), boxes=None, masks=None
                    )

                    predicted_masks, _, _, _ = model.sam_mask_decoder(
                        image_embeddings=image_embeddings["vision_features"],
                        image_pe=model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb.to(device),
                        dense_prompt_embeddings=dense_emb.to(device),
                        multimask_output=True, repeat_image=False, high_res_features=None,
                    )

                    multi_class_logits = proj_layer(predicted_masks)
                    if multi_class_logits.shape[-2:] != true_masks.shape[-2:]:
                        multi_class_logits = F.interpolate(multi_class_logits, size=true_masks.shape[-2:], mode="bilinear")

                # 3. Weer UIT de snelweg (spring terug naar links!)
                # Zet hem expliciet op .float() (32-bit) zodat de leraar niet in de war raakt
                loss = criterion(multi_class_logits.float(), true_masks)

                # 4. Leren van de fouten (Slechts EEN keer backward!)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
            scheduler.step()
            print(f"Fold {fold+1} - Epoch {epoch+1}/{epochs_per_fold} | Train Loss: {epoch_loss/len(train_loader):.4f}")

        # VALIDATIE LOOP (Het Examen!)
        print("🔍 Beoordelen op onzichtbare patiënten (3D Evaluatie)...")
        model.eval()
        proj_layer.eval()
        
        # Woordenboeken om voorspellingen per patiënt op te sparen
        patient_preds = {}
        patient_targets = {}

        with torch.no_grad():
            # Let op: we vangen nu ook patient_ids op uit de loader!
            for images, true_masks, patient_ids in val_loader:
                images = images.to(device)
                true_masks = true_masks.squeeze(1).long().to(device) 

                # SAM2 Forward Pass (Hetzelfde als je al had)
                image_embeddings = model.image_encoder(images)
                sam_point_coords = torch.zeros(images.shape[0], 1, 2, device=device)
                sam_point_labels = -torch.ones(images.shape[0], 1, dtype=torch.int32, device=device)
                sparse_emb, dense_emb = model.sam_prompt_encoder(
                    points=(sam_point_coords, sam_point_labels), boxes=None, masks=None
                )

                predicted_masks, _, _, _ = model.sam_mask_decoder(
                    image_embeddings=image_embeddings["vision_features"],
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True, repeat_image=False, high_res_features=None,
                )

                multi_class_logits = proj_layer(predicted_masks)
                if multi_class_logits.shape[-2:] != true_masks.shape[-2:]:
                    multi_class_logits = F.interpolate(multi_class_logits, size=true_masks.shape[-2:], mode="bilinear")

                # Omzetten naar harde klasses (0, 1, 2 of 3) en naar CPU verplaatsen
                batch_preds = torch.argmax(multi_class_logits, dim=1).cpu().numpy()
                batch_targets = true_masks.cpu().numpy()

                # Voorspellingen groeperen per patiënt
                for b in range(images.shape[0]):
                    pid = patient_ids[b]
                    if pid not in patient_preds:
                        patient_preds[pid] = []
                        patient_targets[pid] = []
                    
                    patient_preds[pid].append(batch_preds[b])
                    patient_targets[pid].append(batch_targets[b])

        # === 3D BEREKENING NA DE LOOP ===
        val_dice = {1: [], 2: [], 3: []}
        val_hd95 = {1: [], 2: [], 3: []}

        # Loop door alle complete patiënt-volumes heen
        for pid in patient_preds.keys():
            # Stapel de 2D slices op tot een 3D array: (Z, Y, X)
            pred_vol = np.stack(patient_preds[pid], axis=0)
            target_vol = np.stack(patient_targets[pid], axis=0)
            
            # Bereken de metrics over het hele volume
            p_dice = compute_3d_dice(pred_vol, target_vol)
            p_hd95 = compute_3d_hd95(pred_vol, target_vol) # Voeg hier evt voxel_spacing toe als je die hebt!
            
            for c in range(1, 4):
                if c in p_dice: 
                    val_dice[c].append(p_dice[c])
                if c in p_hd95 and not np.isnan(p_hd95[c]): 
                    val_hd95[c].append(p_hd95[c])
                    
        # Gemiddelden van deze fold berekenen
        avg_d = {c: np.mean(val_dice[c]) for c in range(1, 4) if len(val_dice[c]) > 0}
        avg_h = {c: np.mean(val_hd95[c]) for c in range(1, 4) if len(val_hd95[c]) > 0}
        
        print(f"✅ SCORE FOLD {fold+1}:")
        print(f" - Rechter Ventrikel (Klasse 1): Dice {avg_d.get(1,0):.4f} | HD95 {avg_h.get(1,0):.2f} units")
        print(f" - Myocardium        (Klasse 2): Dice {avg_d.get(2,0):.4f} | HD95 {avg_h.get(2,0):.2f} units")
        print(f" - Linker Ventrikel  (Klasse 3): Dice {avg_d.get(3,0):.4f} | HD95 {avg_h.get(3,0):.2f} units")
        
        # Sla op voor het eindoordeel
        fold_results.append((avg_d, avg_h))
        print("🔍 Beoordelen op onzichtbare patiënten (Test Set)...")
        model.eval()
        proj_layer.eval()
        
        val_dice = {1: [], 2: [], 3: []}
        val_hd95 = {1: [], 2: [], 3: []}

        # Aan het einde van je validatie loop in een epoch:
        gemiddelde_dice_epoch = np.mean([avg_d.get(c, 0) for c in range(1, 4)])
        if gemiddelde_dice_epoch > best_dice:
            best_dice = gemiddelde_dice_epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'proj_layer_state_dict': proj_layer.state_dict()
            }, f"best_medsam2_fold100epoch_{fold}.pth")
            print(f"🔥 Nieuw beste model opgeslagen met Dice: {best_dice:.4f}")
    # ==========================================
    # DE VUILNISMAN (Nieuw! Voorkomt OutOfMemory)
    # ==========================================
    # Verwijder de grote variabelen uit het werkgeheugen
        del model
        del proj_layer
        del optimizer
        del train_loader
        del val_loader
        
        # Dwing de videokaart (CUDA) om de prullenbak écht te legen
        torch.cuda.empty_cache()
        # ==========================================

    # ================================
    # HET DEFINITIEVE EINDOORDEEL
    # ================================
    print(f"\n{'='*40}")
    print("🏆 EINDOORDEEL (5-Fold Cross Validation)")
    print(f"{'='*40}")
    
    for c, name in zip([1,2,3], ["Rechter Ventrikel", "Myocardium", "Linker Ventrikel"]):
        final_dice = np.mean([f[0].get(c, 0) for f in fold_results])
        final_hd95 = np.mean([f[1].get(c, 0) for f in fold_results])
        print(f"{name:18s} -> Gemiddelde Dice: {final_dice:.4f} | Gemiddelde HD95: {final_hd95:.2f} pixels")

if __name__ == "__main__":
    train_model()
