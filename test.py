"""
Full end-to-end PyTorch implementation for DeepFashion-MultiModal
- Dataloader that reads images, parsing masks, keypoints, and label files
- ResNet50 backbone + keypoint MLP + multi-task classification heads
- Training loop, evaluation, and inference
- At inference: simple 3-region color detection using computer vision
  (upper_body, lower_body, shoes) based on vertical bands.

Usage examples:
  python deepfashion_multimodal_training.py --mode train --data_root /path/to/DeepFashion-MultiModal
  python deepfashion_multimodal_training.py --mode infer --data_root /path/to/DeepFashion-MultiModal --image example.jpg

Notes:
- You must adapt file names/paths if the dataset layout differs.
- This is a single-file baseline. For production: split into modules.

Requirements:
  python 3.8+
  pip install torch torchvision tqdm pillow numpy

"""

import os
import json
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# -----------------------------
# Helper label maps (from dataset README)
# -----------------------------
SHAPE_LABELS = {
    'sleeve_length': [
        'sleeveless','short-sleeve','medium-sleeve','long-sleeve','not-long-sleeve','NA'
    ], # 6
    'lower_length': ['three-point','medium-short','three-quarter','long','NA'], # 5
    'socks': ['no','socks','leggings','NA'], #4
    'hat': ['no','yes','NA'], #3
    'glasses': ['no','eyeglasses','sunglasses','have-glasses','NA'], #5
    'neckwear': ['no','yes','NA'], #3
    'wristwear': ['no','yes','NA'], #3
    'ring': ['no','yes','NA'], #3
    'waist_accessory': ['no','belt','have-clothing','hidden','NA'], #5
    'neckline': ['V-shape','square','round','standing','lapel','suspenders','NA'], #7
    'outer_cardigan': ['yes','no','NA'], #3
    'upper_cover_navel': ['no','yes','NA'] #3
}

FABRIC_LABELS = ['denim','cotton','leather','furry','knitted','chiffon','other','NA']
COLOR_LABELS = ['floral','graphic','striped','pure-color','lattice','other','color-block','NA']

# For convenience, build sizes dict
SHAPE_SIZES = {k: len(v) for k,v in SHAPE_LABELS.items()}

# -----------------------------
# Dataset
# -----------------------------
class DeepFashionMM(Dataset):
    """
    Expects directory structure pointed by data_root with subfolders:
      images/ or image/ (JPG/PNG)
      parsing/ (PNG) optional (NOT used for color in this version)
      keypoints_loc.txt
      keypoints_vis.txt
      shape_label.txt
      fabric_label.txt
      color_label.txt
    File formats are as described in dataset README.
    """
    def __init__(self, data_root: str, split: str = 'all', transforms_img=None, use_parsing: bool = True, use_keypoints: bool = True):
        super().__init__()
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, 'image') if os.path.isdir(os.path.join(data_root, 'image')) else os.path.join(data_root, 'images')
        self.parsing_dir = os.path.join(data_root, 'parsing')
        self.transforms_img = transforms_img
        # parsing is still read into dataset, but NOT used for color in this version
        self.use_parsing = use_parsing and os.path.isdir(self.parsing_dir)
        self.use_keypoints = use_keypoints

        # load image list
        self.images = sorted([f for f in os.listdir(self.img_dir)
                              if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')])

        # load keypoints
        self.kp_loc = {}
        self.kp_vis = {}
        kp_loc_path = os.path.join(data_root, 'keypoints_loc.txt')
        kp_vis_path = os.path.join(data_root, 'keypoints_vis.txt')
        if os.path.isfile(kp_loc_path):
            with open(kp_loc_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 1+42:
                        continue
                    name = parts[0]
                    coords = list(map(float, parts[1:1+42]))
                    self.kp_loc[name] = coords
        if os.path.isfile(kp_vis_path):
            with open(kp_vis_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    name = parts[0]
                    vs = list(map(int, parts[1:1+21]))
                    self.kp_vis[name] = vs

        # load labels (shape, fabric, color)
        self.shape_labels = {}
        shape_path = os.path.join(data_root, 'labels', 'shape_label.txt') if os.path.isdir(os.path.join(data_root, 'labels')) else os.path.join(data_root, 'shape_label.txt')
        if os.path.isfile(shape_path):
            with open(shape_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    name = parts[0]
                    vals = list(map(int, parts[1:]))
                    # dataset says 12 shape attributes
                    self.shape_labels[name] = vals

        self.fabric_labels = {}
        fabric_path = os.path.join(data_root, 'labels', 'fabric_label.txt') if os.path.isdir(os.path.join(data_root, 'labels')) else os.path.join(data_root, 'fabric_label.txt')
        if os.path.isfile(fabric_path):
            with open(fabric_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    name = parts[0]
                    vals = list(map(int, parts[1:1+3]))
                    self.fabric_labels[name] = vals

        self.color_labels = {}
        color_path = os.path.join(data_root, 'labels', 'color_label.txt') if os.path.isdir(os.path.join(data_root, 'labels')) else os.path.join(data_root, 'color_label.txt')
        if os.path.isfile(color_path):
            with open(color_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split()
                    name = parts[0]
                    vals = list(map(int, parts[1:1+3]))
                    self.color_labels[name] = vals

        # final list: only images that have at least one label entry
        filtered = []
        for img in self.images:
            base = os.path.splitext(img)[0]
            # Check BOTH full name (img) and base name (base)
            has_shape = (img in self.shape_labels) or (base in self.shape_labels)
            has_fabric = (img in self.fabric_labels) or (base in self.fabric_labels)
            has_color = (img in self.color_labels) or (base in self.color_labels)
            if has_shape or has_fabric or has_color:
                filtered.append(img)
        self.images = filtered
        print(f"Dataset initialized. Found {len(self.images)} valid images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transforms_img:
            img_t = self.transforms_img(img)
        else:
            img_t = transforms.ToTensor()(img)

        # parsing mask (not used for color in this version, but kept for compatibility)
        _, h, w = img_t.shape
        parsing_mask = torch.zeros((h, w), dtype=torch.long)
        if self.use_parsing:
            ppath = os.path.join(self.parsing_dir, base + '.png')
            if os.path.isfile(ppath):
                pm = Image.open(ppath)
                pm = pm.resize((w, h), Image.NEAREST)
                parsing_mask = torch.from_numpy(np.array(pm)).long()

        # keypoints
        kp = None
        if self.kp_loc and (base in self.kp_loc):
            coords = np.array(self.kp_loc[base]).astype(np.float32)
            # normalize by image size
            w0, h0 = img.size
            coords[0::2] /= float(w0)
            coords[1::2] /= float(h0)
            kp = torch.from_numpy(coords).float()
        else:
            kp = torch.full((42,), -1.0, dtype=torch.float32)

        # Shape
        s_val = self.shape_labels.get(img_name, self.shape_labels.get(base, None))
        if s_val is not None:
            shape = torch.tensor(s_val, dtype=torch.long)
        else:
            shape = torch.full((12,), -1, dtype=torch.long)

        # Fabric
        f_val = self.fabric_labels.get(img_name, self.fabric_labels.get(base, None))
        if f_val is not None:
            fabric = torch.tensor(f_val, dtype=torch.long)
        else:
            fabric = torch.full((3,), -1, dtype=torch.long)

        # Color
        c_val = self.color_labels.get(img_name, self.color_labels.get(base, None))
        if c_val is not None:
            color = torch.tensor(c_val, dtype=torch.long)
        else:
            color = torch.full((3,), -1, dtype=torch.long)

        sample = {
            'image': img_t,
            'parsing': parsing_mask,
            'keypoints': kp,
            'shape': shape,
            'fabric': fabric,
            'color': color,
            'name': img_name
        }
        return sample

# -----------------------------
# Model
# -----------------------------
class MultiTaskFashionModel(nn.Module):
    def __init__(self, backbone_name: str = 'resnet50', use_keypoints: bool = True, use_segmentation: bool = False):
        super().__init__()
        assert backbone_name in ['resnet50','resnet18']
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feat_dim = 2048
        else:
            self.backbone = models.resnet18(pretrained=True)
            feat_dim = 512

        # remove final fc
        self.backbone.fc = nn.Identity()

        self.use_keypoints = use_keypoints
        kp_dim = 64
        if use_keypoints:
            self.kp_mlp = nn.Sequential(
                nn.Linear(42, 128),
                nn.ReLU(),
                nn.Linear(128, kp_dim),
                nn.ReLU()
            )
        else:
            kp_dim = 0

        fusion_dim = feat_dim + kp_dim
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # create heads for each shape attribute
        self.shape_heads = nn.ModuleDict()
        for k,v in SHAPE_SIZES.items():
            self.shape_heads[k] = nn.Linear(fusion_dim, v)

        # fabric (3 heads)
        self.fabric_heads = nn.ModuleList([nn.Linear(fusion_dim, len(FABRIC_LABELS)) for _ in range(3)])
        self.color_heads  = nn.ModuleList([nn.Linear(fusion_dim, len(COLOR_LABELS)) for _ in range(3)])

        # seg_decoder is not used in this code path (kept for compatibility)
        self.use_segmentation = use_segmentation
        if use_segmentation:
            self.seg_decoder = nn.Sequential(
                nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 24, kernel_size=1) # 24 parsing classes (0..23)
            )
        else:
            self.seg_decoder = None

    def forward(self, x_image, x_kp=None):
        # backbone expects 3xHxW
        feat = self.backbone(x_image)  # [B, feat_dim]

        if self.use_keypoints and x_kp is not None:
            kp_feat = self.kp_mlp(x_kp)
            fused = torch.cat([feat, kp_feat], dim=1)
        else:
            fused = feat

        fused = self.fusion_norm(fused)

        outputs = {}
        for k,head in self.shape_heads.items():
            outputs[f'shape_{k}'] = head(fused)

        for i,head in enumerate(self.fabric_heads):
            outputs[f'fabric_{i}'] = head(fused)

        for i,head in enumerate(self.color_heads):
            outputs[f'color_{i}'] = head(fused)

        return outputs

# -----------------------------
# Utilities: loss, metrics, human readable
# -----------------------------

def multitask_loss(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], device: torch.device, weights: Optional[Dict[str,float]] = None):
    # outputs: keys like 'shape_sleeve_length' -> [B, C]
    # labels: dict contains tensors
    total = 0.0
    loss = 0.0
    weights = weights or {}

    # shape labels
    for i, (k,size) in enumerate(SHAPE_SIZES.items()):
        out = outputs[f'shape_{k}']
        lab = labels['shape'][:, i].to(device)  # -1 is missing
        mask = lab >= 0
        if mask.sum() == 0:
            continue
        l = F.cross_entropy(out[mask], lab[mask])
        w = weights.get(f'shape_{k}', 1.0)
        loss = loss + w * l
        total += 1

    # fabric
    for i in range(3):
        out = outputs[f'fabric_{i}']
        lab = labels['fabric'][:, i].to(device)
        mask = lab >= 0
        if mask.sum() == 0: continue
        l = F.cross_entropy(out[mask], lab[mask])
        loss = loss + l
        total += 1

    # color
    for i in range(3):
        out = outputs[f'color_{i}']
        lab = labels['color'][:, i].to(device)
        mask = lab >= 0
        if mask.sum() == 0: continue
        l = F.cross_entropy(out[mask], lab[mask])
        loss = loss + l
        total += 1

    if total == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)
    return loss / float(total)


def decode_predictions(outputs: Dict[str, torch.Tensor]) -> Dict[str,str]:
    # outputs contain logits per head for a single sample (batch dim stripped)
    out = {}
    # shape
    for i,(k,labels) in enumerate(SHAPE_LABELS.items()):
        logits = outputs[f'shape_{k}']
        idx = int(logits.argmax().item())
        out[k] = labels[idx] if idx < len(labels) else str(idx)

    # fabric
    for i in range(3):
        logits = outputs[f'fabric_{i}']
        idx = int(logits.argmax().item())
        out[f'fabric_{i}'] = FABRIC_LABELS[idx]

    # color patterns (not pixel color, but category)
    for i in range(3):
        logits = outputs[f'color_{i}']
        idx = int(logits.argmax().item())
        out[f'color_{i}'] = COLOR_LABELS[idx]

    return out

# -----------------------------
# Region-based color detection (3 regions)
# -----------------------------

def extract_region_colors(img_np: np.ndarray) -> Dict[str, Dict[str, Optional[List[int]]]]:
    """
    img_np: H x W x 3 (uint8 or float)
    Returns average RGB for 3 vertical regions:
      - upper_body
      - lower_body
      - shoes
    Uses central 30% width and user-specified height ranges.
    """
    h, w, _ = img_np.shape

    # vertical band in the center 30% width
    x0 = int(w * 0.35)
    x1 = int(w * 0.65)

    regions_y = {
        "upper_body": (int(h * 0.20), int(h * 0.45)),  # chest area
        "lower_body": (int(h * 0.50), int(h * 0.75)),  # hips/thighs
        "shoes":      (int(h * 0.90), int(h * 0.98)),  # feet
    }

    result: Dict[str, Dict[str, Optional[List[int]]]] = {}

    for name, (y0, y1) in regions_y.items():
        # Clamp bounds to be safe
        y0_clamp = max(0, min(h, y0))
        y1_clamp = max(0, min(h, y1))
        if y1_clamp <= y0_clamp:
            result[name] = {"rgb": None, "hex": None}
            continue

        region = img_np[y0_clamp:y1_clamp, x0:x1]

        if region.size == 0:
            result[name] = {"rgb": None, "hex": None}
            continue

        # Average color
        avg = region.mean(axis=(0,1))   # average RGB
        r, g, b = [int(v) for v in avg]
        hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

        result[name] = {
            "rgb": [r, g, b],
            "hex": hex_color
        }

    return result


def visualize_color_regions(image_path: str, save_path: str = "vis_color_regions.png"):
    """
    Save an overlay image showing the three sampled regions:
      - upper_body (red)
      - lower_body (green)
      - shoes (blue)
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x0 = int(w * 0.35)
    x1 = int(w * 0.65)

    regions_y = {
        "upper_body": (int(h * 0.20), int(h * 0.45)),
        "lower_body": (int(h * 0.50), int(h * 0.75)),
        "shoes":      (int(h * 0.90), int(h * 0.98)),
    }

    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    colors = {
        "upper_body": (255,   0,   0, 90),
        "lower_body": (  0, 255,   0, 90),
        "shoes":      (  0,   0, 255, 90),
    }

    for name, (y0, y1) in regions_y.items():
        draw.rectangle([x0, y0, x1, y1], fill=colors[name])

    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    out.save(save_path)
    print(f"[Viz] Saved region overlay to {save_path}")

# -----------------------------
# Training and evaluation
# -----------------------------

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc='train'):
        imgs = batch['image'].to(device)
        kps = batch['keypoints'].to(device)
        labels = {'shape': batch['shape'].to(device),
                  'fabric': batch['fabric'].to(device),
                  'color': batch['color'].to(device)}

        outputs = model(imgs, kps)
        loss = multitask_loss(outputs, labels, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            imgs = batch['image'].to(device)
            kps = batch['keypoints'].to(device)
            labels = {'shape': batch['shape'].to(device),
                      'fabric': batch['fabric'].to(device),
                      'color': batch['color'].to(device)}
            outputs = model(imgs, kps)
            loss = multitask_loss(outputs, labels, device)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# -----------------------------
# Inference util
# -----------------------------

def infer_single(model, image_path: str, device: torch.device, transforms_img):
    """
    Run attribute prediction + region-based color detection
    on a single image.
    """
    model.eval()

    # Load original image for region color detection
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32)  # H x W x 3

    # Preprocess image for model
    x = transforms_img(img).unsqueeze(0).to(device)
    kp = torch.full((1,42), -1.0, dtype=torch.float32).to(device)

    with torch.no_grad():
        outs = model(x, kp)

    outs_squeezed = {k: v[0].cpu() for k,v in outs.items()}
    decoded = decode_predictions(outs_squeezed)

    # region colors from raw image
    region_colors = extract_region_colors(img_np)

    return decoded, region_colors

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=r"C:\Users\Jeffrey M. Ivanson\Documents\MLDEMO\dataset",
                        help='Path to dataset')
    parser.add_argument('--mode', type=str, choices=['train','infer'], default='infer')
    parser.add_argument('--image', type=str, default=r"C:\Users\Jeffrey M. Ivanson\Documents\MLDEMO\54500.jpg", help='Path to input image for inference')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--backbone', type=str, choices=['resnet50','resnet18'], default='resnet50')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    transforms_img = transforms.Compose([
        transforms.Resize((384,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    if args.mode == 'train':
        ds = DeepFashionMM(args.data_root, transforms_img=transforms_img)
        val_split = int(len(ds)*0.1)
        if val_split < 1:
            val_split = max(1, int(len(ds)*0.1))
        indices = list(range(len(ds)))
        np.random.shuffle(indices)
        val_idx = indices[:val_split]
        train_idx = indices[val_split:]
        train_ds = torch.utils.data.Subset(ds, train_idx)
        val_ds = torch.utils.data.Subset(ds, val_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = MultiTaskFashionModel(backbone_name=args.backbone, use_keypoints=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        os.makedirs(args.save_dir, exist_ok=True)
        best_val = 1e9
        for epoch in range(1, args.epochs+1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict()},
                           os.path.join(args.save_dir, 'best.pth'))

    else: # infer
        assert args.image is not None, 'Provide --image for inference'
        model = MultiTaskFashionModel(backbone_name=args.backbone, use_keypoints=True)
        ckpt = os.path.join(args.save_dir, 'best.pth')
        if os.path.isfile(ckpt):
            d = torch.load(ckpt, map_location='cpu')
            model.load_state_dict(d['model_state'])
        else:
            print(f"[Warning] Checkpoint not found at {ckpt}. Using randomly initialized model.")
        model = model.to(device)

        decoded, region_colors = infer_single(model, args.image, device, transforms_img)

        result = {
            "attributes": decoded,
            "region_colors": region_colors
        }
        print('Inference results:')
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # also dump a visualization overlay to inspect regions
        visualize_color_regions(args.image, "vis_color_regions.png")

if __name__ == '__main__':
    main()
