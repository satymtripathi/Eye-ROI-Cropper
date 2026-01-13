import os
# --- FIX FOR OMP ERROR ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------

import argparse
import sys
import glob
import time
from datetime import timedelta
import cv2
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import segmentation_models_pytorch as smp
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing dependency {e.name}. Please install requirements.txt")
    print("pip install -r requirements.txt")
    sys.exit(1)

# ==========================================
# CONFIGURATION & ARGS
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Unified ROI Crop Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input folder containing images")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output folder for cropped images")
    parser.add_argument("--type", "-t", type=str, required=True, choices=['standard', 'slitlamp'], help="Model type to use")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Device (cuda/cpu)")
    parser.add_argument("--padding", "-p", type=int, default=0, help="Padding around crop in pixels")
    return parser.parse_args()

# ==========================================
# DATASET
# ==========================================
IMG_SIZE = 768

class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Image is None")
            original_h, original_w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                augmented = self.transform(image=image_rgb)
                image_tensor = augmented['image']
            else:
                 # Fallback manual transform if needed, but A.Compose is standard
                 pass
                 
            return image_tensor, img_path, (original_h, original_w)
            
        except Exception as e:
            # Return dummy to keep batching alive
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), img_path, (0, 0)

def get_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ==========================================
# MAIN
# ==========================================
def main():
    args = parse_args()
    
    # 1. Setup Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"PIPELINE: {args.type.upper()} | DEVICE: {device}")
    
    # 2. Select Model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.type == 'standard':
        model_filename = "model_standard.pth"
    else:
        model_filename = "model_slitlamp.pth"
        
    model_path = os.path.join(script_dir, model_filename)
    if not os.path.exists(model_path):
         print(f"Error: Model file '{model_filename}' not found in {script_dir}")
         return

    # 3. Load Images
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return
        
    os.makedirs(args.output, exist_ok=True)
    
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    
    if not image_paths:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(image_paths)} images.")

    # 4. Load Model Structure
    print("Loading model...")
    try:
        model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 5. Pipeline
    dataset = InferenceDataset(image_paths, transform=get_transform())
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=(device == 'cuda')
    )

    print("Starting processing...")
    start_time = time.time()
    processed_count = 0
    
    with torch.no_grad():
        with tqdm(total=len(image_paths), unit="img") as pbar:
            for batch_images, batch_paths, batch_dims in loader:
                batch_images = batch_images.to(device)
                
                # Inference
                if device == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_images)
                        probs = torch.sigmoid(outputs)
                else:
                    outputs = model(batch_images)
                    probs = torch.sigmoid(outputs)
                
                probs = probs.cpu().numpy()
                
                # Post-processing
                for i in range(len(batch_paths)):
                    img_path = batch_paths[i]
                    prob_map = probs[i][0]
                    orig_h = batch_dims[0][i].item()
                    orig_w = batch_dims[1][i].item()
                    
                    if orig_h == 0: continue 
                    
                    # Resize mask
                    mask = (prob_map > 0.5).astype(np.uint8)
                    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Crop
                    coords = cv2.findNonZero(mask_resized)
                    original_img = cv2.imread(img_path)
                    
                    if coords is not None and original_img is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        
                        # Apply padding
                        pad = args.padding
                        x = max(0, x - pad)
                        y = max(0, y - pad)
                        w = min(orig_w - x, w + 2*pad)
                        h = min(orig_h - y, h + 2*pad)
                        
                        crop = original_img[y:y+h, x:x+w]
                    else:
                        crop = original_img # Fallback
                        
                    # Save
                    base_name = os.path.basename(img_path)
                    save_path = os.path.join(args.output, base_name)
                    if crop is not None:
                        cv2.imwrite(save_path, crop)
                    
                    processed_count += 1
                    pbar.update(1)

    total_time = time.time() - start_time
    print(f"\nDone! Processed {processed_count} images in {str(timedelta(seconds=int(total_time)))}")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
