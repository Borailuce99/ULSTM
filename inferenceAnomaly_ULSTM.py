import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import utils.ULSTM_Definitions as ULSTM_Definitions
import config
import csv

# --- CONFIG ---
PATH_ANOMALY_CSV = config.INF_DATASET
MODEL_CHECKPOINT = os.path.join(config.models_dir, "best_model")
DEVICE = config.cuda_device
SAVE_DIR = os.path.join(config.save_inference_dir, "anomaly_results")
os.makedirs(SAVE_DIR, exist_ok=True)

HIDDEN_STATE_DIR = os.path.join(config.save_inference_dir, "hidden_state_maps")
os.makedirs(HIDDEN_STATE_DIR, exist_ok=True)

VISUALIZE_EVERY_N_BATCHES = 1
MAX_BATCHES_TO_VIS = 20

def calculate_psnr(pred, gt, max_val=1.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

    return psnr.item()

def run_inference():
    model = ULSTM_Definitions.ULSTM(in_channels=3, num_classes=3, img_size=(256,256)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
    torch.cuda.empty_cache()
    model.eval()

    convlstm_layers = {
        'e1': model.e1.convLSTM,
        'e2': model.e2.convLSTM,
        'e3': model.e3.convLSTM,
        'e4': model.e4.convLSTM,
    }
    hidden_states = {}
    hooks = []
    for name, layer in convlstm_layers.items():
        def make_hook(n):
            def hook(module, inp, out):
                hidden_states[n] = out.detach().cpu()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(name)))

    dataset = ULSTM_Definitions.ULSTM_Dataset(PATH_ANOMALY_CSV)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)
    
    results = {}

    print(f"Iniciando inferencia...")
    batch_count = 0
    with torch.no_grad():
        for imgs, labels, target_paths, _ in tqdm(dataloader):
            sample_path = target_paths[0][0]

            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            predictions = model(imgs)

            if batch_count < MAX_BATCHES_TO_VIS and batch_count % VISUALIZE_EVERY_N_BATCHES == 0:
                for level_name in ['e1', 'e2', 'e3', 'e4']:
                    hs = hidden_states.get(level_name)
                    if hs is None:
                        continue
                    T = hs.size(1)
                    level_dir = os.path.join(HIDDEN_STATE_DIR, level_name)
                    for t in range(T):
                        ULSTM_Definitions.save_hidden_state_grid(
                            hs, level_name, t, level_dir, batch_idx=0, num_maps=32)
                        ULSTM_Definitions.save_hidden_state_overlay(
                            hs, imgs[0, t].cpu(), level_name, t, level_dir, batch_idx=0)
            batch_count += 1

            for b in range(predictions.size(0)):
                seq_psnr = 0.0
                seq_ssim = 0.0
                seq_lpips = 0.0
                sample_path = target_paths[b][0]
                seq_name = sample_path.split("/")[-2]

                for t in range(predictions.size(1)):
                    pred_frame = predictions[b][t].unsqueeze(0) # (1, C, H, W)
                    label_frame = labels[b][t].unsqueeze(0)     # (1, C, H, W)

                    seq_psnr += calculate_psnr(pred_frame.cpu(), label_frame.cpu())
                    seq_ssim += ssim_m(pred_frame, label_frame).item()
                    seq_lpips += lpips_m(pred_frame, label_frame).item()

                seq_avg_psnr = seq_psnr / predictions.size(1)
                seq_avg_ssim = seq_ssim / predictions.size(1)
                seq_avg_lpips = seq_lpips / predictions.size(1)

                if seq_name not in results:
                    results[seq_name] = {'psnr': [], 'ssim': [], 'lpips': []}
                results[seq_name]['psnr'].append(seq_avg_psnr)
                results[seq_name]['ssim'].append(seq_avg_ssim)
                results[seq_name]['lpips'].append(seq_avg_lpips)

    for h in hooks:
        h.remove()

    csv_filename = os.path.join(SAVE_DIR, "logs/metrics.csv")
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seq_name', 'psnr_evolution', 'ssim_evolution', 'lpips_evolution'])

        for name, metrics in results.items():
            writer.writerow([name, metrics['psnr'], metrics['ssim'], metrics['lpips']])
            
    print(f"Resultados guardados en {csv_filename}")
    return results

if __name__ == "__main__":
    results = run_inference()