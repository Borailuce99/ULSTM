import copy
import os
import random
import shutil
import zipfile
import numpy as np
import pandas as pd
import ast

import torch
import torchvision
from torch import nn
from torch.utils.data.dataset import Dataset
from PIL import Image

# from utils.ConvLSTMCell import ConvLSTMCell
from .ConvLSTM import ConvLSTM
import config

import matplotlib.pyplot as plt

# ULSTM (U-Net + LSTM) BASED ON PAPER: https://arxiv.org/pdf/1805.11247
device = torch.device(config.cuda_device)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-5) # Es equivalente a LayerNorm en visión y converje mejor en U-Nets
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.groupnorm(out)
        out = self.relu(out)
        out = self.conv2(out)
        output = self.relu(out)
        return output

class DownSample(nn.Module): # [Conv2D + ReLU] + MaxPool2D
    def __init__(self, in_channels, out_channels, frame_size):
        super().__init__()
        self.convLSTM = ConvLSTM(in_channels, out_channels, kernel_size=3, padding=1, activation="tanh", frame_size=frame_size)
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-5)
        self.conv2D_stable = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-5),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        self.down_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-5),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    
    def forward(self, x):
        B, T, C, H, W = x.size()
        clstm_out = self.convLSTM(x) 
        if torch.isnan(clstm_out).any():
            print("NaNs detectados DENTRO de la ConvLSTM del Encoder")

        reshaped_clstm = clstm_out.contiguous().view(B * T, clstm_out.size(2), H, W)
        clstm_norm = self.groupnorm(reshaped_clstm)                                                         # [B, T, C_out, H, W]

        skc_4d = self.conv2D_stable(clstm_norm)
        p_4d = self.down_conv(skc_4d)

        skc = skc_4d.view(B, T, skc_4d.size(1), H, W)
        p = p_4d.view(B, T, p_4d.size(1), p_4d.size(2), p_4d.size(3))

        return skc, p
  
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, frame_size):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_lstm_reconstruct = ConvLSTM(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            activation="relu",
            frame_size=frame_size
        )
        self.conv_final = DoubleConv(out_channels, out_channels)

    def forward(self, x1, skc):
        B, T, C_x1, H, W = x1.size()
        
        x1_4d = x1.view(B * T, C_x1, H, W)
        x1_up_4d = self.up(x1_4d)
        x1_up = x1_up_4d.view(B, T, C_x1, H*2, W*2)

        x = torch.cat([x1_up, skc], dim=2) 
        
        x = self.conv_lstm_reconstruct(x) # x es [B, T, out_channels, H*2, W*2]
        
        B, T, C_out, H_f, W_f = x.shape
        x_out = self.conv_final(x.view(B * T, C_out, H_f, W_f))
        
        return x_out.view(B, T, C_out, H_f, W_f)

class ULSTM(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, seq_len=5):
        super().__init__()
        H, W = img_size
        T = seq_len
        self.seq_len = T

        self.e1 = DownSample(in_channels, 64, frame_size=(H, W))
        self.e2 = DownSample(64, 128, frame_size=(H//2, W//2))
        self.e3 = DownSample(128, 256, frame_size=(H//4, W//4))
        self.e4 = DownSample(256, 512, frame_size=(H//8, W//8))

        self.bottle_neck = DoubleConv(512, 1024)
        
        s = config.img_size
        self.d4 = UpSample(1024, 512, frame_size=(s//8, s//8))
        self.d3 = UpSample(512, 256, frame_size=(s//4, s//4))
        self.d2 = UpSample(256, 128, frame_size=(s//2,s//2))
        self.d1 = UpSample(128, 64, frame_size=(s, s))

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1)

        self._initialize_weights()

    def forward(self, x):
        # Función para debugear errores nans
        def check(tensor, name):
            if torch.isnan(tensor).any():
                print(f"[!] NAN detectado en: {name}")
            if torch.isinf(tensor).any():
                print(f"[!] INF DETECTADO en: {name}")
        B, T, C, H, W = x.size()

        # --- ENCODER ---
        check(x, "Entrada")
        skc1, e1 = self.e1(x)                                           # x:  [B, 5, 3, 512, 512]   -> e1: [B, 5, 64, 256, 256]
        check(e1, "Encoder 1 (e1)")
        skc2, e2 = self.e2(e1)                                          # e1: [B, 5, 64, 256, 256]  -> e2: [B, 5, 128, 128, 128]
        check(e2, "Encoder 2 (e2)") 
        skc3, e3 = self.e3(e2)                                          # e2: [B, 5, 128, 128, 128] -> e3: [B, 5, 256, 64, 64]
        check(e3, "Encoder 3 (e3)")        
        skc4, e4 = self.e4(e3)                                          # e3: [B, 5, 256, 64, 64]   -> e4: [B, 5, 512, 32, 32]
        check(e4, "Encoder 4 (e4)")
        
        # --- BOTTLENECK ---
        b_input_4d = e4.contiguous().view(B * T, e4.size(2), e4.size(3), e4.size(4))     # e4: [B, T, 512, 32, 32]  -> b4d: [B*T, 512, 32, 32]
        b_4d = self.bottle_neck(b_input_4d)                                              # b4d: [B*5, 512, 32, 32]  -> b4d: [B*5, 1024, 32, 32]
        _, C_b, H_b, W_b = b_4d.size()
        b = b_4d.view(B, T, C_b, H_b, W_b)                                               # b4d: [B*5, 1024, 32, 32] -> b: [B, 5, 1024, 32, 32]    
        check(b, "Bottleneck")
        
        # --- DECODER ---
        d4 = self.d4(b, skc4)
        check(d4, "Decoder (d4)")
        d3 = self.d3(d4, skc3)
        check(d3, "Decoder (d3)")
        d2 = self.d2(d3, skc2)
        check(d2, "Decoder (d2)")
        d1 = self.d1(d2, skc1)
        check(d1, "Decoder (d1)")

        # --- OUTPUT ---
        out_4d = d1.contiguous().view(B * T, d1.size(2), d1.size(3), d1.size(4))   # d1: [B, T, 64, H, W]  -> out4d: [B*T, 64, H, W]
        out_4d = self.out(out_4d)  
        out = out_4d.view(B, T, out_4d.size(1), H, W)
        frame_predicted = torch.sigmoid(out)   # Evitar que la salida colapse a 0 o 1
        return frame_predicted

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif 'ConvLSTM' in m.__class__.__name__:
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class ULSTM_Dataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        self.input_images = data['input_frames'].apply(ast.literal_eval)
        self.label_images = data['output_frames'].apply(ast.literal_eval)
        self.anomaly = data['anomaly']

    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, index):
        input_images = []
        label_images = []
        target_paths = self.label_images[index]

        for path in self.input_images[index]:
            try:
                img = Image.open(path).convert("RGB")
                input_images.append(self.transform_img(img))
            except Exception as e:
                print(f"Warning: Failed to open image at {path}. Skipping this image. Error: {e}")
                continue
        for path in self.label_images[index]:
            try:
                img = Image.open(path).convert("RGB")
                label_images.append(self.transform_img(img))
            except Exception as e:
                print(f"Warning: Failed to open label image at {path}. Skipping this image. Error: {e}")
                continue
        if len(input_images) > 0 and len(label_images) > 0:
            input_tensor = torch.stack(input_images, dim=0)
            label_tensor = torch.stack(label_images, dim=0)
            
            return input_tensor, label_tensor, target_paths, self.anomaly[index]
        return torch.empty(0), torch.empty(0), [], 0
    
    def cropImg(self, image):
        return torchvision.transforms.functional.crop(image, 200, 0, 520, 1080)
    
    def transform_img(self, image):
        transform_ops = torchvision.transforms.Compose([
                                                # torchvision.transforms.Lambda(self.cropImg),
                                                # torchvision.transforms.Resize((config.img_resize, config.img_resize)),
                                                torchvision.transforms.ToTensor()
        ])

        return transform_ops(image)

# --- FEATURE MAP VISUALIZATION UTILITIES ---
activations = {}
def hook_forward(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

def save_feature_maps(activations_tensor, epoch, name, num_maps=32):
    """
    Visualiza los primeros 'num_maps' mapas de características del bottleneck.
    
    Args:
        activations_tensor (torch.Tensor): El tensor capturado por el hook [B, C, H, W].
        num_maps (int): Cuántos mapas (canales) visualizar.
    """
    feature_maps = activations_tensor[0]

    if config.DEBUG:
        print(f"Editing {config.save_dir}/activations.txt")
        with open(f"{config.save_dir}/activations.txt", "a") as f:
            if name == "encoder_layer_1stable":
                f.write(f"\n===================== EPOCH {epoch} =========================\n")
            f.write(f"Feature map {name}:\n{activations_tensor[0]}\n")
        print(f"activations.txt updated")
    num_maps_to_show = min(num_maps, feature_maps.shape[0]) 

    fig, axes = plt.subplots(1, num_maps_to_show, figsize=(2 * num_maps_to_show, 2))
    
    if num_maps_to_show == 1:
        axes = [axes]

    for i in range(num_maps_to_show):
        ax = axes[i]
        
        feature_map = feature_maps[i].numpy()

        min_val = feature_map.min()
        max_val = feature_map.max()
        if max_val > min_val:
            normalized_map = (feature_map - min_val) / (max_val - min_val)
        else:
            normalized_map = feature_map # Si es constante (ej: todo cero o un valor fijo)

        ax.imshow(normalized_map, cmap='viridis') 
        ax.set_title(f"Mapa {i+1}", fontsize=8)
        ax.axis('off')
        
    plt.suptitle("Mapas de Características del Bottleneck [32x32]", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(config.feature_maps_dir, f"epoch_{epoch}_{name}_feature_maps.png"))
    plt.close(fig)


def save_hidden_state_grid(hidden_tensor, level_name, time_step, save_dir, batch_idx=0, num_maps=32):
    
    feature_maps = hidden_tensor[batch_idx, time_step]  # [C, H, W]
    C, H, W = feature_maps.shape
    num_maps = min(num_maps, C)
    ncols = min(num_maps, 8)
    nrows = (num_maps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    axes = axes.flatten() if nrows > 1 else (axes if num_maps > 1 else [axes])

    for i in range(num_maps):
        ax = axes[i]
        fm = feature_maps[i].detach().cpu().numpy()
        mn, mx = fm.min(), fm.max()
        if mx > mn:
            fm = (fm - mn) / (mx - mn)
        ax.imshow(fm, cmap='viridis')
        ax.set_title(f"Ch {i+1}", fontsize=7)
        ax.axis('off')

    for i in range(num_maps, len(axes)):
        axes[i].axis('off')

    suffix = f"t{time_step}" if num_maps >= ncols else f"t{time_step}"
    plt.suptitle(f"{level_name} hidden state — t={time_step} [{H}x{W}]", fontsize=10)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"batch{batch_idx}_{level_name}_t{time_step}_grid.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_hidden_state_overlay(hidden_tensor, input_image, level_name, time_step, save_dir, batch_idx=0):

    fm = hidden_tensor[batch_idx, time_step]  # [C, H, W]
    heatmap = fm.mean(dim=0).detach().cpu().numpy()  # [H, W]
    mn, mx = heatmap.min(), heatmap.max()
    if mx > mn:
        heatmap = (heatmap - mn) / (mx - mn)

    img = input_image.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[time_step] if time_step < img.shape[0] else img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
    img = np.clip(img.squeeze(), 0, 1)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    H, W = heatmap.shape
    img_resized = img

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis('off')

    im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f"{level_name} t={time_step} (avg ch)", fontsize=9)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_resized, alpha=0.6)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.4, vmin=0, vmax=1)
    axes[2].set_title("Overlay", fontsize=9)
    axes[2].axis('off')

    plt.suptitle(f"{level_name} — Hidden State Heatmap t={time_step}", fontsize=11)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"batch{batch_idx}_{level_name}_t{time_step}_overlay.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)