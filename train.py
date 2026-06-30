import os
import gc
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

import config
import utils.ULSTM_Definitions as ULSTM_Definitions

import random

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
dataset = ULSTM_Definitions.ULSTM_Dataset(config.DATASET)
generator = torch.Generator().manual_seed(25)

train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
test_dataset, val_dataset = random_split(test_dataset, [0.5,  0.5], generator=generator)

train_dataloader = DataLoader(dataset=train_dataset,
                          num_workers=config.num_workers, pin_memory=True,
                          batch_size=config.batch_size,
                          shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,
                            num_workers=config.num_workers, pin_memory=True,
                            batch_size=config.batch_size,
                            shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,
                             num_workers=config.num_workers, pin_memory=True,
                             batch_size=config.batch_size,
                             shuffle=True)

best_val_loss = float('inf')

model = ULSTM_Definitions.ULSTM(
    in_channels=3, 
    num_classes=3, 
    img_size=(config.img_size, config.img_size), 
    seq_len=5).to(config.cuda_device)
def count_parameters(model):
    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_param

num_params = count_parameters(model)
print(f"Total Trainable Parameters: {num_params:,}")
bytes_per_param = 4
model_size_bytes = num_params * bytes_per_param
model_size_MB = model_size_bytes / (1024 * 1024)
print(f"Total Estimated Size (FP32): {model_size_MB:.2f} MB")

# For Feature Map Visualization =============
hooks_names = ["encoder_layer_1stable", "encoder_layer_2stable", "encoder_layer_3stable", "encoder_layer_4stable", "bottleneck_conv1", "bottleneck_conv2", "decoder_layer_4", "decoder_layer_3", "decoder_layer_2", "decoder_layer_1", "output_layer"]
en_layers = [model.e1.conv2D_stable, model.e2.conv2D_stable,
             model.e3.conv2D_stable, model.e4.conv2D_stable]
de_layers = [model.d4.conv_final.conv2, model.d3.conv_final.conv2,
             model.d2.conv_final.conv2, model.d1.conv_final.conv2, model.out]
layers_to_register = en_layers + [model.bottle_neck.conv1, model.bottle_neck.conv2] + de_layers
all_hooks_handles = []

optimizer = optim.AdamW(model.parameters(), lr=config.lr, eps=1e-4)
criterion = nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# --- LOSS LOG ---
if os.path.exists(os.path.join(config.models_dir, f"loss_log.txt")):
    os.remove(os.path.join(config.models_dir, f"loss_log.txt"))

open(os.path.join(config.models_dir, f"loss_log.txt"), "w").close()

# COPY TRAIN CODE TO KEEP CHANGES
import shutil
shutil.copyfile("./train_ViT_WIP.py", os.path.join(config.save_dir, "train_ViT_WIP.py"))
shutil.copyfile("./config.py", os.path.join(config.save_dir, "config.py"))
shutil.copyfile("./utils/ULSTM_Definitions.py", os.path.join(config.save_dir, "ULSTM_Definitions.py"))

# To accelerate train with RTX50 - FP16
scaler = torch.amp.GradScaler('cuda', init_scale=2.**10)

model.load_state_dict(torch.load("/raid2/datasets/ShanghaiTech/shanghaitech/ULSTM_OUTPUT/shanghaitech256_train_5to5_5step_1000_4/models/best_model"))

for epoch in tqdm(range(config.num_epochs)):
    model.train()
    total_train_loss = 0
    idx = 0
    for input_seq, target_frame, _, _ in train_dataloader:
        if input_seq is None or target_frame is None:
            print(f"Batch None: {input_seq}\t{target_frame}")
            continue
        input_seq = input_seq.to(config.cuda_device)        # [B, 5, 3, H, W]
        target_frame = target_frame.to(config.cuda_device)  # [B, 5, 3, H, W]

        optimizer.zero_grad()
        if not torch.isfinite(input_seq).all() or not torch.isfinite(target_frame).all():
            print("¡Detectado valor No-Finito (NaN/Inf) en el tensor del Dataset!")
            continue
        pred = model(input_seq)
        loss = criterion(pred, target_frame)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        idx += 1
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    with open(os.path.join(config.models_dir, f"loss_log.txt"), "a") as f:
        f.write(f"Epoch {epoch}/{config.num_epochs}: Total Train Loss: {total_train_loss} | Avg Train Loss: {avg_train_loss}\n")
    torch.cuda.empty_cache()
    gc.collect()

    model.eval()
    if True:
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, _, _ in tqdm(val_dataloader, desc=f"Epoch {epoch}/{config.num_epochs} [Val]"):
                imgs, labels = imgs.to(config.cuda_device), labels.to(config.cuda_device)
                preds = model(imgs)
                loss = criterion(preds, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        if torch.isnan(torch.tensor(avg_val_loss)):
            print("Detección de NaN: Deteniendo entrenamiento para evitar daños al modelo.")
            break
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.models_dir, "best_model"))
            print(f"Saved new best model (val_loss={avg_val_loss:.4f})")
        else:
            print(f"Not saved new model because {avg_val_loss} > {best_val_loss}")

        with torch.no_grad():
            imgs, labels, _, _ = next(iter(test_dataloader))
            imgs, labels = imgs.to(config.cuda_device), labels.to(config.cuda_device)
            batch_size = imgs.size(0)
            idx = random.randint(0, batch_size - 1)
            preds = model(imgs[idx].unsqueeze(0))
            for i in range(preds.size(1)):
                save_image(preds[0][i], os.path.join(config.figures_dir, f"epoch_{epoch}_pred{idx}_{i}.png"))
                save_image(labels[idx][i], os.path.join(config.figures_dir, f"epoch_{epoch}_gt{idx}_{i}.png"))
        
        with torch.no_grad():
            temp_hooks = []
            for name, layer in zip(hooks_names, layers_to_register):
                handle = layer.register_forward_hook(ULSTM_Definitions.hook_forward(name))
                temp_hooks.append(handle)
            data, target, _, _ = next(iter(val_dataloader))
            data = data.to(config.cuda_device)
            output = model(data)

            hook_outputs = []
            for idx, hook in enumerate(all_hooks_handles):
                handle = ULSTM_Definitions.activations[f'{hooks_names[idx]}']
                ULSTM_Definitions.save_feature_maps(handle, epoch, hooks_names[idx], num_maps=32)

            for handle in temp_hooks:
                handle.remove()

            ULSTM_Definitions.activations.clear()

    torch.cuda.empty_cache()
    gc.collect()