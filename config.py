import os

BASE_DIR = os.path.dirname(__file__)
DATASET = "<path-to-dataset-csv>"
INF_DATASET = "<path-to-inference-dataset-csv>"

num_epochs = 1000
batch_size = 2
prefetch_factor = 8
num_workers = 4
lr = 1e-4
step_size = 2
gamma = 0.9
valid_size = 2
valid_epoch = 10

img_size = 256

cuda_device = "cuda:0"

dataset_name = os.path.splitext(os.path.basename(DATASET))[0]
save_dir = os.path.join("<save-path>")
save_inference_dir = os.path.join("<inference-save-path>")
figures_dir = os.path.join(save_dir, "figures")
models_dir = os.path.join(save_dir, "models")
feature_maps_dir = os.path.join(save_dir, "feature_maps")
psnr_file_path = os.path.join(save_dir, "psnr_values.txt")
loss_file_path = os.path.join(save_dir, "loss_value.txt")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(feature_maps_dir, exist_ok=True)

DEBUG = True