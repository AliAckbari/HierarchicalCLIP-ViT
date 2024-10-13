import torch
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EEG Data
train_data_path = ""
val_data_path   = ""
test_data_path   = ""

image_height = 128
image_width  = 128
input_channel= 3
batch_size   = 32
num_workers  = 8
latent_dim   = 512
# n_classes    = 40
# n_subjects   = 6
diff_augment_policies = "color,translation,cutout"
lr           = 3e-4
gen_lr       = 3e-4
dis_lr       = 3e-4
beta_1       = 0.5
beta_2       = 0.9999
EPOCH        = 5001
num_col      = 32
c_dim        = 256
dis_level    = 3
feat_dim       = 512
projection_dim = 512
test_batch_size= 512
generate_image = 50000
fig_freq       = 10
ckpt_freq      = 10
generate_batch_size = 1
num_layers     = 4
generate_freq  = 200
dataset_name   = 'CVPR40'