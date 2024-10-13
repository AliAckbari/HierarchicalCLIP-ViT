# Imports
from torchvision import transforms
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal
import numpy as np
import importlib
from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTFeatureExtractor
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn


import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from braindecode.augmentation import FTSurrogate, SmoothTimeMask, ChannelsDropout

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
image_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip images horizontally
    transforms.RandomRotation(degrees=15),    # Random rotation
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Random color jitter
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),  # Randomly crop and resize
])




def v1_filtering_all_channels(image):
    filtered_channels = []
    for i in range(3): 
        channel = image[:, :, i]
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        filtered_channels.append(edges)
    
    filtered_image = np.stack(filtered_channels, axis=-1)
    
    return filtered_image

def v2_filtering(image):
    # Define LBP parameters (reduce the radius to mitigate patterns)
    radius = 1
    n_points = 8 * radius

    image = cv2.bilateralFilter(image, 9, 50, 50)
    lbp_channels = []
    edge_channels = []

    for i in range(3):  # Assuming the input image is RGB
        channel = image[:, :, i]
        channel = cv2.GaussianBlur(channel, (5, 5), 0)
        lbp = local_binary_pattern(channel, n_points, radius, method="uniform")
        lbp = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lbp_channels.append(lbp)
        edge_channels.append(sobel)
    lbp_image = np.stack(lbp_channels, axis=-1)
    edge_image = np.stack(edge_channels, axis=-1)
    combined_image = cv2.addWeighted(lbp_image, .5, edge_image, 1, 0.5)

    return combined_image




def v3_filtering(image):
    filtered_channels = []
    for i in range(3):  
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1] 
        filtered_channels.append(saturation)
    filtered_image = np.stack(filtered_channels, axis=-1)
    
    return filtered_image




def apply_custom_filters(image, filter_type):
    if filter_type == 'v1':
        return v1_filtering_all_channels(image)
    elif filter_type == 'v2':
        return v2_filtering(image)
    elif filter_type == 'v3':
        return v3_filtering(image)
    else:
        raise ValueError("Invalid filter type. Choose 'v1', 'v2', or 'v3'.")

def process_and_save_images(images, filter_type, file_prefix=''):
    num_images, _, height, width = images.shape
    batch_size = 16
    
    # Create directory if it doesn't exist
    save_dir = 'filtered_images'
    os.makedirs(save_dir, exist_ok=True)
    
    for start_idx in tqdm(range(0, num_images, batch_size), desc=f"Processing {filter_type} filter"):
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images[start_idx:end_idx].numpy().transpose(0, 2, 3, 1)
        filtered_images = np.zeros_like(batch_images)
        for i in range(len(batch_images)):
            filtered_images[i] = apply_custom_filters(batch_images[i], filter_type)
        filtered_images_tensor = torch.tensor(filtered_images).permute(0, 3, 1, 2).float()
        # Save to file in the specified directory
        torch.save(filtered_images_tensor, os.path.join(save_dir, f'{file_prefix}_{filter_type}_batch_{start_idx // batch_size}.pt'))






# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, patch_dropout = 0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype
        img = img.unsqueeze(1)
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.patch_dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return x
    
    
    
class EEGEncoder(nn.Module):
        def __init__(self, model=None, pretrained=True, trainable=True):
            super().__init__()

            model = SimpleViT(
                image_size = (128,440),
                patch_size = (8, 20),
                num_classes = 40,
                dim = 256,
                depth = 4,
                dim_head=16,
                heads = 16,
                mlp_dim = 16,
                channels = 1
            ).to(device)
            self.model = model
            self.model = torch.nn.DataParallel(self.model).to(device)
            eegckpt   = '/home/ubuntu/EXPERIMENT_276/bestckpt/eegfeat_all_0.6015625.pth'
            eegcheckpoint = torch.load(eegckpt, map_location=device)
            self.model.load_state_dict(eegcheckpoint['model_state_dict'])
            print('Loading EEG checkpoint from previous epoch: {}'.format(eegcheckpoint['epoch']))
          
            
        def forward(self, x):
              return self.model(x)


class ImgEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", pretrained=True, trainable=True):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(768, 256)
        )
        if pretrained:
            self.model = ViTModel.from_pretrained(model_name)
        else:
            self.model = ViTModel(config=ViTConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable
            
        
            

    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        last_hidden_state = output.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :] 
        return self.mlp_head(cls_embedding)



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        projection_dim=128,
        dropout=.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x



class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=1,
        EEG_embedding=256,
        ImgEncoder_dim=256,
        image_encoder=None,
        text_encoder=None,
        image_projection=None,
        text_projection=None,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder().to(device)
        self.img_encoder = ImgEncoder().to(device)  # Updated to img_encoder
        self.img_encoder     = torch.nn.DataParallel(self.img_encoder).to(device)
        eegckpt   = image_encoder
        eegcheckpoint = torch.load(eegckpt, map_location=device)
        self.img_encoder.load_state_dict(eegcheckpoint['model_state_dict'])
    
    
        self.eeg_projection = ProjectionHead(embedding_dim=EEG_embedding).to(device)
        self.img_projection = ProjectionHead(embedding_dim=ImgEncoder_dim).to(device)
        self.temperature = temperature

    def forward(self, batch):
        eeg_features = self.eeg_encoder(batch["eeg"])  # Corrected from EEGEncoder to eeg_encoder
        image_features = self.img_encoder(batch["image"])  # Corrected from ImgEncoder to img_encoder
        eeg_embeddings = self.eeg_projection(eeg_features)
        image_embeddings = self.img_projection(image_features)

        logits = (eeg_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        eeg_similarity = eeg_embeddings @ eeg_embeddings.T
        targets = F.softmax(
            (images_similarity + eeg_similarity) / 2 * self.temperature, dim=-1
        )
        eeg_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + 2 * eeg_loss) / 2.0  # shape: (batch_size)
        return loss.mean()



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
    
    
    
    
class K_means:
    def __init__(self, n_clusters=40, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state
    def transform(self, embed, gt_labels):
        pred_labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10).fit_predict(embed)
        accuracy = self.cluster_metrics(gt_labels, pred_labels)
        return accuracy

    def cluster_metrics(self, y_true, y_pred):

        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        ind = linear_assignment(w.max() - w)
        accuracy = sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
        precision = np.zeros(D)
        for cluster in range(D):
              if np.sum(w[cluster, :]) > 0: 
                precision[cluster] = w[cluster, cluster] / np.sum(w[cluster, :])


        overall_precision = np.mean(precision)

        return accuracy

def train_epoch(model, train_loader, optimizer, scheduler, device, writer=None, epoch=None):
    model.train()
    loss_meter = AvgMeter()
    train_losses = []

    tqdm_object = tqdm(train_loader, desc="Training", total=len(train_loader))

    for batch in tqdm_object:
        eeg_batch ,labels, images_batch = batch
        norm   = torch.max(eeg_batch) / 2.0
        eeg_batch    = (eeg_batch - norm)/ norm
        eeg_batch = eeg_batch.to(device)
        eeg_batch = eeg_batch.permute(0, 2, 1)
        
        augmented_images = []
        for img_tensor in images_batch:
            img = transforms.ToPILImage()(img_tensor)  # Convert tensor to PIL
            img = image_augmentations(img)  # Apply augmentations
            augmented_images.append(img)

        augmented_images = [transforms.ToTensor()(img) for img in augmented_images]
        augmented_images = torch.stack(augmented_images).to(device)
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

        images_batch = feature_extractor(images=images_batch, return_tensors="pt").pixel_values.to(device)
        

        
        eeg_batch, labels = eeg_batch.to(device), labels.to(device)
        images_batch = images_batch.to(device)
        batch = {
            'image': images_batch,
            'eeg': eeg_batch
        }
        
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        
        optimizer.step()
         

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        train_losses.append(loss.item())

        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    return loss_meter.avg, train_losses

def evaluate_model(model, test_loader, device, writer=None, epoch=None):
    model.eval()
    test_loss_meter = AvgMeter()
    test_losses = []

    tqdm_object = tqdm(test_loader, desc="Evaluating", total=len(test_loader))
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    with torch.no_grad():
        for batch in tqdm_object:
            
            eeg_batch , _ ,images_batch = batch
            norm   = torch.max(eeg_batch) / 2.0
            eeg_batch    = (eeg_batch - norm)/ norm
            eeg_batch = eeg_batch.to(device)
            eeg_batch = eeg_batch.permute(0, 2, 1)

            images_batch = images_batch.to(device)
            images_batch = feature_extractor(images=images_batch, return_tensors="pt").pixel_values.to(device)

            batch = {
                'image': images_batch,
                'eeg': eeg_batch
            }
            loss = model(batch)
            count = batch["image"].size(0)
            test_loss_meter.update(loss.item(), count)
            test_losses.append(loss.item())

            tqdm_object.set_postfix(test_loss=test_loss_meter.avg)

    running_loss      = []
    eeg_featvec       = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])

    tq = tqdm(test_loader, desc="Evaluating", total=len(test_loader))
    model.eval()
    for batch_idx, (_, labels, imgs) in enumerate(tq, start=1):
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        images_batch = feature_extractor(images=imgs, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            x_proj = model.img_encoder(images_batch).to(device)
            x_proj = model.img_projection(x_proj).to(device)
        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    num_clusters   = 40
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print("[Epoch: {}, Val KMeans score Proj: {}]".format(epoch, clustering_acc_proj))
    
    
    
    running_loss      = []
    eeg_featvec       = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])

    tq = tqdm(test_loader, desc="Evaluating", total=len(test_loader))
    model.eval()
    for batch_idx, (eeg, labels, imgs) in enumerate(tq, start=1):
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        images_batch = feature_extractor(images=imgs, return_tensors="pt").pixel_values.to(device)
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm
        eeg, labels = eeg.to(device), labels.to(device)
        eeg = eeg.permute(0, 2, 1)
        with torch.no_grad():
            x_proj = model.eeg_encoder(eeg).to(device)
            x_proj = model.eeg_projection(x_proj).to(device)
        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    num_clusters   = 40
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print("[Epoch: {}, Val KMeans score Proj: {}]".format(epoch, clustering_acc_proj))
    
    model.train()
    
    
    

    return clustering_acc_proj


device = "cuda"
Sur = FTSurrogate(probability=0.5, phase_noise_magnitude=1).to(device)
mask = SmoothTimeMask(probability=0.5, mask_len_samples=50).to(device)

seed = 45
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)




train_eeg_data = torch.load("/home/ubuntu/train_eeg_data.pt")
test_eeg_data = torch.load("/home/ubuntu/test_eeg_data.pt")


train_labels = torch.load("/home/ubuntu/train_labels.pt")
test_labels = torch.load("/home/ubuntu/test_labels.pt")


train_images = torch.load("/home/ubuntu/train_images.pt")
test_images = torch.load("/home/ubuntu/test_images.pt")


print(f"Training data shape: {train_eeg_data.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing data shape: {test_eeg_data.shape}")
print(f"Testing labels shape: {test_labels.shape}")


train_dataset = TensorDataset(train_eeg_data, train_labels, train_images)
test_dataset = TensorDataset(test_eeg_data, test_labels, test_images)


batch_size = 128 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")


num_epochs = 75

# Model 1
model_1 = CLIPModel(image_encoder="/home/ubuntu/EXPERIMENT_230/bestckpt/eegfeat_mid_all_0.9748263888888888.pth").to(device)
params_1 = [
    {"params": model_1.eeg_encoder.parameters(), "lr": 1e-4},
    {"params": itertools.chain(
        model_1.img_projection.parameters(), model_1.eeg_projection.parameters()
    ), "lr": 1e-3, "weight_decay": 1e-3}
]
optimizer_1 = torch.optim.AdamW(params_1, weight_decay=0.)
scheduler_1 = CosineAnnealingLR(optimizer_1, T_max=num_epochs)

# Model 2
model_2 = CLIPModel(image_encoder="/home/ubuntu/EXPERIMENT_264/checkpoints/eegfeat_band_final_all.pth").to(device)
params_2 = [
    {"params": model_2.eeg_encoder.parameters(), "lr": 1e-4},
    {"params": itertools.chain(
        model_2.img_projection.parameters(), model_2.eeg_projection.parameters()
    ), "lr": 1e-3, "weight_decay": 1e-3}
]
optimizer_2 = torch.optim.AdamW(params_2, weight_decay=0.)
scheduler_2 = CosineAnnealingLR(optimizer_2, T_max=num_epochs)

# Model 3
model_3 = CLIPModel(image_encoder="/home/ubuntu/EXPERIMENT_266/checkpoints/eegfeat_band_final_all.pth").to(device)
params_3 = [
    {"params": model_3.eeg_encoder.parameters(), "lr": 1e-4},
    {"params": itertools.chain(
        model_3.img_projection.parameters(), model_3.eeg_projection.parameters()
    ), "lr": 1e-3, "weight_decay": 1e-3}
]
optimizer_3 = torch.optim.AdamW(params_3, weight_decay=0.)
scheduler_3 = CosineAnnealingLR(optimizer_3, T_max=num_epochs)

# Model 4
model_4 = CLIPModel(image_encoder="/home/ubuntu/EXPERIMENT_225/bestckpt/eegfeat_all_0.9704861111111112.pth").to(device)
params_4 = [
    {"params": model_4.img_encoder.parameters(), "lr": 1e-4, "weight_decay": 1e-2},
    {"params": model_4.eeg_encoder.parameters(), "lr": 1e-4, "weight_decay": 1e-2},
    {"params": itertools.chain(
        model_4.img_projection.parameters(), model_4.eeg_projection.parameters()
    ), "lr": 1e-3, "weight_decay": 1e-1}
]
optimizer_4 = torch.optim.Adam(params_4, weight_decay=0.)
scheduler_4 = CosineAnnealingLR(optimizer_4, T_max=num_epochs)

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]



def train_and_evaluate(model, optimizer, scheduler, model_name, num_epochs, train_loader, test_loader):
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    best_test_acc = 0
    test_acc = evaluate_model(model, test_loader, device, epoch=1)
    print(test_acc)
    best_test_acc = test_acc
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} for {model_name}")

        train_loss, train_losses = train_epoch(model, train_loader, optimizer, scheduler, device, epoch=epoch)
        train_losses_per_epoch.append(train_losses)
        

        test_acc = evaluate_model(model, test_loader, device, epoch=epoch)
 
        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'{model_name}_best_model_updated.pth')
            print(f"Model checkpoint saved.")


    plt.figure(figsize=(12, 6))
    train_losses_flat = [loss for epoch_losses in train_losses_per_epoch for loss in epoch_losses]
    test_losses_flat = [loss for epoch_losses in test_losses_per_epoch for loss in epoch_losses]

    plt.plot(train_losses_flat, label=f'{model_name} Train Loss')
    plt.plot(test_losses_flat, label=f'{model_name} Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Train and Test Loss')
    plt.legend()
    plt.show()




class FilteredImageAndEEGDataset(Dataset):
    def __init__(self, images, eeg_data, labels, filter_type):
        self.images = images
        self.eeg_data = eeg_data
        self.filter_type = filter_type
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].numpy().transpose(1, 2, 0)  
        filtered_image = apply_custom_filters(image, self.filter_type) 
        filtered_image_tensor = torch.tensor(filtered_image).permute(2, 0, 1).float() 
        eeg_tensor = self.eeg_data[idx].float() 
        
        return eeg_tensor, self.labels[idx] ,filtered_image_tensor

def create_dataloader(images, eeg_data, labels, filter_type, batch_size=16, shuffle=True, num_workers=0):
    dataset = FilteredImageAndEEGDataset(images, eeg_data, labels, filter_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


num_workers = 8
batch_size = 128

chkp = torch.load("//home/ubuntu/model_4_org_vit_best_model_updated.pth", map_location="cuda")
model_4.load_state_dict(chkp)
train_and_evaluate(model_4, optimizer_4, scheduler_4, "model_4_org_vit", 50, train_loader, test_loader)


train_loader_low = create_dataloader(train_images, train_eeg_data, train_labels,'v1', batch_size=batch_size, num_workers=num_workers)
test_loader_low = create_dataloader(test_images, test_eeg_data, test_labels,'v1', batch_size=batch_size, num_workers=num_workers, shuffle = False)

chkp = torch.load("/home/ubuntu/model_v1_2_vit_best_model.pth", map_location="cuda")
model_1.load_state_dict(chkp)
train_and_evaluate(model_1, optimizer_1, scheduler_1, "model_v1_2_vit", num_epochs, train_loader_low, test_loader_low)


train_loader_band = create_dataloader(train_images, train_eeg_data, train_labels, 'v2', batch_size=batch_size, num_workers=num_workers)
test_loader_band = create_dataloader(test_images, test_eeg_data, test_labels, 'v2' , batch_size=batch_size, num_workers=num_workers, shuffle = False)

train_and_evaluate(model_2, optimizer_2, scheduler_2, "model_v2_2_vit", num_epochs, train_loader_band, test_loader_band)


train_loader_high = create_dataloader(train_images, train_eeg_data, train_labels,'v3', batch_size=batch_size, num_workers=num_workers)
test_loader_high = create_dataloader(test_images, test_eeg_data, test_labels,'v3', batch_size=batch_size, num_workers=num_workers, shuffle = False)

train_and_evaluate(model_3, optimizer_3, scheduler_3, "model_v3_3_vit", num_epochs, train_loader_high, test_loader_high)













