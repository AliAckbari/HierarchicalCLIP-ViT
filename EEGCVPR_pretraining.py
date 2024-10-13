# Imports
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
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tqdm import tqdm
import numpy as np
import pdb
import cv2
from glob import glob
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import decimate
from numpy import linspace
from braindecode.augmentation import FTSurrogate, SmoothTimeMask, ChannelsDropout
from natsort import natsorted

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange






seed = 45
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False



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
## place it in utils##




class K_means:
    def __init__(self, n_clusters=40, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state
    def transform(self, embed, gt_labels):
        pred_labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10).fit_predict(embed)
        accuracy = self.cluster_metrics(gt_labels, pred_labels)
        return accuracy

    def cluster_metrics(self, y_true, y_pred):
        """
        Calculate clustering accuracy and precision. Require scikit-learn installed

        Arguments:
          y_true: true labels, numpy.array with shape `(n_samples,)`
          y_pred: predicted labels, numpy.array with shape `(n_samples,)`

        Returns:
          accuracy: float, in [0,1]
          precision: float, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        # Calculate accuracy (same as before)
        ind = linear_assignment(w.max() - w)
        accuracy = sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

        # Calculate precision for each cluster
        precision = np.zeros(D)
        for cluster in range(D):
              if np.sum(w[cluster, :]) > 0:  # Avoid division by zero
                precision[cluster] = w[cluster, cluster] / np.sum(w[cluster, :])

  
        overall_precision = np.mean(precision)

        return accuracy

class TsnePlot:
    def __init__(self, perplexity=30, learning_rate=200, n_iter=1000):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def plot(self, embedding, labels, score, exp_type, experiment_num, epoch, proj_type):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val)/(max_val - min_val)

        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, len(unique_labels)))
        fig, ax = plt.subplots()
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], c=colors[i], label=label, alpha=0.6)
        ax.legend(fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.savefig('EXPERIMENT_{}/{}/tsne/{}_{}_eeg_tsne_plot_kmean_{}.pdf'.format(experiment_num, exp_type, epoch, proj_type, score), bbox_inches='tight')
        plt.close()
        return reduced_embedding

    def plot3d(self, embedding, labels, score, exp_type, experiment_num, epoch, proj_type):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=3, perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        # print(max_val, min_val)
        reduced_embedding = (reduced_embedding - min_val)/(max_val - min_val)

        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, len(unique_labels)))
        # fig, ax = plt.subplots()

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111,projection='3d')
        RADIUS = 5.0  
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], reduced_embedding[mask, 2], c=colors[i], label=label, alpha=0.6)
        ax.legend(fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.savefig('EXPERIMENT_{}/{}/tsne/{}_{}_eeg_tsne3d_plot_kmean_{}.pdf'.format(experiment_num, exp_type, epoch, proj_type, score), bbox_inches='tight')
        plt.close()
        return reduced_embedding



def save_image(spectrogram, gt, experiment_num, epoch, folder_label):

    num_rows = 2
    num_cols = 2

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 10))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i*num_cols + j
            if index < spectrogram.shape[0]:
                # Get the spectrogram and convert it to a numpy array
                spec = np.squeeze(spectrogram[index].numpy(), axis=0)
                # Plot the spectrogram using a heatmap with the 'viridis' color map
                im = axes[i,j].imshow(spec, cmap='viridis', aspect='auto')

                # Set the title and axis labels
                axes[i,j].set_title('EEG {}'.format(index+1))
                axes[i,j].set_xlabel('Time')
                axes[i,j].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('EXPERIMENT_{}/{}/{}_pred.png'.format(experiment_num, folder_label, epoch))


    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 10))
    spectrogram = gt

    for i in range(num_rows):
        for j in range(num_cols):
            index = i*num_cols + j
            if index < spectrogram.shape[0]:
                # Get the spectrogram and convert it to a numpy array
                spec = np.squeeze(spectrogram[index].numpy(), axis=0)

                # Plot the spectrogram using a heatmap with the 'viridis' color map
                im = axes[i,j].imshow(spec, cmap='viridis', aspect='auto')

                # Set the title and axis labels
                axes[i,j].set_title('EEG {}'.format(index+1))
                axes[i,j].set_xlabel('Time')
                axes[i,j].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('EXPERIMENT_{}/{}/{}_gt.png'.format(experiment_num, folder_label, epoch))
    plt.close('all')
    
    




train_eeg_data = torch.load("/home/ubuntu/train_eeg_data.pt")
test_eeg_data = torch.load("/home/ubuntu/test_eeg_data.pt")


train_labels = torch.load("/home/ubuntu/train_labels.pt")
test_labels = torch.load("/home/ubuntu/test_labels.pt")
print(f"Training data shape: {train_eeg_data.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing data shape: {test_eeg_data.shape}")
print(f"Testing labels shape: {test_labels.shape}")

train_dataset = TensorDataset(train_eeg_data, train_labels)
test_dataset = TensorDataset(test_eeg_data, test_labels)

batch_size = 256  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")









device = "cuda"
Sur = FTSurrogate(probability=0.5, phase_noise_magnitude=1).to(device)
mask = SmoothTimeMask(probability=0.5, mask_len_samples=50).to(device)



# a little bit of cleaning along with changing the how things are loaded
def train(epoch, model, optimizer, loss_fn, miner, train_dataloader, experiment_num, device='cuda'):
    model.to(device)
    running_loss = []

    tq = tqdm(train_dataloader)
    

    for batch_idx, (eeg, labels) in enumerate(tq, start=1):
        eeg, labels = eeg.to(device), labels.to(device)
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm
        optimizer.zero_grad()
        eeg = eeg.permute(0, 2, 1)
        

        sur_params = Sur.get_augmentation_params(eeg, labels)
        eeg, labels = Sur.operation(eeg, labels,
                                    phase_noise_magnitude=sur_params['phase_noise_magnitude'],
                                    channel_indep=sur_params['channel_indep'],
                                    random_state = sur_params['random_state'])
        
        mask_params = mask.get_augmentation_params(eeg, labels)
        eeg, labels = mask.operation(eeg, labels,
                                    mask_start_per_sample = mask_params['mask_start_per_sample'],
                                    mask_len_samples = mask_params['mask_len_samples'])        
        
        
        eeg, labels = eeg.to(device), labels.to(device)
        
        x_proj = model(eeg)
        hard_pairs = miner(x_proj, labels)
        loss = loss_fn(x_proj, labels, hard_pairs)

        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        tq.set_description(f'Train:[{epoch}, {np.mean(running_loss):0.3f}]')

    return running_loss

def evaluate_clustering(epoch, model, dataloader, device='cuda'):
    model.eval()
    eeg_featvec_proj, labels_array = [], []

    with torch.no_grad():
        for eeg, labels in tqdm(dataloader):
            norm   = torch.max(eeg) / 2.0
            eeg    = (eeg - norm)/ norm
            eeg, labels = eeg.to(device), labels.to(device)
            eeg = eeg.permute(0, 2, 1)
            x_proj = model(eeg)
            eeg_featvec_proj.append(x_proj.cpu().numpy())
            labels_array.append(labels.cpu().numpy())

    eeg_featvec_proj = np.concatenate(eeg_featvec_proj, axis=0)
    labels_array = np.concatenate(labels_array, axis=0)

    num_clusters = 40
    k_means = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)

    print(f"[Epoch: {epoch}, Train KMeans score Proj: {clustering_acc_proj}]")
    model.train()


def validation(epoch, model, optimizer, loss_fn, miner, val_dataloader, experiment_num):

    running_loss      = []
    eeg_featvec       = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])

    tq = tqdm(val_dataloader)
    model.eval()
    for batch_idx, (eeg, labels) in enumerate(tq, start=1):
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm
        eeg, labels = eeg.to(device), labels.to(device)
        eeg = eeg.permute(0, 2, 1)
        with torch.no_grad():

            x_proj = model(eeg)

            hard_pairs = miner(x_proj, labels)
            loss       = loss_fn(x_proj, labels, hard_pairs)

            running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    num_clusters   = 40
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print("[Epoch: {}, Val KMeans score Proj: {}]".format(epoch, clustering_acc_proj))
    model.train()
    return running_loss, clustering_acc_proj



    

# ## hyperparameters
batch_size     = batch_size
EPOCHS         = 8000
device = "cuda"




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
    )

model     = torch.nn.DataParallel(model).to(device)
optimizer = torch.optim.Adam(list(model.parameters()),lr=3e-4,betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8000, eta_min=0, last_epoch=-1)


# maybe change how things are saved? Like romving EXPERIMENT name :)
dir_info  = natsorted(glob('EXPERIMENT_*'))
if len(dir_info)==0:
    experiment_num = 1
else:
    experiment_num = int(dir_info[-1].split('_')[-1]) + 1
if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
    os.makedirs('EXPERIMENT_{}'.format(experiment_num))
    os.makedirs('EXPERIMENT_{}/val/tsne'.format(experiment_num))


ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/eegfeat_*.pth'.format(experiment_num)))

START_EPOCH = 0
pre = True
if pre:
    ckpt_path  = '/home/ubuntu/EXPERIMENT_187/bestckpt/eegfeat_all_0.5954861111111112.pth'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    START_EPOCH = checkpoint['epoch']
    os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
    os.makedirs('EXPERIMENT_{}/bestckpt/'.format(experiment_num))
    START_EPOCH += 1
else:
    os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
    os.makedirs('EXPERIMENT_{}/bestckpt/'.format(experiment_num))



miner   = miners.MultiSimilarityMiner()
loss_fn = losses.TripletMarginLoss()




best_val_acc   = 0.0
best_val_epoch = 0
EPOCHS = 8000

for epoch in range(START_EPOCH, EPOCHS):

    running_train_loss = train(epoch, model, optimizer, loss_fn, miner, train_loader, experiment_num)
    running_val_loss, val_acc   = validation(epoch, model, optimizer, loss_fn, miner, test_loader, experiment_num)
    scheduler.step()


    if best_val_acc < val_acc:
            best_val_acc   = val_acc
            best_val_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, 'EXPERIMENT_{}/bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, 'all', val_acc))


            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, 'EXPERIMENT_{}/checkpoints/eegfeat_{}.pth'.format(experiment_num, 'all'))






