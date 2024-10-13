import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from natsort import natsorted
import os
device = "cuda"
from glob import glob
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning import regularizers
from braindecode.augmentation import FTSurrogate, SmoothTimeMask
import random
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm
    
import pickle
from transformers import ViTModel, ViTConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


from torch.utils.data import TensorDataset, DataLoader, Dataset

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


Sur = FTSurrogate(probability=0.5, phase_noise_magnitude=1).to(device)
mask = SmoothTimeMask(probability=0.5, mask_len_samples=16).to(device)


class K_means:
    def __init__(self, n_clusters=40, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def transform(self, embed, gt_labels):
        pred_labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(embed)
        score       = self.cluster_acc(gt_labels, pred_labels)
        return score

    # Thanks to: https://github.com/k-han/DTC/blob/master/utils/util.py
class K_means:
    def __init__(self, n_clusters=40, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state
    def transform(self, embed, gt_labels):
        pred_labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(embed)
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

        # Overall precision (average across clusters)
        # You can choose other ways to aggregate precision, like weighted average
        overall_precision = np.mean(precision)

        return accuracy
    
    



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


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, in_chan, fine_grained_kernel=11, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dim = int(dim * 0.5)
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout)
            ]))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        dense_feature = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)  # (b, in_chan)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg
        x_dense = torch.cat(dense_feature, dim=-1)  # b, in_chan*depth
        x = x.view(x.size(0), -1)   # b, in_chan*d_hidden_last_layer
        emd = torch.cat((x, x_dense), dim=-1)  # b, in_chan*(depth + d_hidden_last_layer)
        return emd

    def get_info(self, x):
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Deformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(self, *, num_chan, num_time, temporal_kernel, num_kernel=64,
                 num_classes, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.):
        super().__init__()

        self.cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan)

        dim = int(0.5*num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        )

        L = self.get_hidden_size(input_size=dim, num_layer=depth)

        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)

        self.mlp_head = nn.Sequential(
            nn.Linear(out_size, num_classes)
        )

    def forward(self, eeg):
        # eeg: (b, chan, time)
        eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        x += self.pos_embedding
        x = self.transformer(x)
        return self.mlp_head(x)

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EEGDataset(Dataset):
    def __init__(self, eegs, labels):
        self.eegs         = eegs
        self.labels       = labels

    def __getitem__(self, index):
        eeg    = self.eegs[index]
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm
        label  = self.labels[index]
        return eeg, label
    def __len__(self):
        return len(self.eegs)
    
    
    
def train(epoch, model, optimizer, loss_fn, miner, train_data, train_dataloader, experiment_num):

    running_loss      = []
    eeg_featvec_proj  = np.array([])
    labels_array      = np.array([])


    tq = tqdm(train_dataloader)
    for _, (eeg, labels) in enumerate(tq, start=1):
        eeg    = eeg.to(device)
        labels = labels.to(device)
        eeg = eeg.squeeze(1)
        optimizer.zero_grad()
        
        sur_params = Sur.get_augmentation_params(eeg, labels)
        eeg, labels = Sur.operation(eeg, labels,
                                    phase_noise_magnitude=sur_params['phase_noise_magnitude'],
                                    channel_indep=sur_params['channel_indep'],
                                    random_state = sur_params['random_state'])
        

        #eeg = eeg.unsqueeze(1)
        x_proj = model(eeg)
        hard_pairs = miner(x_proj, labels)
        loss       = loss_fn(x_proj, labels, hard_pairs)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Train:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

    if (epoch%5) == 0:
        model.eval()
        for batch_idx, (eeg, labels) in enumerate(tqdm(train_dataloader)):
            eeg, labels = eeg.to(device), labels.to(device)
            eeg = eeg.squeeze(1)
            with torch.no_grad():
                x_proj = model(eeg)
            eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
            labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()
        num_clusters   = 10
        k_means        = K_means(n_clusters=num_clusters)
        clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
        print("[Epoch: {}, Train KMeans score Proj: {}]".format(epoch, clustering_acc_proj))
        model.train()
    return running_loss


def validation(epoch, model, optimizer, loss_fn, miner, train_data, val_dataloader, experiment_num):

    running_loss      = []
    eeg_featvec       = np.array([])
    eeg_featvec_proj  = np.array([])
    eeg_gamma         = np.array([])
    labels_array      = np.array([])

    tq = tqdm(val_dataloader)
    for batch_idx, (eeg, labels) in enumerate(tq, start=1):
        eeg, labels = eeg.to(device), labels.to(device)
        eeg = eeg.squeeze(1)
        model.eval()
        with torch.no_grad():
            x_proj = model(eeg)
            hard_pairs = miner(x_proj, labels)
            loss       = loss_fn(x_proj, labels, hard_pairs)
            running_loss = running_loss + [loss.detach().cpu().numpy()]

        tq.set_description('Val:[{}, {:0.3f}]'.format(epoch, np.mean(running_loss)))

        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, x_proj.cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else x_proj.cpu().detach().numpy()
        labels_array     = np.concatenate((labels_array, labels.cpu().detach().numpy()), axis=0) if labels_array.size else labels.cpu().detach().numpy()

    ### compute k-means score and Umap score on the text and image embeddings
    num_clusters   = 10
    print(eeg_featvec_proj.shape)
    k_means        = K_means(n_clusters=num_clusters)
    clustering_acc_proj = k_means.transform(eeg_featvec_proj, labels_array)
    print("[Epoch: {}, Val KMeans score Proj: {}]".format(epoch, clustering_acc_proj))
    model.train()
    return running_loss, clustering_acc_proj


def compute_fft(data):
    return np.abs(np.fft.fft(data, axis=0))


if __name__ == '__main__':
    
    seed = 45
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    base_path = ""

    with open(base_path + "thoughtviz/eeg/image/data.pkl", 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        train_X = data['x_train']
        train_Y = data['y_train']
        val_X = data['x_test']
        val_Y = data['y_test']
    num_train_samples = train_X.shape[0]
    indices = np.random.choice(num_train_samples, 4000, replace=False)
    selected_train_X = train_X[indices]
    selected_train_Y = train_Y[indices]

    # Remove the selected samples from the training set
    train_X = np.delete(train_X, indices, axis=0)
    train_Y = np.delete(train_Y, indices, axis=0)

    # Add the selected samples to the validation set
    val_X = np.append(val_X, selected_train_X, axis=0)
    val_Y = np.append(val_Y, selected_train_Y, axis=0)

    x_train_eeg = []
    x_train_image = []
    labels = []
    x_train_subject=[]

    # ## hyperparameters
    batch_size     = 256
    EPOCHS         = 2048

    class_labels   = {}
    label_count    = 0
    print(train_X.shape[0])
    for idx in range(train_X.shape[0]):
        eeg_data = np.squeeze(np.transpose(train_X[idx], (2, 0, 1)), axis=0)

        x_train_eeg.append(eeg_data)
        x_train_image.append(np.zeros(shape=(2, 2)))
        x_train_subject.append(0)
        labels.append(np.argmax(train_Y[idx]))

    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    train_labels  = np.array(labels)
    x_train_subject = np.array(x_train_subject)

    print(x_train_eeg.shape, x_train_image.shape, train_labels.shape, x_train_subject.shape)
    print('Total number of classes: {}'.format(len(np.unique(train_labels))), np.unique(train_labels))

    x_train_eeg   = torch.from_numpy(x_train_eeg).float()#.to(device)
    x_train_eeg = x_train_eeg.unsqueeze(1)
    x_train_image = torch.from_numpy(x_train_image).float()#.to(device)
    train_labels  = torch.from_numpy(train_labels).long()#.to(device)
    x_train_subject  = torch.from_numpy(x_train_subject).long()#.to(device)
    
    torch.save(x_train_eeg, "train_data_eeg_thought.pt")
    torch.save(train_labels, "train_data_labels_thought.pt")
    
    train_data       = EEGDataset(x_train_eeg, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)

    
    ## Validation data
    x_val_eeg   = []
    x_val_image = []
    label_val   = []
    x_val_subject = []

    for idx in range(val_X.shape[0]):
        eeg_data = np.squeeze(np.transpose(val_X[idx], (2, 0, 1)), axis=0)

        x_val_eeg.append(eeg_data)
        x_val_image.append(np.zeros(shape=(2, 2)))
        x_val_subject.append(0.0)
        label_val.append(np.argmax(val_Y[idx]))

    x_val_eeg   = np.array(x_val_eeg)
    x_val_image = np.array(x_val_image)
    label_val   = np.array(label_val)
    x_val_subject = np.array(x_val_subject)

    print(x_val_eeg.shape, x_val_image.shape, label_val.shape, x_val_subject.shape)
    print('Total number of classes: {}'.format(len(np.unique(label_val))), np.unique(label_val))

    x_val_eeg   = torch.from_numpy(x_val_eeg).float().to(device)
    x_val_eeg = x_val_eeg.unsqueeze(1)
    x_val_image = torch.from_numpy(x_val_image).float()#.to(device)
    label_val   = torch.from_numpy(label_val).long().to(device)
    x_val_subject  = torch.from_numpy(x_val_subject).long()#.to(device)


    torch.save(x_val_eeg, "test_data_eeg_thought.pt")
    torch.save(label_val, "test_data_labels_thought.pt")
    
    
    val_data       = EEGDataset(x_val_eeg, label_val)
    torch.save( val_data, "val_data.pt")
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)
    
    model = Deformer(
        num_chan=14,
        num_time=32,
        temporal_kernel=1,  # Use an odd number to ensure 'same' padding. The temporal kernel is defined as Odd[0.1*fs], where fs is the sampling rate, and Odd[.] will get the closest odd number.
        num_kernel=8,
        num_classes=256,
        depth=3,
        heads=8,
        mlp_dim=8,
        dim_head=8,
        dropout=0.25
    )
    model     = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(\
                                    list(model.parameters()),\
                                    lr=1e-3,\
                                    betas=(0.9, 0.999)
                                )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=0, last_epoch=-1)

        
    dir_info  = natsorted(glob('EXPERIMENT_*'))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-1]) + 1
    if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
        os.makedirs('EXPERIMENT_{}'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/val/tsne'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/train/tsne/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/test/tsne/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/test/umap/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/finetune_ckpt/'.format(experiment_num))
        os.makedirs('EXPERIMENT_{}/finetune_bestckpt/'.format(experiment_num))
        #os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))

    ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/eegfeat_*.pth'.format(experiment_num)))

    START_EPOCH = 0
    pre = False
    if pre:
        ckpt_path  = '/home/ubuntu/EXPERIMENT_187/bestckpt/eegfeat_all_0.5954861111111112.pth'
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
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
        #running_val_loss, val_acc   = validation(1, model, optimizer, loss_fn, miner, train_data, val_dataloader, experiment_num)

        for epoch in range(START_EPOCH, EPOCHS):
            running_train_loss = train(epoch, model, optimizer, loss_fn, miner, train_data, train_dataloader, experiment_num)
            running_val_loss, val_acc   = validation(epoch, model, optimizer, loss_fn, miner, train_data, val_dataloader, experiment_num)
            scheduler.step()
            if best_val_acc < val_acc:
                best_val_acc   = val_acc
                best_val_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                }, 'EXPERIMENT_{}/bestckpt/eegfeat_{}_{}.pth'.format(experiment_num, 'all', val_acc))

            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                }, 'EXPERIMENT_{}/checkpoints/eegfeat_{}.pth'.format(experiment_num, 'all'))
