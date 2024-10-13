import torch
from torch import nn

import pandas as pd
from einops import rearrange
from einops.layers.torch import Rearrange
import os
import cv2
#import config
from sklearn.decomposition import PCA
from statsmodels.multivariate.manova import MANOVA
from transformers import ViTModel
from transformers import ViTFeatureExtractor

import numpy as np
import zipfile
import PIL.Image
import json
import torch
#import dnnlib
from tqdm import tqdm
from natsort import natsorted
from glob import glob
from torch.utils.data import Dataset
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
        accuracy, precision = self.cluster_metrics(gt_labels, pred_labels)
        return accuracy, precision

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

        return accuracy, overall_precision
class TsnePlot:
    def __init__(self, perplexity=30, learning_rate=200, n_iter=1000):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def plot(self, embedding, labels, score, exp_type, experiment_num, epoch, proj_type):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)

        # Normalize the embedding for a clean plot
        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val) / (max_val - min_val)

        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))  # Using viridis cmap

        fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted size for better label visualization
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], c=colors[i], 
                    label=f'Label {label}', alpha=0.6)

        # Add labels and legend
        #ax.set_title('t-SNE Visualization of EEG Embeddings')

        # Configure the legend to handle many labels
        # ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', 
        #         fancybox=True, shadow=True, ncol=2, fontsize='small')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
        plt.savefig(f'{epoch}_{proj_type}_eeg_tsne_plot_kmean_{score}.pdf', bbox_inches='tight')
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
        RADIUS = 5.0  # Control this value.
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], reduced_embedding[mask, 2], c=colors[i], label=label, alpha=0.6)
        # ax.legend(fancybox=True, shadow=True, ncol=1, bbox_to_anchor=(0.88, 0.5))
        ax.legend(fancybox=True, shadow=True, ncol=1)
        plt.tight_layout()
        plt.savefig('EXPERIMENT_{}/{}/tsne/{}_{}_eeg_tsne3d_plot_kmean_{}.pdf'.format(experiment_num, exp_type, epoch, proj_type, score), bbox_inches='tight')
        plt.close()
        return reduced_embedding



def save_image(spectrogram, gt, experiment_num, epoch, folder_label):
    # Assuming `spectrogram` is the 3D tensor of shape `(440, 33, 9)`
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
    # plt.show()
    plt.savefig('EXPERIMENT_{}/{}/{}_pred.png'.format(experiment_num, folder_label, epoch))

    # plt.clf()

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
        loss =  (images_loss + eeg_loss) / 2.0  # shape: (batch_size)
        return loss.mean()



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
    
    
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device))
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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, patch_dropout = 0.5):
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

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.patch_dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return x
    
    
    
device = "cuda"
class EEG2ImageDataset(Dataset):
    def __init__(self, path, resolution=None, **super_kwargs):
        print(super_kwargs)
        import torch
        from torch import nn

        seed = 45
        torch.manual_seed(seed)
        np.random.seed(seed)

        
        self.eeg_model = SimpleViT(
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
        self.eeg_model = torch.nn.DataParallel(self.eeg_model).to(device)
        eegckpt   = '/home/ubuntu/EXPERIMENT_276/bestckpt/eegfeat_all_0.6015625.pth'
        eegcheckpoint = torch.load(eegckpt, map_location=device)
        self.eeg_model.load_state_dict(eegcheckpoint['model_state_dict'])
        
        
        chkp1 = "/home/ubuntu/model_v1_2_vit_best_model_updated.pth"
        chkp2 = "/home/ubuntu/model_v2_2_vit_best_model_updated.pth"
        chkp3 = "/home/ubuntu/model_v3_3_vit_best_model_updated.pth"
        chkp4 = "/home/ubuntu/model_4_org_vit_best_model_updated.pth"
        
        eeg_model1 = CLIPModel(image_encoder="/home/ubuntu/EXPERIMENT_230/bestckpt/eegfeat_mid_all_0.9748263888888888.pth").to(device)
        eeg_model2 = CLIPModel(image_encoder="/home/ubuntu/EXPERIMENT_264/checkpoints/eegfeat_band_final_all.pth").to(device)
        eeg_model3 = CLIPModel(image_encoder="/home/ubuntu/EXPERIMENT_266/checkpoints/eegfeat_band_final_all.pth").to(device)
        eeg_model4 = CLIPModel(image_encoder="/home/ubuntu/EXPERIMENT_225/bestckpt/eegfeat_all_0.9704861111111112.pth").to(device)
        self.eeg_model.eval()
        
        eegcheckpoint1 = torch.load(chkp1, map_location=device)
        eegcheckpoint2 = torch.load(chkp2, map_location=device)
        eegcheckpoint3 = torch.load(chkp3, map_location=device)
        eegcheckpoint4 = torch.load(chkp4, map_location=device)
        eeg_model1.load_state_dict(eegcheckpoint1)
        eeg_model2.load_state_dict(eegcheckpoint2)
        eeg_model3.load_state_dict(eegcheckpoint3)
        eeg_model4.load_state_dict(eegcheckpoint4)
            ########################################

        print('loading dataset...')

        self.eegs = torch.load("/home/ubuntu/test_eeg_data.pt")
        self.images = torch.load("/home/ubuntu/test_images.pt")
        self.images = (self.images).byte()
        self.labels = torch.load("/home/ubuntu/test_labels.pt")
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(self.eegs, self.images)  
        batch_size = 16  
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        eeg_model1.eval() 
        eeg_model2.eval()  
        eeg_model3.eval()  
        eeg_model4.eval()   
        eeg_features1 = []
        eeg_features2 = []
        eeg_features3 = []
        eeg_features4 = []
        eeg_features5 = []
        from tqdm import tqdm
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing EEG Data"):
                eegs_batch, _ = batch  # Unpack the batch
                eegs_batch = eegs_batch.to(device)  # Move batch to the appropriate device
                
                eegs_batch = eegs_batch.permute(0, 2, 1)
                e6 = self.eeg_model(eegs_batch)
                
                eeg_feat_batch1 = eeg_model1.eeg_encoder(eegs_batch).to(device)
                eeg_feat_batch1 = eeg_model1.eeg_projection(eeg_feat_batch1).to(device)
                
                eeg_feat_batch2 = eeg_model2.eeg_encoder(eegs_batch).to(device)
                eeg_feat_batch2 = eeg_model2.eeg_projection(eeg_feat_batch2).to(device)
                
                eeg_feat_batch3 = eeg_model3.eeg_encoder(eegs_batch).to(device)
                eeg_feat_batch3 = eeg_model3.eeg_projection(eeg_feat_batch3).to(device)
                
                
                eeg_feat_batch4 = eeg_model4.eeg_encoder(eegs_batch).to(device)
                eeg_feat_batch4 = eeg_model4.eeg_projection(eeg_feat_batch4).to(device)
            
                eeg_features1.append(eeg_feat_batch1)
                eeg_features2.append(eeg_feat_batch2)
                eeg_features3.append(eeg_feat_batch3)
                eeg_features4.append(eeg_feat_batch4)
                eeg_features5.append(e6)

        
        eeg_features1 = torch.cat(eeg_features1, dim=0)
        eeg_features2 = torch.cat(eeg_features2, dim=0)     
        eeg_features3 = torch.cat(eeg_features3, dim=0)     
        eeg_features4 = torch.cat(eeg_features4, dim=0)
        eeg_features5 = torch.cat(eeg_features5, dim=0)     
                 
        self.eeg_feat = torch.cat((eeg_features1, eeg_features2, eeg_features3, eeg_features4), dim=1)



        eeg_feat_np = self.eeg_feat.cpu().numpy()
        eeg_feat_1_np = eeg_features1.cpu().numpy()
        eeg_feat_2_np = eeg_features2.cpu().numpy()
        eeg_feat_3_np = eeg_features3.cpu().numpy()
        eeg_feat_4_np = eeg_features4.cpu().numpy()

        # Create a DataFrame with the features and a group label
        n_samples = eeg_feat_1_np.shape[0]  # Assuming all feature sets have the same number of samples

        # Concatenate the feature sets and create a grouping variable
        all_features = np.vstack([eeg_feat_1_np, eeg_feat_2_np, eeg_feat_3_np, eeg_feat_4_np])
        group_labels = np.concatenate([np.full(n_samples, 'Model1'),
                                    np.full(n_samples, 'Model2'),
                                    np.full(n_samples, 'Model3'),
                                    np.full(n_samples, 'Model4')])

        # Convert features to a DataFrame and rename columns to strings
        df = pd.DataFrame(all_features)
        df['Group'] = group_labels
        df.columns = [f'Feature_{i}' for i in range(df.shape[1] - 1)] + ['Group']  # Rename feature columns to strings

        # Run MANOVA
        manova = MANOVA.from_formula(' + '.join(df.columns[:-1]) + ' ~ Group', data=df)
        print(manova.mv_test())
        k_means             = K_means(n_clusters=40)
        clustering_acc_proj_1 = k_means.transform(np.array(self.eeg_feat.cpu()), np.array(self.labels.cpu()))
        #print("[KMeans score Proj: {}]".format(clustering_acc_proj))

        clustering_acc_proj_2 = k_means.transform(np.array(eeg_features1.cpu()), np.array(self.labels.cpu()))
        #print("[KMeans score Proj: {}]".format(clustering_acc_proj))
        
        clustering_acc_proj_3 = k_means.transform(np.array(eeg_features2.cpu()), np.array(self.labels.cpu()))
        #print("[KMeans score Proj: {}]".format(clustering_acc_proj))
        
        clustering_acc_proj_4 = k_means.transform(np.array(eeg_features3.cpu()), np.array(self.labels.cpu()))
        #print("[KMeans score Proj: {}]".format(clustering_acc_proj))
        
        clustering_acc_proj_5 = k_means.transform(np.array(eeg_features4.cpu()), np.array(self.labels.cpu()))
        #print("[KMeans score Proj: {}]".format(clustering_acc_proj))                
        
        clustering_acc_proj_6 = k_means.transform(np.array(eeg_features5.cpu()), np.array(self.labels.cpu()))
        #print("[KMeans score Proj: {}]".format(clustering_acc_proj))
        
        
        tsne_plotter = TsnePlot(perplexity=30, learning_rate=200, n_iter=1000)
        labels = self.labels
        # Perform t-SNE and plot for each set of EEG features
        eeg_feat_np = eeg_features1.cpu().numpy()
        tsne_plotter.plot(eeg_feat_np, labels.cpu().numpy(), clustering_acc_proj_2, 'exp1', 1, 10, 'tsne1')

        eeg_feat_np = eeg_features2.cpu().numpy()
        tsne_plotter.plot(eeg_feat_np, labels.cpu().numpy(), clustering_acc_proj_3, 'exp2', 1, 10, 'tsne2')

        eeg_feat_np = eeg_features3.cpu().numpy()
        tsne_plotter.plot(eeg_feat_np, labels.cpu().numpy(), clustering_acc_proj_4, 'exp3', 1, 10, 'tsne3')

        eeg_feat_np = eeg_features4.cpu().numpy()
        tsne_plotter.plot(eeg_feat_np, labels.cpu().numpy(), clustering_acc_proj_5, 'exp4', 1, 10, 'tsne4')

        eeg_feat_np = eeg_features5.cpu().numpy()
        tsne_plotter.plot(eeg_feat_np, labels.cpu().numpy(), clustering_acc_proj_6, 'exp5', 1, 10, 'tsne5')

        


EEG2ImageDataset("ss")