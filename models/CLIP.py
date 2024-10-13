from transformers import ViTModel
from transformers import ViTFeatureExtractor


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
            ckpt   = '/home/ubuntu/EXPERIMENT_276/bestckpt/eegfeat_all_0.6015625.pth'
            eegcheckpoint = torch.load(ckpt, map_location=device)
            self.model.load_state_dict(eegcheckpoint['model_state_dict'])
          
            
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
        self.img_encoder = ImgEncoder().to(device)  
        self.img_encoder     = torch.nn.DataParallel(self.img_encoder).to(device)
        eegckpt   = image_encoder
        eegcheckpoint = torch.load(eegckpt, map_location=device)
        self.img_encoder.load_state_dict(eegcheckpoint['model_state_dict'])
    
    
        self.eeg_projection = ProjectionHead(embedding_dim=EEG_embedding).to(device)
        self.img_projection = ProjectionHead(embedding_dim=ImgEncoder_dim).to(device)
        self.temperature = temperature

    def forward(self, batch):
        eeg_features = self.eeg_encoder(batch["eeg"]) 
        image_features = self.img_encoder(batch["image"])  
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
    
    
    
if name == "__main__":