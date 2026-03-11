import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embeddings, labels):
        dist_matrix = self._pairwise_distance(embeddings)
        
        label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask = label_eq - torch.eye(len(labels), device=labels.device)
        neg_mask = 1 - label_eq
        
        pos_dist = dist_matrix.clone()
        pos_dist[pos_mask == 0] = -1e-7
        hardest_pos, _ = torch.max(pos_dist, dim=1)
        
        neg_dist = dist_matrix.clone()
        neg_dist[neg_mask == 0] = float('inf')
        hardest_neg, _ = torch.min(neg_dist, dim=1)
        
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

    @staticmethod
    def _pairwise_distance(x):
        return torch.cdist(x, x, p=2)


class ArcFaceLoss(nn.Module):
    
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.mm = self.sin_m * m
        self.threshold = np.cos(np.pi - m)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        cos_theta = F.linear(embeddings, weight)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        sin_theta = torch.sqrt((1.0 - cos_theta * cos_theta).clamp(0, 1))
        
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        cos_theta_m = torch.where(
            cos_theta > self.threshold,
            cos_theta_m,
            cos_theta - self.mm
        )
        
        logits = cos_theta_m * self.s
        
        return F.cross_entropy(logits, labels)


class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        pos_loss = label_matrix * 0.5 * dist_matrix ** 2
        neg_loss = (1 - label_matrix) * 0.5 * F.relu(self.margin - dist_matrix) ** 2
        
        loss = (pos_loss + neg_loss).mean()
        return loss


class MetricLearningTrainer:
    
    def __init__(self, model, device='cuda', learning_rate=1e-3, loss_type='triplet'):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        
        if loss_type == 'triplet':
            self.criterion = TripletLoss(margin=1.0)
        elif loss_type == 'contrastive':
            self.criterion = ContrastiveLoss(margin=1.0)
        elif loss_type == 'arcface':
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.optimizer = None
        
    def prepare_arcface(self, embedding_dim, num_classes):
        self.criterion = ArcFaceLoss(embedding_dim, num_classes, s=64.0, m=0.5).to(self.device)
        self._setup_optimizer()
    
    def _setup_optimizer(self):
        params = list(self.model.parameters())
        if self.criterion is not None and hasattr(self.criterion, 'parameters'):
            params += list(self.criterion.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
    
    def train_epoch(self, embeddings, labels, batch_size=128):
        if self.optimizer is None:
            self._setup_optimizer()
        
        emb_tensor = torch.FloatTensor(embeddings).to(self.device)
        labels_tensor = torch.LongTensor(labels).to(self.device)
        
        dataset = TensorDataset(emb_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        if hasattr(self.criterion, 'train'):
            self.criterion.train()
        
        total_loss = 0
        for batch_emb, batch_labels in loader:
            self.optimizer.zero_grad()
            
            output = self.model(batch_emb)
            loss = self.criterion(output, batch_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(batch_labels)
        
        return total_loss / len(embeddings)
    
    def train(self, embeddings, labels, epochs=10, batch_size=128, val_split=0.1):
        if self.loss_type == 'arcface':
            output_dim = 128
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    output_dim = module.out_features
            self.prepare_arcface(output_dim, len(np.unique(labels)))
        elif self.optimizer is None:
            self._setup_optimizer()
        
        n_val = int(len(embeddings) * val_split)
        indices = np.random.permutation(len(embeddings))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_emb = embeddings[train_indices]
        train_labels = labels[train_indices]
        val_emb = embeddings[val_indices]
        val_labels = labels[val_indices]
        
        print(f"Training {self.loss_type} loss for {epochs} epochs")
        print(f"  Train: {len(train_emb)} samples")
        print(f"  Val: {len(val_emb)} samples")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_emb, train_labels, batch_size)
            
            self.model.eval()
            if hasattr(self.criterion, 'eval'):
                self.criterion.eval()
            
            with torch.no_grad():
                val_emb_tensor = torch.FloatTensor(val_emb).to(self.device)
                val_labels_tensor = torch.LongTensor(val_labels).to(self.device)
                
                val_output = self.model(val_emb_tensor)
                val_loss = self.criterion(val_output, val_labels_tensor).item()
            
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def transform(self, embeddings):
        self.model.eval()
        
        emb_tensor = torch.FloatTensor(embeddings).to(self.device)
        
        with torch.no_grad():
            output = self.model(emb_tensor)
        
        return output.cpu().numpy()

class EmbeddingHead(nn.Module):
    
    def __init__(self, in_dim, out_dim=128, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = in_dim
            
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = torch.eye(batch_size * 2, device=z_i.device).bool()
        negatives = similarity_matrix.masked_fill(mask, -9e15)

        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=z_i.device)
        return F.cross_entropy(logits, labels)

class SimCLRTrainer:
    def __init__(self, model, device='cuda', learning_rate=1e-3, temperature=0.5):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = SimCLRLoss(temperature=temperature)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @staticmethod
    def augment_embeddings(emb, noise_level=0.02, dropout_prob=0.1):
        noise = torch.randn_like(emb) * noise_level
        mask = (torch.rand_like(emb) > dropout_prob).float()
        return (emb + noise) * mask

    def train_epoch(self, embeddings, batch_size=128):
        emb_tensor = torch.FloatTensor(embeddings).to(self.device)
        dataset = TensorDataset(emb_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.model.train()
        total_loss = 0

        for (batch_emb,) in loader:
            self.optimizer.zero_grad()

            z_i = self.model(self.augment_embeddings(batch_emb))
            z_j = self.model(self.augment_embeddings(batch_emb))

            loss = self.criterion(z_i, z_j)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_emb.size(0)

        return total_loss / len(embeddings)

    def train(self, embeddings, epochs=10, batch_size=128, val_split=0.1):
        n_val = int(len(embeddings) * val_split)
        indices = np.random.permutation(len(embeddings))
        val_emb = embeddings[indices[:n_val]]
        train_emb = embeddings[indices[n_val:]]

        print(f"Training SimCLR: {len(train_emb)} train, {len(val_emb)} val")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_emb, batch_size)

            self.model.eval()
            with torch.no_grad():
                v_emb = torch.FloatTensor(val_emb).to(self.device)
                v_z_i = self.model(self.augment_embeddings(v_emb))
                v_z_j = self.model(self.augment_embeddings(v_emb))
                val_loss = self.criterion(v_z_i, v_z_j).item()

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def transform(self, embeddings):
        self.model.eval()
        emb_tensor = torch.FloatTensor(embeddings).to(self.device)
        with torch.no_grad():
            output = self.model(emb_tensor)
        return output.cpu().numpy()