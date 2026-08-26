import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards Negative Log-Likelihood Loss
    """

    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk_pred, event, time):
        # risk_pred: [Batch_Size, 1] - log-hazard predicted
        # event: [Batch_Size] - 1=event/death, 0=censured/alive
        # time: [Batch_Size] - observation period

        risk_pred = risk_pred.squeeze(-1)

        # batch ordered based on decreasing time (to define risk sets)
        sorted_indices = torch.argsort(time, descending=True)
        risk_pred_sorted = risk_pred[sorted_indices]
        event_sorted = event[sorted_indices]

        # Log-sum-exp cumulated on patients with risk (risk set)
        log_risk_cumsum = torch.logcumsumexp(risk_pred_sorted, dim=0)

        # partial likelihood computation considering only patients with Event=1
        uncensored_likelihood = risk_pred_sorted - log_risk_cumsum
        censored_likelihood = uncensored_likelihood * event_sorted

        # avg loss normalized on observed events
        loss = - torch.sum(censored_likelihood) / (torch.sum(event_sorted) + 1e-7)
        return loss


class MultiOmicGATSurvival(nn.Module):
    def __init__(self, in_features, hidden_dim, out_channels, heads=4, dropout=0.2, num_clinical_features=0):
        super(MultiOmicGATSurvival, self).__init__()

        # GATv2Conv solves static attention problem mentioned in the original paper

        # Layer 1: receives node feats (RNA + CNV) and applies multi-head attention
        self.gat1 = GATv2Conv(
            in_channels=in_features,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,  # --> dim = hidden_dim * heads
            add_self_loops=True,
            dropout=dropout,
        )

        # linear projection for residual connection (Skip Connection)
        self.residual_proj = nn.Linear(in_features, hidden_dim * heads)

        # Layer 2: features compression and genes relation consolidation
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            add_self_loops=True,
            dropout=dropout,
        )

        # Norm Layer to stabilize training
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.clinical_encoder = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        if num_clinical_features > 0:
            #fusion_dim = hidden_dim + 32
            fusion_dim = hidden_dim * 2 + 32
            total_emb_dim = fusion_dim
        else:
            #total_emb_dim = hidden_dim
            total_emb_dim = hidden_dim * 2

        # graph classification/prediction
        self.classifier = nn.Sequential(
            nn.Linear(total_emb_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1)  # unitary output: Log-Hazard
        )

        self.dropout_rate = dropout

    def forward(self, x, edge_index, batch=None, clinical_x=None):
        """
        x: Tensor [N_nodes, in_features] -> Genes features matrix
        edge_index: Tensor [2, E] -> Adjacency matrix
        batch: Tensor [N_nodes] -> to combine more graphs, observe them together
        """

        # --- LAYER 1 ---
        h_res = self.residual_proj(x)  # Residual connection
        h = self.gat1(x, edge_index)
        h = h + h_res
        h = self.norm1(h)
        h = F.elu(h)  # ELU activation (GAT standard)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # --- LAYER 2 ---
        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = F.elu(h)

        #graph_emb = global_mean_pool(h, batch)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        graph_emb = torch.cat([mean_pool, max_pool], dim=1)  # [Batch_Size, hidden_dim * 2]

        if not clinical_x is None:
            clin_emb = self.clinical_encoder(clinical_x)
            fused_emb = torch.cat([graph_emb, clin_emb], dim=1) # [Batch_Size, hidden_dim * 2 + 32]
            out = self.classifier(fused_emb)
            return out, fused_emb
        else:
            out = self.classifier(graph_emb)
            return out, graph_emb
