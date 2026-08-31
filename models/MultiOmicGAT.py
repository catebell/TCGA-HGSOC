import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, BatchNorm


# GATv2Conv: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html
# solves static attention problem mentioned in the original paper

class MultiOmicGAT(nn.Module):
    def __init__(self, in_node_features, hidden_dim, out_channels, heads=4, dropout=0.2, num_clinical_features=0):
        super(MultiOmicGAT, self).__init__()

        # Layer 1: receives node feats (RNA + CNV) and applies multi-head attention
        self.gat1 = GATv2Conv(
            in_channels=in_node_features,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,  # --> dim = hidden_dim * heads
            add_self_loops=True,
            dropout=dropout,
        )

        # linear projection for residual connection (Skip Connection)
        self.residual_proj1 = nn.Linear(in_node_features, hidden_dim * heads)
        self.residual_proj2 = nn.Linear(hidden_dim * heads, hidden_dim)

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
        #self.norm1 = nn.LayerNorm(hidden_dim * heads)
        #self.norm2 = nn.LayerNorm(hidden_dim)

        self.bn1 = BatchNorm(hidden_dim * heads)
        self.bn2 = BatchNorm(hidden_dim)

        self.num_clinical_features = num_clinical_features

        if num_clinical_features > 0:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(num_clinical_features, 32),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )

            fusion_dim = hidden_dim * 2 + 32
            total_emb_dim = fusion_dim
        else:
            total_emb_dim = hidden_dim * 2

        # graph classification/prediction
        self.classifier = nn.Sequential(
            nn.Linear(total_emb_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, out_channels)
        )

        self.dropout_rate = dropout

    def forward(self, x, edge_index, batch=None, clinical_x=None):
        """
        x: Tensor [N_nodes, in_node_features] -> Genes features matrix
        edge_index: Tensor [2, E] -> Adjacency matrix
        batch: Tensor [N_nodes] -> to combine more graphs, observe them together
        """

        # --- LAYER 1 ---
        h_res = self.residual_proj1(x)  # Residual connection
        h = self.gat1(x, edge_index)
        h = self.bn1(h)
        h = h + h_res  # TODO maybe invertire questo e normalization?
        h = F.elu(h)  # ELU activation (GAT standard)
        #h = F.dropout(h, p=self.dropout_rate, training=self.training)  # TODO maybe togliere?

        # --- LAYER 2 ---
        h_res = self.residual_proj2(h)
        h = self.gat2(h, edge_index)
        h = self.bn2(h)  # add x = x + x_projected
        h = h + h_res  # TODO maybe invertire questo e normalization?
        h = F.elu(h)

        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        graph_emb = torch.cat([mean_pool, max_pool], dim=1)  # [Batch_Size, hidden_dim * 2]

        # todo maybe add x = self.dropout(x)
        graph_emb = F.dropout(graph_emb, p=self.dropout_rate, training=self.training)

        if self.num_clinical_features > 0 and clinical_x is not None:
            clin_emb = self.clinical_encoder(clinical_x)
            fused_emb = torch.cat([graph_emb, clin_emb], dim=1) # [Batch_Size, hidden_dim * 2 + 32]
            out = self.classifier(fused_emb)
            return out, fused_emb
        else:
            out = self.classifier(graph_emb)
            return out, graph_emb



# RIS con config attuale
"""

2026-08-28 20:28:40,209 - INFO - Starting training using target feature: disease_code

2026-08-28 20:28:40,222 - INFO - Train + Val Dataset init...
2026-08-28 20:28:40,240 - INFO - Test Dataset init...
2026-08-28 20:28:40,457 - INFO - Clinical Features found : 3
 ['age_at_initial_pathologic_diagnosis', 'days_to_last_followup', 'postoperative_rx_tx']
2026-08-28 20:29:10,686 - INFO - Unique labels found in dataset: tensor([0, 1])
2026-08-28 20:29:10,835 - INFO - classes_counts:
2026-08-28 20:29:10,835 - INFO - tensor([368, 378])
2026-08-28 20:29:10,835 - INFO - weights:
2026-08-28 20:29:10,835 - INFO - tensor([1.0136, 0.9868], device='cuda:0')
2026-08-28 20:29:10,935 - INFO -
--- FOLD 1 ---

2026-08-28 20:29:17,695 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 64, heads=4)
  (residual_proj1): Linear(in_features=10, out_features=256, bias=True)
  (residual_proj2): Linear(in_features=256, out_features=64, bias=True)
  (gat2): GATv2Conv(256, 64, heads=1)
  (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=2, bias=True)
  )
)

  0%|          | 0/100 [00:00<?, ?it/s]2026-08-28 20:29:26,589 - INFO - Epoch 01 | Train Loss: 0.7822, Train Acc: 0.5487 | Val Loss: 0.6617
2026-08-28 20:29:26,589 - INFO - Val metrics: Acc = 0.6333 | F1 = 0.6260, AUC = 0.6947

2026-08-28 20:29:26,589 - INFO - Loss IMPROVEMENT!

  1%|          | 1/100 [00:08<14:45,  8.94s/it]2026-08-28 20:29:35,158 - INFO - Epoch 02 | Train Loss: 0.7178, Train Acc: 0.5503 | Val Loss: 0.6715
2026-08-28 20:29:35,159 - INFO - Val metrics: Acc = 0.6267 | F1 = 0.6243, AUC = 0.6895

  2%|▏         | 2/100 [00:17<14:12,  8.69s/it]2026-08-28 20:29:43,670 - INFO - Epoch 03 | Train Loss: 0.6763, Train Acc: 0.6225 | Val Loss: 0.6623
2026-08-28 20:29:43,670 - INFO - Val metrics: Acc = 0.6200 | F1 = 0.6192, AUC = 0.6945

  3%|▎         | 3/100 [00:25<13:55,  8.61s/it]2026-08-28 20:29:52,236 - INFO - Epoch 04 | Train Loss: 0.6964, Train Acc: 0.5487 | Val Loss: 0.6644
2026-08-28 20:29:52,236 - INFO - Val metrics: Acc = 0.6133 | F1 = 0.6048, AUC = 0.7045

  4%|▍         | 4/100 [00:34<13:44,  8.59s/it]2026-08-28 20:30:00,829 - INFO - Epoch 05 | Train Loss: 0.6767, Train Acc: 0.5973 | Val Loss: 0.6595
2026-08-28 20:30:00,830 - INFO - Val metrics: Acc = 0.6467 | F1 = 0.6463, AUC = 0.7087

2026-08-28 20:30:00,830 - INFO - Loss IMPROVEMENT!

  5%|▌         | 5/100 [00:43<13:36,  8.60s/it]2026-08-28 20:30:09,329 - INFO - Epoch 06 | Train Loss: 0.6810, Train Acc: 0.5772 | Val Loss: 0.6565
2026-08-28 20:30:09,329 - INFO - Val metrics: Acc = 0.6133 | F1 = 0.6133, AUC = 0.7004

2026-08-28 20:30:09,329 - INFO - Loss IMPROVEMENT!

  6%|▌         | 6/100 [00:51<13:24,  8.56s/it]2026-08-28 20:30:17,912 - INFO - Epoch 07 | Train Loss: 0.6610, Train Acc: 0.5940 | Val Loss: 0.6498
2026-08-28 20:30:17,912 - INFO - Val metrics: Acc = 0.6133 | F1 = 0.6063, AUC = 0.7178

2026-08-28 20:30:17,912 - INFO - Loss IMPROVEMENT!

  7%|▋         | 7/100 [01:00<13:16,  8.57s/it]2026-08-28 20:30:26,477 - INFO - Epoch 08 | Train Loss: 0.6650, Train Acc: 0.5973 | Val Loss: 0.6497
2026-08-28 20:30:26,478 - INFO - Val metrics: Acc = 0.6267 | F1 = 0.6261, AUC = 0.7159

2026-08-28 20:30:26,478 - INFO - Loss IMPROVEMENT!

  8%|▊         | 8/100 [01:08<13:08,  8.57s/it]2026-08-28 20:30:35,007 - INFO - Epoch 09 | Train Loss: 0.6470, Train Acc: 0.6275 | Val Loss: 0.6314
2026-08-28 20:30:35,007 - INFO - Val metrics: Acc = 0.6333 | F1 = 0.6286, AUC = 0.7192

2026-08-28 20:30:35,007 - INFO - Loss IMPROVEMENT!

  9%|▉         | 9/100 [01:17<12:58,  8.56s/it]2026-08-28 20:30:43,557 - INFO - Epoch 10 | Train Loss: 0.6611, Train Acc: 0.6393 | Val Loss: 0.6310
2026-08-28 20:30:43,557 - INFO - Val metrics: Acc = 0.6400 | F1 = 0.6394, AUC = 0.7217

2026-08-28 20:30:43,557 - INFO - Loss IMPROVEMENT!

 10%|█         | 10/100 [01:25<12:49,  8.55s/it]2026-08-28 20:30:52,005 - INFO - Epoch 11 | Train Loss: 0.6612, Train Acc: 0.6074 | Val Loss: 0.6509
2026-08-28 20:30:52,005 - INFO - Val metrics: Acc = 0.6800 | F1 = 0.6701, AUC = 0.7159

 11%|█         | 11/100 [01:34<12:38,  8.52s/it]2026-08-28 20:31:00,662 - INFO - Epoch 12 | Train Loss: 0.6650, Train Acc: 0.5940 | Val Loss: 0.6392
2026-08-28 20:31:00,662 - INFO - Val metrics: Acc = 0.6400 | F1 = 0.6394, AUC = 0.7207

 12%|█▏        | 12/100 [01:42<12:33,  8.56s/it]2026-08-28 20:31:09,161 - INFO - Epoch 13 | Train Loss: 0.6599, Train Acc: 0.5956 | Val Loss: 0.6436
2026-08-28 20:31:09,161 - INFO - Val metrics: Acc = 0.6933 | F1 = 0.6787, AUC = 0.7240

 13%|█▎        | 13/100 [01:51<12:23,  8.54s/it]2026-08-28 20:31:17,691 - INFO - Epoch 14 | Train Loss: 0.6542, Train Acc: 0.6493 | Val Loss: 0.6333
2026-08-28 20:31:17,691 - INFO - Val metrics: Acc = 0.6733 | F1 = 0.6606, AUC = 0.7356

 14%|█▍        | 14/100 [01:59<12:14,  8.54s/it]2026-08-28 20:31:26,274 - INFO - Epoch 15 | Train Loss: 0.6405, Train Acc: 0.6527 | Val Loss: 0.6224
2026-08-28 20:31:26,274 - INFO - Val metrics: Acc = 0.6800 | F1 = 0.6753, AUC = 0.7333

2026-08-28 20:31:26,274 - INFO - Loss IMPROVEMENT!

 15%|█▌        | 15/100 [02:08<12:07,  8.55s/it]2026-08-28 20:31:34,952 - INFO - Epoch 16 | Train Loss: 0.6254, Train Acc: 0.6644 | Val Loss: 0.6080
2026-08-28 20:31:34,952 - INFO - Val metrics: Acc = 0.6800 | F1 = 0.6779, AUC = 0.7480

2026-08-28 20:31:34,952 - INFO - Loss IMPROVEMENT!

 16%|█▌        | 16/100 [02:17<12:01,  8.59s/it]2026-08-28 20:31:43,544 - INFO - Epoch 17 | Train Loss: 0.6359, Train Acc: 0.6594 | Val Loss: 0.6245
2026-08-28 20:31:43,544 - INFO - Val metrics: Acc = 0.6733 | F1 = 0.6730, AUC = 0.7210

 17%|█▋        | 17/100 [02:25<11:52,  8.59s/it]2026-08-28 20:31:52,103 - INFO - Epoch 18 | Train Loss: 0.6309, Train Acc: 0.6409 | Val Loss: 0.6174
2026-08-28 20:31:52,103 - INFO - Val metrics: Acc = 0.6733 | F1 = 0.6691, AUC = 0.7390

 18%|█▊        | 18/100 [02:34<11:43,  8.58s/it]2026-08-28 20:32:00,768 - INFO - Epoch 19 | Train Loss: 0.6382, Train Acc: 0.6275 | Val Loss: 0.6164
2026-08-28 20:32:00,768 - INFO - Val metrics: Acc = 0.6933 | F1 = 0.6920, AUC = 0.7464

 19%|█▉        | 19/100 [02:43<11:37,  8.61s/it]2026-08-28 20:32:09,396 - INFO - Epoch 20 | Train Loss: 0.6248, Train Acc: 0.6577 | Val Loss: 0.6145
2026-08-28 20:32:09,396 - INFO - Val metrics: Acc = 0.6733 | F1 = 0.6722, AUC = 0.7393

 20%|██        | 20/100 [02:51<11:28,  8.61s/it]2026-08-28 20:32:17,990 - INFO - Epoch 21 | Train Loss: 0.6165, Train Acc: 0.6695 | Val Loss: 0.6109
2026-08-28 20:32:17,990 - INFO - Val metrics: Acc = 0.7067 | F1 = 0.7048, AUC = 0.7504

 21%|██        | 21/100 [03:00<11:19,  8.61s/it]2026-08-28 20:32:26,616 - INFO - Epoch 22 | Train Loss: 0.6172, Train Acc: 0.6728 | Val Loss: 0.6181
2026-08-28 20:32:26,616 - INFO - Val metrics: Acc = 0.6333 | F1 = 0.6147, AUC = 0.7456

 22%|██▏       | 22/100 [03:08<11:11,  8.61s/it]2026-08-28 20:32:35,263 - INFO - Epoch 23 | Train Loss: 0.6393, Train Acc: 0.6762 | Val Loss: 0.6269
2026-08-28 20:32:35,263 - INFO - Val metrics: Acc = 0.7267 | F1 = 0.7267, AUC = 0.7426

 23%|██▎       | 23/100 [03:17<11:03,  8.62s/it]2026-08-28 20:32:43,976 - INFO - Epoch 24 | Train Loss: 0.6214, Train Acc: 0.6594 | Val Loss: 0.6234
2026-08-28 20:32:43,976 - INFO - Val metrics: Acc = 0.6200 | F1 = 0.6030, AUC = 0.7365

 24%|██▍       | 24/100 [03:26<10:57,  8.65s/it]2026-08-28 20:32:52,540 - INFO - Epoch 25 | Train Loss: 0.6003, Train Acc: 0.6812 | Val Loss: 0.5881
2026-08-28 20:32:52,540 - INFO - Val metrics: Acc = 0.7000 | F1 = 0.6999, AUC = 0.7630

2026-08-28 20:32:52,540 - INFO - Loss IMPROVEMENT!

 25%|██▌       | 25/100 [03:34<10:46,  8.63s/it]2026-08-28 20:33:01,154 - INFO - Epoch 26 | Train Loss: 0.6070, Train Acc: 0.6812 | Val Loss: 0.6013
2026-08-28 20:33:01,155 - INFO - Val metrics: Acc = 0.7067 | F1 = 0.7062, AUC = 0.7687

 26%|██▌       | 26/100 [03:43<10:37,  8.62s/it]2026-08-28 20:33:09,738 - INFO - Epoch 27 | Train Loss: 0.6181, Train Acc: 0.6611 | Val Loss: 0.5952
2026-08-28 20:33:09,738 - INFO - Val metrics: Acc = 0.7133 | F1 = 0.7133, AUC = 0.7922

 27%|██▋       | 27/100 [03:52<10:28,  8.61s/it]2026-08-28 20:33:18,279 - INFO - Epoch 28 | Train Loss: 0.6009, Train Acc: 0.7097 | Val Loss: 0.5759
2026-08-28 20:33:18,279 - INFO - Val metrics: Acc = 0.7200 | F1 = 0.7138, AUC = 0.7957

2026-08-28 20:33:18,280 - INFO - Loss IMPROVEMENT!

 28%|██▊       | 28/100 [04:00<10:18,  8.59s/it]2026-08-28 20:33:26,866 - INFO - Epoch 29 | Train Loss: 0.5881, Train Acc: 0.6930 | Val Loss: 0.5795
2026-08-28 20:33:26,866 - INFO - Val metrics: Acc = 0.7200 | F1 = 0.7198, AUC = 0.7884

 29%|██▉       | 29/100 [04:09<10:09,  8.59s/it]2026-08-28 20:33:35,429 - INFO - Epoch 30 | Train Loss: 0.5820, Train Acc: 0.6795 | Val Loss: 0.5654
2026-08-28 20:33:35,429 - INFO - Val metrics: Acc = 0.7200 | F1 = 0.7182, AUC = 0.8000

2026-08-28 20:33:35,429 - INFO - Loss IMPROVEMENT!

 30%|███       | 30/100 [04:17<10:00,  8.58s/it]2026-08-28 20:33:44,155 - INFO - Epoch 31 | Train Loss: 0.5534, Train Acc: 0.7265 | Val Loss: 0.5783
2026-08-28 20:33:44,155 - INFO - Val metrics: Acc = 0.6933 | F1 = 0.6878, AUC = 0.7948

 31%|███       | 31/100 [04:26<09:55,  8.62s/it]2026-08-28 20:33:52,707 - INFO - Epoch 32 | Train Loss: 0.5876, Train Acc: 0.6812 | Val Loss: 0.6107
2026-08-28 20:33:52,708 - INFO - Val metrics: Acc = 0.6933 | F1 = 0.6645, AUC = 0.8103

 32%|███▏      | 32/100 [04:35<09:44,  8.60s/it]2026-08-28 20:34:01,297 - INFO - Epoch 33 | Train Loss: 0.6126, Train Acc: 0.6611 | Val Loss: 0.5636
2026-08-28 20:34:01,297 - INFO - Val metrics: Acc = 0.7800 | F1 = 0.7792, AUC = 0.8460

2026-08-28 20:34:01,297 - INFO - Loss IMPROVEMENT!

 33%|███▎      | 33/100 [04:43<09:36,  8.60s/it]2026-08-28 20:34:09,836 - INFO - Epoch 34 | Train Loss: 0.5383, Train Acc: 0.7164 | Val Loss: 0.5475
2026-08-28 20:34:09,836 - INFO - Val metrics: Acc = 0.7400 | F1 = 0.7380, AUC = 0.8218

2026-08-28 20:34:09,836 - INFO - Loss IMPROVEMENT!

 34%|███▍      | 34/100 [04:52<09:26,  8.58s/it]2026-08-28 20:34:18,435 - INFO - Epoch 35 | Train Loss: 0.5409, Train Acc: 0.7366 | Val Loss: 0.5652
2026-08-28 20:34:18,435 - INFO - Val metrics: Acc = 0.7733 | F1 = 0.7700, AUC = 0.8249

 35%|███▌      | 35/100 [05:00<09:18,  8.58s/it]2026-08-28 20:34:27,207 - INFO - Epoch 36 | Train Loss: 0.5481, Train Acc: 0.7215 | Val Loss: 0.5239
2026-08-28 20:34:27,207 - INFO - Val metrics: Acc = 0.7600 | F1 = 0.7557, AUC = 0.8549

2026-08-28 20:34:27,207 - INFO - Loss IMPROVEMENT!

 36%|███▌      | 36/100 [05:09<09:13,  8.64s/it]2026-08-28 20:34:35,774 - INFO - Epoch 37 | Train Loss: 0.5401, Train Acc: 0.7198 | Val Loss: 0.5879
2026-08-28 20:34:35,774 - INFO - Val metrics: Acc = 0.6600 | F1 = 0.6404, AUC = 0.8318

 37%|███▋      | 37/100 [05:18<09:02,  8.62s/it]2026-08-28 20:34:44,316 - INFO - Epoch 38 | Train Loss: 0.5246, Train Acc: 0.7500 | Val Loss: 0.5292
2026-08-28 20:34:44,316 - INFO - Val metrics: Acc = 0.7667 | F1 = 0.7664, AUC = 0.8409

 38%|███▊      | 38/100 [05:26<08:52,  8.60s/it]2026-08-28 20:34:52,973 - INFO - Epoch 39 | Train Loss: 0.4881, Train Acc: 0.7517 | Val Loss: 0.5376
2026-08-28 20:34:52,973 - INFO - Val metrics: Acc = 0.7600 | F1 = 0.7589, AUC = 0.8384

 39%|███▉      | 39/100 [05:35<08:45,  8.61s/it]2026-08-28 20:35:01,662 - INFO - Epoch 40 | Train Loss: 0.4938, Train Acc: 0.7617 | Val Loss: 0.4871
2026-08-28 20:35:01,662 - INFO - Val metrics: Acc = 0.7867 | F1 = 0.7811, AUC = 0.9017

2026-08-28 20:35:01,662 - INFO - Loss IMPROVEMENT!

 40%|████      | 40/100 [05:43<08:38,  8.64s/it]2026-08-28 20:35:10,289 - INFO - Epoch 41 | Train Loss: 0.4974, Train Acc: 0.7718 | Val Loss: 0.5349
2026-08-28 20:35:10,289 - INFO - Val metrics: Acc = 0.7733 | F1 = 0.7684, AUC = 0.8778

 41%|████      | 41/100 [05:52<08:29,  8.63s/it]2026-08-28 20:35:18,857 - INFO - Epoch 42 | Train Loss: 0.4851, Train Acc: 0.7685 | Val Loss: 0.4667
2026-08-28 20:35:18,857 - INFO - Val metrics: Acc = 0.8067 | F1 = 0.8065, AUC = 0.8896

2026-08-28 20:35:18,857 - INFO - Loss IMPROVEMENT!

 42%|████▏     | 42/100 [06:01<08:19,  8.62s/it]2026-08-28 20:35:27,449 - INFO - Epoch 43 | Train Loss: 0.4998, Train Acc: 0.7550 | Val Loss: 0.5105
2026-08-28 20:35:27,449 - INFO - Val metrics: Acc = 0.8467 | F1 = 0.8466, AUC = 0.8969

 43%|████▎     | 43/100 [06:09<08:10,  8.61s/it]2026-08-28 20:35:36,024 - INFO - Epoch 44 | Train Loss: 0.4834, Train Acc: 0.7634 | Val Loss: 0.5327
2026-08-28 20:35:36,024 - INFO - Val metrics: Acc = 0.7133 | F1 = 0.6948, AUC = 0.8930

 44%|████▍     | 44/100 [06:18<08:01,  8.60s/it]2026-08-28 20:35:44,503 - INFO - Epoch 45 | Train Loss: 0.4701, Train Acc: 0.7752 | Val Loss: 0.4950
2026-08-28 20:35:44,503 - INFO - Val metrics: Acc = 0.7867 | F1 = 0.7778, AUC = 0.9403

 45%|████▌     | 45/100 [06:26<07:50,  8.56s/it]2026-08-28 20:35:53,006 - INFO - Epoch 46 | Train Loss: 0.4697, Train Acc: 0.7852 | Val Loss: 0.4598
2026-08-28 20:35:53,007 - INFO - Val metrics: Acc = 0.7867 | F1 = 0.7801, AUC = 0.9346

2026-08-28 20:35:53,007 - INFO - Loss IMPROVEMENT!

 46%|████▌     | 46/100 [06:35<07:41,  8.55s/it]2026-08-28 20:36:01,633 - INFO - Epoch 47 | Train Loss: 0.4711, Train Acc: 0.7785 | Val Loss: 0.5680
2026-08-28 20:36:01,633 - INFO - Val metrics: Acc = 0.6067 | F1 = 0.5347, AUC = 0.9171

 47%|████▋     | 47/100 [06:43<07:34,  8.57s/it]2026-08-28 20:36:10,086 - INFO - Epoch 48 | Train Loss: 0.4622, Train Acc: 0.7802 | Val Loss: 0.4317
2026-08-28 20:36:10,086 - INFO - Val metrics: Acc = 0.8533 | F1 = 0.8520, AUC = 0.9321

2026-08-28 20:36:10,086 - INFO - Loss IMPROVEMENT!

 48%|████▊     | 48/100 [06:52<07:23,  8.54s/it]2026-08-28 20:36:18,599 - INFO - Epoch 49 | Train Loss: 0.4273, Train Acc: 0.7936 | Val Loss: 0.5217
2026-08-28 20:36:18,600 - INFO - Val metrics: Acc = 0.6867 | F1 = 0.6557, AUC = 0.9282

 49%|████▉     | 49/100 [07:00<07:14,  8.53s/it]2026-08-28 20:36:27,068 - INFO - Epoch 50 | Train Loss: 0.4341, Train Acc: 0.7936 | Val Loss: 0.4688
2026-08-28 20:36:27,069 - INFO - Val metrics: Acc = 0.8333 | F1 = 0.8331, AUC = 0.8995

 50%|█████     | 50/100 [07:09<07:05,  8.51s/it]2026-08-28 20:36:35,452 - INFO - Epoch 51 | Train Loss: 0.4125, Train Acc: 0.8121 | Val Loss: 0.5183
2026-08-28 20:36:35,452 - INFO - Val metrics: Acc = 0.7200 | F1 = 0.6986, AUC = 0.9106

 51%|█████     | 51/100 [07:17<06:55,  8.47s/it]2026-08-28 20:36:43,877 - INFO - Epoch 52 | Train Loss: 0.4121, Train Acc: 0.8289 | Val Loss: 0.4503
2026-08-28 20:36:43,877 - INFO - Val metrics: Acc = 0.7533 | F1 = 0.7423, AUC = 0.9335

 52%|█████▏    | 52/100 [07:26<06:45,  8.46s/it]2026-08-28 20:36:52,230 - INFO - Epoch 53 | Train Loss: 0.4546, Train Acc: 0.8070 | Val Loss: 0.4948
2026-08-28 20:36:52,230 - INFO - Val metrics: Acc = 0.8400 | F1 = 0.8390, AUC = 0.9134

 53%|█████▎    | 53/100 [07:34<06:36,  8.43s/it]2026-08-28 20:37:00,671 - INFO - Epoch 54 | Train Loss: 0.4026, Train Acc: 0.8188 | Val Loss: 0.3983
2026-08-28 20:37:00,671 - INFO - Val metrics: Acc = 0.8600 | F1 = 0.8592, AUC = 0.9472

2026-08-28 20:37:00,671 - INFO - Loss IMPROVEMENT!

 54%|█████▍    | 54/100 [07:42<06:27,  8.43s/it]2026-08-28 20:37:09,077 - INFO - Epoch 55 | Train Loss: 0.4488, Train Acc: 0.7953 | Val Loss: 0.5093
2026-08-28 20:37:09,077 - INFO - Val metrics: Acc = 0.7867 | F1 = 0.7801, AUC = 0.8972

 55%|█████▌    | 55/100 [07:51<06:19,  8.42s/it]2026-08-28 20:37:17,455 - INFO - Epoch 56 | Train Loss: 0.4635, Train Acc: 0.7919 | Val Loss: 0.4690
2026-08-28 20:37:17,455 - INFO - Val metrics: Acc = 0.8067 | F1 = 0.8002, AUC = 0.9445

 56%|█████▌    | 56/100 [07:59<06:10,  8.41s/it]2026-08-28 20:37:25,842 - INFO - Epoch 57 | Train Loss: 0.3799, Train Acc: 0.8372 | Val Loss: 0.5077
2026-08-28 20:37:25,842 - INFO - Val metrics: Acc = 0.7067 | F1 = 0.6817, AUC = 0.9335

 57%|█████▋    | 57/100 [08:08<06:01,  8.40s/it]2026-08-28 20:37:34,361 - INFO - Epoch 58 | Train Loss: 0.4060, Train Acc: 0.8238 | Val Loss: 0.3809
2026-08-28 20:37:34,362 - INFO - Val metrics: Acc = 0.8667 | F1 = 0.8647, AUC = 0.9612

2026-08-28 20:37:34,362 - INFO - Loss IMPROVEMENT!

 58%|█████▊    | 58/100 [08:16<05:54,  8.44s/it]2026-08-28 20:37:42,853 - INFO - Epoch 59 | Train Loss: 0.4030, Train Acc: 0.8121 | Val Loss: 0.5873
2026-08-28 20:37:42,853 - INFO - Val metrics: Acc = 0.6133 | F1 = 0.5454, AUC = 0.9491

 59%|█████▉    | 59/100 [08:25<05:46,  8.45s/it]2026-08-28 20:37:51,270 - INFO - Epoch 60 | Train Loss: 0.4128, Train Acc: 0.8221 | Val Loss: 0.5086
2026-08-28 20:37:51,270 - INFO - Val metrics: Acc = 0.6667 | F1 = 0.6250, AUC = 0.9611

 60%|██████    | 60/100 [08:33<05:37,  8.44s/it]2026-08-28 20:37:59,843 - INFO - Epoch 61 | Train Loss: 0.3670, Train Acc: 0.8406 | Val Loss: 0.4145
2026-08-28 20:37:59,843 - INFO - Val metrics: Acc = 0.8600 | F1 = 0.8599, AUC = 0.9424

 61%|██████    | 61/100 [08:42<05:30,  8.48s/it]2026-08-28 20:38:08,489 - INFO - Epoch 62 | Train Loss: 0.3805, Train Acc: 0.8121 | Val Loss: 0.4578
2026-08-28 20:38:08,489 - INFO - Val metrics: Acc = 0.7800 | F1 = 0.7688, AUC = 0.9575

 62%|██████▏   | 62/100 [08:50<05:24,  8.53s/it]2026-08-28 20:38:17,091 - INFO - Epoch 63 | Train Loss: 0.3544, Train Acc: 0.8456 | Val Loss: 0.4621
2026-08-28 20:38:17,091 - INFO - Val metrics: Acc = 0.8600 | F1 = 0.8586, AUC = 0.9404

 63%|██████▎   | 63/100 [08:59<05:16,  8.55s/it]2026-08-28 20:38:25,715 - INFO - Epoch 64 | Train Loss: 0.3514, Train Acc: 0.8406 | Val Loss: 0.5367
2026-08-28 20:38:25,715 - INFO - Val metrics: Acc = 0.6667 | F1 = 0.6250, AUC = 0.9529

 64%|██████▍   | 64/100 [09:08<05:08,  8.57s/it]2026-08-28 20:38:34,278 - INFO - Epoch 65 | Train Loss: 0.3836, Train Acc: 0.8188 | Val Loss: 0.5669
2026-08-28 20:38:34,278 - INFO - Val metrics: Acc = 0.6467 | F1 = 0.5918, AUC = 0.9541

 65%|██████▌   | 65/100 [09:16<04:59,  8.57s/it]2026-08-28 20:38:42,743 - INFO - Epoch 66 | Train Loss: 0.4124, Train Acc: 0.8121 | Val Loss: 0.4833
2026-08-28 20:38:42,743 - INFO - Val metrics: Acc = 0.7533 | F1 = 0.7354, AUC = 0.9554

 66%|██████▌   | 66/100 [09:25<04:50,  8.54s/it]2026-08-28 20:38:51,233 - INFO - Epoch 67 | Train Loss: 0.3594, Train Acc: 0.8574 | Val Loss: 0.4836
2026-08-28 20:38:51,233 - INFO - Val metrics: Acc = 0.7333 | F1 = 0.7129, AUC = 0.9523

 67%|██████▋   | 67/100 [09:33<04:41,  8.52s/it]2026-08-28 20:38:59,668 - INFO - Epoch 68 | Train Loss: 0.3696, Train Acc: 0.8356 | Val Loss: 0.5687
2026-08-28 20:38:59,668 - INFO - Val metrics: Acc = 0.6067 | F1 = 0.5347, AUC = 0.9474

 68%|██████▊   | 68/100 [09:41<04:31,  8.50s/it]2026-08-28 20:39:08,168 - INFO - Epoch 69 | Train Loss: 0.3659, Train Acc: 0.8423 | Val Loss: 0.4029
2026-08-28 20:39:08,168 - INFO - Val metrics: Acc = 0.8400 | F1 = 0.8365, AUC = 0.9634

 69%|██████▉   | 69/100 [09:50<04:23,  8.50s/it]2026-08-28 20:39:16,716 - INFO - Epoch 70 | Train Loss: 0.3272, Train Acc: 0.8658 | Val Loss: 0.4747
2026-08-28 20:39:16,716 - INFO - Val metrics: Acc = 0.7200 | F1 = 0.6962, AUC = 0.9570

 70%|███████   | 70/100 [09:59<04:15,  8.51s/it]2026-08-28 20:39:25,289 - INFO - Epoch 71 | Train Loss: 0.3207, Train Acc: 0.8725 | Val Loss: 0.4956
2026-08-28 20:39:25,289 - INFO - Val metrics: Acc = 0.7200 | F1 = 0.6962, AUC = 0.9641

 71%|███████   | 71/100 [10:07<04:07,  8.53s/it]2026-08-28 20:39:33,990 - INFO - Epoch 72 | Train Loss: 0.3259, Train Acc: 0.8389 | Val Loss: 0.5094
2026-08-28 20:39:33,990 - INFO - Val metrics: Acc = 0.6667 | F1 = 0.6250, AUC = 0.9657

 72%|███████▏  | 72/100 [10:16<04:00,  8.58s/it]2026-08-28 20:39:42,497 - INFO - Epoch 73 | Train Loss: 0.2882, Train Acc: 0.8792 | Val Loss: 0.5668
2026-08-28 20:39:42,497 - INFO - Val metrics: Acc = 0.6267 | F1 = 0.5662, AUC = 0.9573

 73%|███████▎  | 73/100 [10:24<03:51,  8.56s/it]2026-08-28 20:39:50,970 - INFO - Epoch 74 | Train Loss: 0.3071, Train Acc: 0.8624 | Val Loss: 0.5745
2026-08-28 20:39:50,970 - INFO - Val metrics: Acc = 0.6200 | F1 = 0.5559, AUC = 0.9609

 74%|███████▍  | 74/100 [10:33<03:41,  8.53s/it]2026-08-28 20:39:59,496 - INFO - Epoch 75 | Train Loss: 0.3335, Train Acc: 0.8490 | Val Loss: 0.5242
2026-08-28 20:39:59,497 - INFO - Val metrics: Acc = 0.6733 | F1 = 0.6343, AUC = 0.9507

 75%|███████▌  | 75/100 [10:41<03:33,  8.53s/it]2026-08-28 20:40:07,934 - INFO - Epoch 76 | Train Loss: 0.3077, Train Acc: 0.8591 | Val Loss: 0.4697
2026-08-28 20:40:07,934 - INFO - Val metrics: Acc = 0.7533 | F1 = 0.7374, AUC = 0.9557

 76%|███████▌  | 76/100 [10:50<03:24,  8.50s/it]2026-08-28 20:40:16,451 - INFO - Epoch 77 | Train Loss: 0.3440, Train Acc: 0.8406 | Val Loss: 0.4902
2026-08-28 20:40:16,451 - INFO - Val metrics: Acc = 0.6933 | F1 = 0.6615, AUC = 0.9627

 77%|███████▋  | 77/100 [10:58<03:15,  8.51s/it]2026-08-28 20:40:24,925 - INFO - Epoch 78 | Train Loss: 0.3145, Train Acc: 0.8540 | Val Loss: 0.6917
2026-08-28 20:40:24,925 - INFO - Val metrics: Acc = 0.5733 | F1 = 0.4709, AUC = 0.9570

 78%|███████▊  | 78/100 [11:07<03:06,  8.50s/it]2026-08-28 20:40:33,340 - INFO - Epoch 79 | Train Loss: 0.2959, Train Acc: 0.8708 | Val Loss: 0.5326
2026-08-28 20:40:33,340 - INFO - Val metrics: Acc = 0.6533 | F1 = 0.6060, AUC = 0.9531

2026-08-28 20:40:33,340 - INFO - --- Stopping training due to early stopping, 20 epochs without improvement ---

 78%|███████▊  | 78/100 [11:15<03:10,  8.66s/it]
2026-08-28 20:40:33,341 - INFO -
--- FOLD 2 ---

2026-08-28 20:40:39,777 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 64, heads=4)
  (residual_proj1): Linear(in_features=10, out_features=256, bias=True)
  (residual_proj2): Linear(in_features=256, out_features=64, bias=True)
  (gat2): GATv2Conv(256, 64, heads=1)
  (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=2, bias=True)
  )
)

  0%|          | 0/100 [00:00<?, ?it/s]2026-08-28 20:40:48,334 - INFO - Epoch 01 | Train Loss: 0.8141, Train Acc: 0.5528 | Val Loss: 0.6790
2026-08-28 20:40:48,334 - INFO - Val metrics: Acc = 0.5705 | F1 = 0.5580, AUC = 0.6458

2026-08-28 20:40:48,334 - INFO - Loss IMPROVEMENT!

  1%|          | 1/100 [00:08<14:12,  8.61s/it]2026-08-28 20:40:56,883 - INFO - Epoch 02 | Train Loss: 0.7011, Train Acc: 0.5477 | Val Loss: 0.6875
2026-08-28 20:40:56,884 - INFO - Val metrics: Acc = 0.4832 | F1 = 0.3663, AUC = 0.6825

  2%|▏         | 2/100 [00:17<13:57,  8.54s/it]2026-08-28 20:41:05,334 - INFO - Epoch 03 | Train Loss: 0.6964, Train Acc: 0.5528 | Val Loss: 0.6784
2026-08-28 20:41:05,334 - INFO - Val metrics: Acc = 0.5436 | F1 = 0.5062, AUC = 0.6777

2026-08-28 20:41:05,334 - INFO - Loss IMPROVEMENT!

  3%|▎         | 3/100 [00:25<13:44,  8.50s/it]2026-08-28 20:41:13,881 - INFO - Epoch 04 | Train Loss: 0.6786, Train Acc: 0.5762 | Val Loss: 0.6810
2026-08-28 20:41:13,882 - INFO - Val metrics: Acc = 0.5638 | F1 = 0.5334, AUC = 0.6820

  4%|▍         | 4/100 [00:34<13:37,  8.52s/it]2026-08-28 20:41:22,542 - INFO - Epoch 05 | Train Loss: 0.6811, Train Acc: 0.5896 | Val Loss: 0.6757
2026-08-28 20:41:22,543 - INFO - Val metrics: Acc = 0.5973 | F1 = 0.5815, AUC = 0.6802

2026-08-28 20:41:22,543 - INFO - Loss IMPROVEMENT!

  5%|▌         | 5/100 [00:42<13:34,  8.57s/it]2026-08-28 20:41:31,067 - INFO - Epoch 06 | Train Loss: 0.6693, Train Acc: 0.6047 | Val Loss: 0.6714
2026-08-28 20:41:31,067 - INFO - Val metrics: Acc = 0.6040 | F1 = 0.6022, AUC = 0.6769

2026-08-28 20:41:31,067 - INFO - Loss IMPROVEMENT!

  6%|▌         | 6/100 [00:51<13:24,  8.56s/it]2026-08-28 20:41:39,590 - INFO - Epoch 07 | Train Loss: 0.6689, Train Acc: 0.5963 | Val Loss: 0.6584
2026-08-28 20:41:39,590 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6663, AUC = 0.7160

2026-08-28 20:41:39,590 - INFO - Loss IMPROVEMENT!

  7%|▋         | 7/100 [00:59<13:14,  8.55s/it]2026-08-28 20:41:48,070 - INFO - Epoch 08 | Train Loss: 0.6721, Train Acc: 0.6114 | Val Loss: 0.6683
2026-08-28 20:41:48,070 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6777, AUC = 0.6996

  8%|▊         | 8/100 [01:08<13:04,  8.52s/it]2026-08-28 20:41:56,612 - INFO - Epoch 09 | Train Loss: 0.6569, Train Acc: 0.6164 | Val Loss: 0.6594
2026-08-28 20:41:56,612 - INFO - Val metrics: Acc = 0.5973 | F1 = 0.5857, AUC = 0.7047

  9%|▉         | 9/100 [01:16<12:56,  8.53s/it]2026-08-28 20:42:05,231 - INFO - Epoch 10 | Train Loss: 0.6559, Train Acc: 0.6281 | Val Loss: 0.6532
2026-08-28 20:42:05,231 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.6287, AUC = 0.7115

2026-08-28 20:42:05,231 - INFO - Loss IMPROVEMENT!

 10%|█         | 10/100 [01:25<12:50,  8.56s/it]2026-08-28 20:42:13,744 - INFO - Epoch 11 | Train Loss: 0.6487, Train Acc: 0.6281 | Val Loss: 0.6467
2026-08-28 20:42:13,744 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.6253, AUC = 0.7209

2026-08-28 20:42:13,744 - INFO - Loss IMPROVEMENT!

 11%|█         | 11/100 [01:33<12:40,  8.54s/it]2026-08-28 20:42:22,260 - INFO - Epoch 12 | Train Loss: 0.6520, Train Acc: 0.6382 | Val Loss: 0.6326
2026-08-28 20:42:22,260 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6709, AUC = 0.7209

2026-08-28 20:42:22,260 - INFO - Loss IMPROVEMENT!

 12%|█▏        | 12/100 [01:42<12:31,  8.54s/it]2026-08-28 20:42:30,760 - INFO - Epoch 13 | Train Loss: 0.6456, Train Acc: 0.6214 | Val Loss: 0.6375
2026-08-28 20:42:30,760 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6577, AUC = 0.7247

 13%|█▎        | 13/100 [01:50<12:21,  8.52s/it]2026-08-28 20:42:39,320 - INFO - Epoch 14 | Train Loss: 0.6447, Train Acc: 0.6482 | Val Loss: 0.6381
2026-08-28 20:42:39,320 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6425, AUC = 0.7272

 14%|█▍        | 14/100 [01:59<12:13,  8.53s/it]2026-08-28 20:42:47,887 - INFO - Epoch 15 | Train Loss: 0.6483, Train Acc: 0.6198 | Val Loss: 0.6449
2026-08-28 20:42:47,887 - INFO - Val metrics: Acc = 0.6040 | F1 = 0.5733, AUC = 0.7321

 15%|█▌        | 15/100 [02:08<12:06,  8.54s/it]2026-08-28 20:42:56,376 - INFO - Epoch 16 | Train Loss: 0.6339, Train Acc: 0.6516 | Val Loss: 0.6278
2026-08-28 20:42:56,376 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.6316, AUC = 0.7306

2026-08-28 20:42:56,376 - INFO - Loss IMPROVEMENT!

 16%|█▌        | 16/100 [02:16<11:56,  8.53s/it]2026-08-28 20:43:04,822 - INFO - Epoch 17 | Train Loss: 0.6317, Train Acc: 0.6516 | Val Loss: 0.6225
2026-08-28 20:43:04,823 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6761, AUC = 0.7364

2026-08-28 20:43:04,823 - INFO - Loss IMPROVEMENT!

 17%|█▋        | 17/100 [02:25<11:45,  8.50s/it]2026-08-28 20:43:13,241 - INFO - Epoch 18 | Train Loss: 0.6353, Train Acc: 0.6466 | Val Loss: 0.6215
2026-08-28 20:43:13,241 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6706, AUC = 0.7337

2026-08-28 20:43:13,242 - INFO - Loss IMPROVEMENT!

 18%|█▊        | 18/100 [02:33<11:35,  8.48s/it]2026-08-28 20:43:21,792 - INFO - Epoch 19 | Train Loss: 0.6037, Train Acc: 0.6851 | Val Loss: 0.6078
2026-08-28 20:43:21,792 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6809, AUC = 0.7562

2026-08-28 20:43:21,792 - INFO - Loss IMPROVEMENT!

 19%|█▉        | 19/100 [02:42<11:28,  8.50s/it]2026-08-28 20:43:30,273 - INFO - Epoch 20 | Train Loss: 0.6324, Train Acc: 0.6482 | Val Loss: 0.6122
2026-08-28 20:43:30,273 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6673, AUC = 0.7625

 20%|██        | 20/100 [02:50<11:19,  8.49s/it]2026-08-28 20:43:38,696 - INFO - Epoch 21 | Train Loss: 0.6086, Train Acc: 0.6801 | Val Loss: 0.6117
2026-08-28 20:43:38,696 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6512, AUC = 0.7643

 21%|██        | 21/100 [02:58<11:09,  8.47s/it]2026-08-28 20:43:47,275 - INFO - Epoch 22 | Train Loss: 0.5852, Train Acc: 0.7035 | Val Loss: 0.6002
2026-08-28 20:43:47,275 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6725, AUC = 0.7739

2026-08-28 20:43:47,275 - INFO - Loss IMPROVEMENT!

 22%|██▏       | 22/100 [03:07<11:03,  8.51s/it]2026-08-28 20:43:55,805 - INFO - Epoch 23 | Train Loss: 0.5749, Train Acc: 0.7270 | Val Loss: 0.5916
2026-08-28 20:43:55,806 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6899, AUC = 0.7957

2026-08-28 20:43:55,806 - INFO - Loss IMPROVEMENT!

 23%|██▎       | 23/100 [03:16<10:55,  8.51s/it]2026-08-28 20:44:04,302 - INFO - Epoch 24 | Train Loss: 0.5879, Train Acc: 0.6884 | Val Loss: 0.6034
2026-08-28 20:44:04,302 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.7036, AUC = 0.7618

 24%|██▍       | 24/100 [03:24<10:46,  8.51s/it]2026-08-28 20:44:12,835 - INFO - Epoch 25 | Train Loss: 0.5552, Train Acc: 0.7337 | Val Loss: 0.5911
2026-08-28 20:44:12,836 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.6998, AUC = 0.7822

2026-08-28 20:44:12,836 - INFO - Loss IMPROVEMENT!

 25%|██▌       | 25/100 [03:33<10:38,  8.52s/it]2026-08-28 20:44:21,298 - INFO - Epoch 26 | Train Loss: 0.5612, Train Acc: 0.7253 | Val Loss: 0.6120
2026-08-28 20:44:21,298 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6924, AUC = 0.7544

 26%|██▌       | 26/100 [03:41<10:28,  8.50s/it]2026-08-28 20:44:29,775 - INFO - Epoch 27 | Train Loss: 0.5526, Train Acc: 0.7136 | Val Loss: 0.5793
2026-08-28 20:44:29,776 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.7114, AUC = 0.7777

2026-08-28 20:44:29,776 - INFO - Loss IMPROVEMENT!

 27%|██▋       | 27/100 [03:50<10:20,  8.49s/it]2026-08-28 20:44:38,351 - INFO - Epoch 28 | Train Loss: 0.5639, Train Acc: 0.7219 | Val Loss: 0.5818
2026-08-28 20:44:38,352 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.7036, AUC = 0.7888

 28%|██▊       | 28/100 [03:58<10:13,  8.52s/it]2026-08-28 20:44:47,170 - INFO - Epoch 29 | Train Loss: 0.5535, Train Acc: 0.7236 | Val Loss: 0.5780
2026-08-28 20:44:47,170 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7135, AUC = 0.7924

2026-08-28 20:44:47,170 - INFO - Loss IMPROVEMENT!

 29%|██▉       | 29/100 [04:07<10:11,  8.61s/it]2026-08-28 20:44:55,798 - INFO - Epoch 30 | Train Loss: 0.5446, Train Acc: 0.7270 | Val Loss: 0.5611
2026-08-28 20:44:55,798 - INFO - Val metrics: Acc = 0.7248 | F1 = 0.7248, AUC = 0.8081

2026-08-28 20:44:55,798 - INFO - Loss IMPROVEMENT!

 30%|███       | 30/100 [04:16<10:03,  8.62s/it]2026-08-28 20:45:04,302 - INFO - Epoch 31 | Train Loss: 0.5223, Train Acc: 0.7353 | Val Loss: 0.5733
2026-08-28 20:45:04,303 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7144, AUC = 0.7998

 31%|███       | 31/100 [04:24<09:51,  8.58s/it]2026-08-28 20:45:12,823 - INFO - Epoch 32 | Train Loss: 0.4900, Train Acc: 0.7940 | Val Loss: 0.5445
2026-08-28 20:45:12,824 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7501, AUC = 0.8124

2026-08-28 20:45:12,824 - INFO - Loss IMPROVEMENT!

 32%|███▏      | 32/100 [04:33<09:42,  8.56s/it]2026-08-28 20:45:21,535 - INFO - Epoch 33 | Train Loss: 0.5100, Train Acc: 0.7471 | Val Loss: 0.5644
2026-08-28 20:45:21,535 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7575, AUC = 0.8020

 33%|███▎      | 33/100 [04:41<09:36,  8.61s/it]2026-08-28 20:45:30,214 - INFO - Epoch 34 | Train Loss: 0.5039, Train Acc: 0.7521 | Val Loss: 0.5908
2026-08-28 20:45:30,214 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7450, AUC = 0.7879

 34%|███▍      | 34/100 [04:50<09:29,  8.63s/it]2026-08-28 20:45:38,935 - INFO - Epoch 35 | Train Loss: 0.4954, Train Acc: 0.7722 | Val Loss: 0.5545
2026-08-28 20:45:38,935 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7335, AUC = 0.8258

 35%|███▌      | 35/100 [04:59<09:22,  8.66s/it]2026-08-28 20:45:47,549 - INFO - Epoch 36 | Train Loss: 0.4824, Train Acc: 0.7588 | Val Loss: 0.5306
2026-08-28 20:45:47,549 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7583, AUC = 0.8339

2026-08-28 20:45:47,549 - INFO - Loss IMPROVEMENT!

 36%|███▌      | 36/100 [05:07<09:13,  8.65s/it]2026-08-28 20:45:56,292 - INFO - Epoch 37 | Train Loss: 0.4763, Train Acc: 0.7772 | Val Loss: 0.5592
2026-08-28 20:45:56,292 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7488, AUC = 0.8171

 37%|███▋      | 37/100 [05:16<09:06,  8.67s/it]2026-08-28 20:46:04,999 - INFO - Epoch 38 | Train Loss: 0.4619, Train Acc: 0.7772 | Val Loss: 0.5270
2026-08-28 20:46:04,999 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7644, AUC = 0.8485

2026-08-28 20:46:04,999 - INFO - Loss IMPROVEMENT!

 38%|███▊      | 38/100 [05:25<08:58,  8.69s/it]2026-08-28 20:46:13,669 - INFO - Epoch 39 | Train Loss: 0.5023, Train Acc: 0.7722 | Val Loss: 0.5600
2026-08-28 20:46:13,669 - INFO - Val metrics: Acc = 0.7718 | F1 = 0.7713, AUC = 0.8213

 39%|███▉      | 39/100 [05:33<08:49,  8.68s/it]2026-08-28 20:46:22,513 - INFO - Epoch 40 | Train Loss: 0.4441, Train Acc: 0.8057 | Val Loss: 0.5523
2026-08-28 20:46:22,514 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7100, AUC = 0.8456

 40%|████      | 40/100 [05:42<08:43,  8.73s/it]2026-08-28 20:46:31,246 - INFO - Epoch 41 | Train Loss: 0.4377, Train Acc: 0.8040 | Val Loss: 0.5866
2026-08-28 20:46:31,246 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6136, AUC = 0.8382

 41%|████      | 41/100 [05:51<08:35,  8.73s/it]2026-08-28 20:46:39,885 - INFO - Epoch 42 | Train Loss: 0.4812, Train Acc: 0.7873 | Val Loss: 0.5702
2026-08-28 20:46:39,885 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7261, AUC = 0.8135

 42%|████▏     | 42/100 [06:00<08:24,  8.70s/it]2026-08-28 20:46:48,555 - INFO - Epoch 43 | Train Loss: 0.5009, Train Acc: 0.7554 | Val Loss: 0.5536
2026-08-28 20:46:48,556 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7383, AUC = 0.8007

 43%|████▎     | 43/100 [06:08<08:15,  8.69s/it]2026-08-28 20:46:57,248 - INFO - Epoch 44 | Train Loss: 0.4472, Train Acc: 0.8040 | Val Loss: 0.5880
2026-08-28 20:46:57,248 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6339, AUC = 0.8375

 44%|████▍     | 44/100 [06:17<08:06,  8.69s/it]2026-08-28 20:47:05,966 - INFO - Epoch 45 | Train Loss: 0.4293, Train Acc: 0.7889 | Val Loss: 0.5306
2026-08-28 20:47:05,966 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7086, AUC = 0.8631

 45%|████▌     | 45/100 [06:26<07:58,  8.70s/it]2026-08-28 20:47:14,755 - INFO - Epoch 46 | Train Loss: 0.4442, Train Acc: 0.8007 | Val Loss: 0.5512
2026-08-28 20:47:14,756 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6793, AUC = 0.8539

 46%|████▌     | 46/100 [06:34<07:51,  8.73s/it]2026-08-28 20:47:23,462 - INFO - Epoch 47 | Train Loss: 0.4313, Train Acc: 0.8124 | Val Loss: 0.5689
2026-08-28 20:47:23,463 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6103, AUC = 0.8616

 47%|████▋     | 47/100 [06:43<07:42,  8.72s/it]2026-08-28 20:47:32,220 - INFO - Epoch 48 | Train Loss: 0.4177, Train Acc: 0.8157 | Val Loss: 0.5066
2026-08-28 20:47:32,221 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7387, AUC = 0.8793

2026-08-28 20:47:32,221 - INFO - Loss IMPROVEMENT!

 48%|████▊     | 48/100 [06:52<07:34,  8.73s/it]2026-08-28 20:47:40,967 - INFO - Epoch 49 | Train Loss: 0.4542, Train Acc: 0.7688 | Val Loss: 0.5608
2026-08-28 20:47:40,967 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6626, AUC = 0.8632

 49%|████▉     | 49/100 [07:01<07:25,  8.74s/it]2026-08-28 20:47:49,553 - INFO - Epoch 50 | Train Loss: 0.4144, Train Acc: 0.8191 | Val Loss: 0.6223
2026-08-28 20:47:49,553 - INFO - Val metrics: Acc = 0.6040 | F1 = 0.5442, AUC = 0.8647

 50%|█████     | 50/100 [07:09<07:14,  8.69s/it]2026-08-28 20:47:58,330 - INFO - Epoch 51 | Train Loss: 0.3798, Train Acc: 0.8392 | Val Loss: 0.6026
2026-08-28 20:47:58,330 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.5920, AUC = 0.8472

 51%|█████     | 51/100 [07:18<07:07,  8.72s/it]2026-08-28 20:48:07,046 - INFO - Epoch 52 | Train Loss: 0.4048, Train Acc: 0.8325 | Val Loss: 0.4857
2026-08-28 20:48:07,046 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7581, AUC = 0.8771

2026-08-28 20:48:07,046 - INFO - Loss IMPROVEMENT!

 52%|█████▏    | 52/100 [07:27<06:58,  8.72s/it]2026-08-28 20:48:15,737 - INFO - Epoch 53 | Train Loss: 0.4404, Train Acc: 0.7940 | Val Loss: 0.4933
2026-08-28 20:48:15,737 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7616, AUC = 0.8757

 53%|█████▎    | 53/100 [07:35<06:49,  8.71s/it]2026-08-28 20:48:24,421 - INFO - Epoch 54 | Train Loss: 0.3984, Train Acc: 0.8241 | Val Loss: 0.6506
2026-08-28 20:48:24,421 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.5493, AUC = 0.8512

 54%|█████▍    | 54/100 [07:44<06:40,  8.70s/it]2026-08-28 20:48:33,178 - INFO - Epoch 55 | Train Loss: 0.4246, Train Acc: 0.8141 | Val Loss: 0.5123
2026-08-28 20:48:33,178 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7510, AUC = 0.8503

 55%|█████▌    | 55/100 [07:53<06:32,  8.72s/it]2026-08-28 20:48:41,837 - INFO - Epoch 56 | Train Loss: 0.4147, Train Acc: 0.8224 | Val Loss: 0.5082
2026-08-28 20:48:41,837 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7766, AUC = 0.8762

 56%|█████▌    | 56/100 [08:02<06:22,  8.70s/it]2026-08-28 20:48:50,530 - INFO - Epoch 57 | Train Loss: 0.3593, Train Acc: 0.8442 | Val Loss: 0.4769
2026-08-28 20:48:50,530 - INFO - Val metrics: Acc = 0.7718 | F1 = 0.7680, AUC = 0.8782

2026-08-28 20:48:50,531 - INFO - Loss IMPROVEMENT!

 57%|█████▋    | 57/100 [08:10<06:14,  8.70s/it]2026-08-28 20:48:59,372 - INFO - Epoch 58 | Train Loss: 0.3785, Train Acc: 0.8358 | Val Loss: 0.5539
2026-08-28 20:48:59,372 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6745, AUC = 0.8850

 58%|█████▊    | 58/100 [08:19<06:07,  8.74s/it]2026-08-28 20:49:08,072 - INFO - Epoch 59 | Train Loss: 0.4013, Train Acc: 0.8275 | Val Loss: 0.4831
2026-08-28 20:49:08,073 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7785, AUC = 0.8629

 59%|█████▉    | 59/100 [08:28<05:57,  8.73s/it]2026-08-28 20:49:16,833 - INFO - Epoch 60 | Train Loss: 0.4156, Train Acc: 0.8342 | Val Loss: 0.6118
2026-08-28 20:49:16,834 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6448, AUC = 0.8214

 60%|██████    | 60/100 [08:37<05:49,  8.74s/it]2026-08-28 20:49:25,690 - INFO - Epoch 61 | Train Loss: 0.3593, Train Acc: 0.8492 | Val Loss: 0.5337
2026-08-28 20:49:25,691 - INFO - Val metrics: Acc = 0.7248 | F1 = 0.7198, AUC = 0.8443

 61%|██████    | 61/100 [08:45<05:42,  8.77s/it]2026-08-28 20:49:34,427 - INFO - Epoch 62 | Train Loss: 0.3571, Train Acc: 0.8476 | Val Loss: 0.6062
2026-08-28 20:49:34,427 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6307, AUC = 0.8557

 62%|██████▏   | 62/100 [08:54<05:32,  8.76s/it]2026-08-28 20:49:43,173 - INFO - Epoch 63 | Train Loss: 0.3669, Train Acc: 0.8476 | Val Loss: 0.5658
2026-08-28 20:49:43,173 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6732, AUC = 0.8544

 63%|██████▎   | 63/100 [09:03<05:24,  8.76s/it]2026-08-28 20:49:51,891 - INFO - Epoch 64 | Train Loss: 0.3601, Train Acc: 0.8526 | Val Loss: 0.5899
2026-08-28 20:49:51,891 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6544, AUC = 0.8820

 64%|██████▍   | 64/100 [09:12<05:14,  8.75s/it]2026-08-28 20:50:00,795 - INFO - Epoch 65 | Train Loss: 0.3739, Train Acc: 0.8275 | Val Loss: 0.5795
2026-08-28 20:50:00,796 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6428, AUC = 0.8640

 65%|██████▌   | 65/100 [09:21<05:07,  8.79s/it]2026-08-28 20:50:09,566 - INFO - Epoch 66 | Train Loss: 0.3767, Train Acc: 0.8291 | Val Loss: 0.4271
2026-08-28 20:50:09,566 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8052, AUC = 0.9050

2026-08-28 20:50:09,566 - INFO - Loss IMPROVEMENT!

 66%|██████▌   | 66/100 [09:29<04:58,  8.79s/it]2026-08-28 20:50:18,273 - INFO - Epoch 67 | Train Loss: 0.4015, Train Acc: 0.8157 | Val Loss: 0.5100
2026-08-28 20:50:18,274 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7261, AUC = 0.8613

 67%|██████▋   | 67/100 [09:38<04:49,  8.76s/it]2026-08-28 20:50:27,011 - INFO - Epoch 68 | Train Loss: 0.3912, Train Acc: 0.8342 | Val Loss: 0.5764
2026-08-28 20:50:27,011 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6508, AUC = 0.8319

 68%|██████▊   | 68/100 [09:47<04:40,  8.75s/it]2026-08-28 20:50:35,807 - INFO - Epoch 69 | Train Loss: 0.3853, Train Acc: 0.8291 | Val Loss: 0.5757
2026-08-28 20:50:35,807 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6160, AUC = 0.8831

 69%|██████▉   | 69/100 [09:56<04:31,  8.77s/it]2026-08-28 20:50:44,560 - INFO - Epoch 70 | Train Loss: 0.3723, Train Acc: 0.8425 | Val Loss: 0.5277
2026-08-28 20:50:44,561 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7210, AUC = 0.8652

 70%|███████   | 70/100 [10:04<04:22,  8.76s/it]2026-08-28 20:50:53,270 - INFO - Epoch 71 | Train Loss: 0.3558, Train Acc: 0.8677 | Val Loss: 0.6375
2026-08-28 20:50:53,270 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.5975, AUC = 0.8638

 71%|███████   | 71/100 [10:13<04:13,  8.75s/it]2026-08-28 20:51:02,014 - INFO - Epoch 72 | Train Loss: 0.3649, Train Acc: 0.8375 | Val Loss: 0.4856
2026-08-28 20:51:02,014 - INFO - Val metrics: Acc = 0.7718 | F1 = 0.7680, AUC = 0.8787

 72%|███████▏  | 72/100 [10:22<04:04,  8.75s/it]2026-08-28 20:51:10,733 - INFO - Epoch 73 | Train Loss: 0.3854, Train Acc: 0.8392 | Val Loss: 0.6184
2026-08-28 20:51:10,733 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.5840, AUC = 0.8605

 73%|███████▎  | 73/100 [10:30<03:55,  8.74s/it]2026-08-28 20:51:19,456 - INFO - Epoch 74 | Train Loss: 0.3321, Train Acc: 0.8677 | Val Loss: 0.4887
2026-08-28 20:51:19,456 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7344, AUC = 0.8760

 74%|███████▍  | 74/100 [10:39<03:47,  8.73s/it]2026-08-28 20:51:28,073 - INFO - Epoch 75 | Train Loss: 0.3425, Train Acc: 0.8710 | Val Loss: 0.7129
2026-08-28 20:51:28,073 - INFO - Val metrics: Acc = 0.5906 | F1 = 0.5175, AUC = 0.8411

 75%|███████▌  | 75/100 [10:48<03:37,  8.70s/it]2026-08-28 20:51:36,836 - INFO - Epoch 76 | Train Loss: 0.3492, Train Acc: 0.8409 | Val Loss: 0.4868
2026-08-28 20:51:36,836 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7649, AUC = 0.8658

 76%|███████▌  | 76/100 [10:57<03:29,  8.72s/it]2026-08-28 20:51:45,595 - INFO - Epoch 77 | Train Loss: 0.3328, Train Acc: 0.8526 | Val Loss: 0.6677
2026-08-28 20:51:45,595 - INFO - Val metrics: Acc = 0.5839 | F1 = 0.5065, AUC = 0.8834

 77%|███████▋  | 77/100 [11:05<03:20,  8.73s/it]2026-08-28 20:51:54,244 - INFO - Epoch 78 | Train Loss: 0.2809, Train Acc: 0.8827 | Val Loss: 0.6198
2026-08-28 20:51:54,244 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.5895, AUC = 0.8858

 78%|███████▊  | 78/100 [11:14<03:11,  8.71s/it]2026-08-28 20:52:03,041 - INFO - Epoch 79 | Train Loss: 0.3550, Train Acc: 0.8526 | Val Loss: 0.5667
2026-08-28 20:52:03,042 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6686, AUC = 0.8564

 79%|███████▉  | 79/100 [11:23<03:03,  8.73s/it]2026-08-28 20:52:11,800 - INFO - Epoch 80 | Train Loss: 0.3297, Train Acc: 0.8610 | Val Loss: 0.5307
2026-08-28 20:52:11,800 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7238, AUC = 0.8535

 80%|████████  | 80/100 [11:32<02:54,  8.74s/it]2026-08-28 20:52:20,457 - INFO - Epoch 81 | Train Loss: 0.3097, Train Acc: 0.8677 | Val Loss: 0.5712
2026-08-28 20:52:20,458 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6181, AUC = 0.8705

 81%|████████  | 81/100 [11:40<02:45,  8.72s/it]2026-08-28 20:52:29,324 - INFO - Epoch 82 | Train Loss: 0.3095, Train Acc: 0.8811 | Val Loss: 0.5747
2026-08-28 20:52:29,324 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6224, AUC = 0.8625

 82%|████████▏ | 82/100 [11:49<02:37,  8.76s/it]2026-08-28 20:52:37,907 - INFO - Epoch 83 | Train Loss: 0.2826, Train Acc: 0.8995 | Val Loss: 0.4873
2026-08-28 20:52:37,907 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7461, AUC = 0.8807

 83%|████████▎ | 83/100 [11:58<02:28,  8.71s/it]2026-08-28 20:52:46,545 - INFO - Epoch 84 | Train Loss: 0.3052, Train Acc: 0.8811 | Val Loss: 0.5108
2026-08-28 20:52:46,545 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7112, AUC = 0.8690

 84%|████████▍ | 84/100 [12:06<02:18,  8.69s/it]2026-08-28 20:52:55,044 - INFO - Epoch 85 | Train Loss: 0.3017, Train Acc: 0.8760 | Val Loss: 0.6116
2026-08-28 20:52:55,044 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6274, AUC = 0.8643

 85%|████████▌ | 85/100 [12:15<02:09,  8.63s/it]2026-08-28 20:53:03,611 - INFO - Epoch 86 | Train Loss: 0.3256, Train Acc: 0.8861 | Val Loss: 0.4828
2026-08-28 20:53:03,612 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7575, AUC = 0.8650

 86%|████████▌ | 86/100 [12:23<02:00,  8.61s/it]2026-08-28 20:53:12,211 - INFO - Epoch 87 | Train Loss: 0.2877, Train Acc: 0.8777 | Val Loss: 0.5774
2026-08-28 20:53:12,211 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.6805, AUC = 0.8732

2026-08-28 20:53:12,211 - INFO - --- Stopping training due to early stopping, 20 epochs without improvement ---

 86%|████████▌ | 86/100 [12:32<02:02,  8.75s/it]
2026-08-28 20:53:12,211 - INFO -
--- FOLD 3 ---

2026-08-28 20:53:18,799 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 64, heads=4)
  (residual_proj1): Linear(in_features=10, out_features=256, bias=True)
  (residual_proj2): Linear(in_features=256, out_features=64, bias=True)
  (gat2): GATv2Conv(256, 64, heads=1)
  (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=2, bias=True)
  )
)

  0%|          | 0/100 [00:00<?, ?it/s]2026-08-28 20:53:27,411 - INFO - Epoch 01 | Train Loss: 0.8341, Train Acc: 0.5427 | Val Loss: 0.7053
2026-08-28 20:53:27,411 - INFO - Val metrics: Acc = 0.4698 | F1 = 0.4601, AUC = 0.4548

2026-08-28 20:53:27,411 - INFO - Loss IMPROVEMENT!

  1%|          | 1/100 [00:08<14:20,  8.70s/it]2026-08-28 20:53:35,995 - INFO - Epoch 02 | Train Loss: 0.6973, Train Acc: 0.5695 | Val Loss: 0.6825
2026-08-28 20:53:35,995 - INFO - Val metrics: Acc = 0.5436 | F1 = 0.5434, AUC = 0.5798

2026-08-28 20:53:35,995 - INFO - Loss IMPROVEMENT!

  2%|▏         | 2/100 [00:17<14:01,  8.59s/it]2026-08-28 20:53:44,542 - INFO - Epoch 03 | Train Loss: 0.6940, Train Acc: 0.5544 | Val Loss: 0.6737
2026-08-28 20:53:44,543 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6439, AUC = 0.6712

2026-08-28 20:53:44,543 - INFO - Loss IMPROVEMENT!

  3%|▎         | 3/100 [00:25<13:51,  8.57s/it]2026-08-28 20:53:53,207 - INFO - Epoch 04 | Train Loss: 0.6754, Train Acc: 0.5946 | Val Loss: 0.6646
2026-08-28 20:53:53,208 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6130, AUC = 0.6560

2026-08-28 20:53:53,208 - INFO - Loss IMPROVEMENT!

  4%|▍         | 4/100 [00:34<13:46,  8.61s/it]2026-08-28 20:54:01,843 - INFO - Epoch 05 | Train Loss: 0.6705, Train Acc: 0.5913 | Val Loss: 0.6676
2026-08-28 20:54:01,844 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6644, AUC = 0.6714

  5%|▌         | 5/100 [00:43<13:38,  8.61s/it]2026-08-28 20:54:10,369 - INFO - Epoch 06 | Train Loss: 0.6577, Train Acc: 0.6114 | Val Loss: 0.6617
2026-08-28 20:54:10,369 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.6363, AUC = 0.6668

2026-08-28 20:54:10,369 - INFO - Loss IMPROVEMENT!

  6%|▌         | 6/100 [00:51<13:27,  8.59s/it]2026-08-28 20:54:18,990 - INFO - Epoch 07 | Train Loss: 0.6583, Train Acc: 0.6131 | Val Loss: 0.6761
2026-08-28 20:54:18,991 - INFO - Val metrics: Acc = 0.5705 | F1 = 0.5027, AUC = 0.6663

  7%|▋         | 7/100 [01:00<13:19,  8.60s/it]2026-08-28 20:54:27,655 - INFO - Epoch 08 | Train Loss: 0.6580, Train Acc: 0.6214 | Val Loss: 0.6483
2026-08-28 20:54:27,655 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6292, AUC = 0.6795

2026-08-28 20:54:27,655 - INFO - Loss IMPROVEMENT!

  8%|▊         | 8/100 [01:08<13:12,  8.62s/it]2026-08-28 20:54:36,276 - INFO - Epoch 09 | Train Loss: 0.6623, Train Acc: 0.5930 | Val Loss: 0.6623
2026-08-28 20:54:36,276 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6163, AUC = 0.6798

  9%|▉         | 9/100 [01:17<13:04,  8.62s/it]2026-08-28 20:54:44,923 - INFO - Epoch 10 | Train Loss: 0.6581, Train Acc: 0.6164 | Val Loss: 0.6548
2026-08-28 20:54:44,923 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6364, AUC = 0.6966

 10%|█         | 10/100 [01:26<12:56,  8.63s/it]2026-08-28 20:54:53,572 - INFO - Epoch 11 | Train Loss: 0.6589, Train Acc: 0.6399 | Val Loss: 0.6494
2026-08-28 20:54:53,572 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6526, AUC = 0.6982

 11%|█         | 11/100 [01:34<12:48,  8.63s/it]2026-08-28 20:55:02,255 - INFO - Epoch 12 | Train Loss: 0.6615, Train Acc: 0.6382 | Val Loss: 0.6460
2026-08-28 20:55:02,255 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6483, AUC = 0.7059

2026-08-28 20:55:02,255 - INFO - Loss IMPROVEMENT!

 12%|█▏        | 12/100 [01:43<12:41,  8.65s/it]2026-08-28 20:55:10,891 - INFO - Epoch 13 | Train Loss: 0.6361, Train Acc: 0.6549 | Val Loss: 0.6300
2026-08-28 20:55:10,892 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6767, AUC = 0.7160

2026-08-28 20:55:10,892 - INFO - Loss IMPROVEMENT!

 13%|█▎        | 13/100 [01:52<12:32,  8.65s/it]2026-08-28 20:55:19,682 - INFO - Epoch 14 | Train Loss: 0.6302, Train Acc: 0.6633 | Val Loss: 0.6402
2026-08-28 20:55:19,682 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6409, AUC = 0.6995

 14%|█▍        | 14/100 [02:00<12:27,  8.69s/it]2026-08-28 20:55:28,369 - INFO - Epoch 15 | Train Loss: 0.6334, Train Acc: 0.6533 | Val Loss: 0.6307
2026-08-28 20:55:28,369 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6567, AUC = 0.7139

 15%|█▌        | 15/100 [02:09<12:18,  8.69s/it]2026-08-28 20:55:36,947 - INFO - Epoch 16 | Train Loss: 0.6153, Train Acc: 0.6901 | Val Loss: 0.6322
2026-08-28 20:55:36,947 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6420, AUC = 0.7099

 16%|█▌        | 16/100 [02:18<12:06,  8.65s/it]2026-08-28 20:55:45,531 - INFO - Epoch 17 | Train Loss: 0.6324, Train Acc: 0.6516 | Val Loss: 0.6420
2026-08-28 20:55:45,531 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6285, AUC = 0.6933

 17%|█▋        | 17/100 [02:26<11:56,  8.63s/it]2026-08-28 20:55:54,200 - INFO - Epoch 18 | Train Loss: 0.6245, Train Acc: 0.6633 | Val Loss: 0.6213
2026-08-28 20:55:54,201 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6637, AUC = 0.7272

2026-08-28 20:55:54,201 - INFO - Loss IMPROVEMENT!

 18%|█▊        | 18/100 [02:35<11:49,  8.65s/it]2026-08-28 20:56:02,872 - INFO - Epoch 19 | Train Loss: 0.6150, Train Acc: 0.6700 | Val Loss: 0.6329
2026-08-28 20:56:02,872 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6562, AUC = 0.7148

 19%|█▉        | 19/100 [02:44<11:40,  8.65s/it]2026-08-28 20:56:11,451 - INFO - Epoch 20 | Train Loss: 0.6330, Train Acc: 0.6449 | Val Loss: 0.6410
2026-08-28 20:56:11,451 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6089, AUC = 0.7099

 20%|██        | 20/100 [02:52<11:30,  8.63s/it]2026-08-28 20:56:20,046 - INFO - Epoch 21 | Train Loss: 0.6005, Train Acc: 0.6717 | Val Loss: 0.6279
2026-08-28 20:56:20,046 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6673, AUC = 0.7106

 21%|██        | 21/100 [03:01<11:20,  8.62s/it]2026-08-28 20:56:28,636 - INFO - Epoch 22 | Train Loss: 0.6372, Train Acc: 0.6616 | Val Loss: 0.6411
2026-08-28 20:56:28,636 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6589, AUC = 0.6968

 22%|██▏       | 22/100 [03:09<11:11,  8.61s/it]2026-08-28 20:56:37,251 - INFO - Epoch 23 | Train Loss: 0.5778, Train Acc: 0.7102 | Val Loss: 0.6390
2026-08-28 20:56:37,251 - INFO - Val metrics: Acc = 0.6242 | F1 = 0.6048, AUC = 0.7301

 23%|██▎       | 23/100 [03:18<11:03,  8.61s/it]2026-08-28 20:56:45,848 - INFO - Epoch 24 | Train Loss: 0.6170, Train Acc: 0.6533 | Val Loss: 0.6457
2026-08-28 20:56:45,848 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6271, AUC = 0.7180

 24%|██▍       | 24/100 [03:27<10:54,  8.61s/it]2026-08-28 20:56:54,479 - INFO - Epoch 25 | Train Loss: 0.6096, Train Acc: 0.6533 | Val Loss: 0.6234
2026-08-28 20:56:54,479 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6486, AUC = 0.7207

 25%|██▌       | 25/100 [03:35<10:46,  8.61s/it]2026-08-28 20:57:03,132 - INFO - Epoch 26 | Train Loss: 0.5871, Train Acc: 0.6985 | Val Loss: 0.6142
2026-08-28 20:57:03,132 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6843, AUC = 0.7346

2026-08-28 20:57:03,132 - INFO - Loss IMPROVEMENT!

 26%|██▌       | 26/100 [03:44<10:38,  8.63s/it]2026-08-28 20:57:11,754 - INFO - Epoch 27 | Train Loss: 0.6161, Train Acc: 0.6784 | Val Loss: 0.6120
2026-08-28 20:57:11,754 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6509, AUC = 0.7350

2026-08-28 20:57:11,754 - INFO - Loss IMPROVEMENT!

 27%|██▋       | 27/100 [03:52<10:29,  8.63s/it]2026-08-28 20:57:20,400 - INFO - Epoch 28 | Train Loss: 0.5910, Train Acc: 0.7035 | Val Loss: 0.6270
2026-08-28 20:57:20,401 - INFO - Val metrics: Acc = 0.6242 | F1 = 0.6094, AUC = 0.7330

 28%|██▊       | 28/100 [04:01<10:21,  8.63s/it]2026-08-28 20:57:29,034 - INFO - Epoch 29 | Train Loss: 0.5998, Train Acc: 0.6884 | Val Loss: 0.6056
2026-08-28 20:57:29,034 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6713, AUC = 0.7505

2026-08-28 20:57:29,034 - INFO - Loss IMPROVEMENT!

 29%|██▉       | 29/100 [04:10<10:12,  8.63s/it]2026-08-28 20:57:37,598 - INFO - Epoch 30 | Train Loss: 0.5451, Train Acc: 0.7270 | Val Loss: 0.6881
2026-08-28 20:57:37,598 - INFO - Val metrics: Acc = 0.5906 | F1 = 0.5652, AUC = 0.6932

 30%|███       | 30/100 [04:18<10:02,  8.61s/it]2026-08-28 20:57:46,265 - INFO - Epoch 31 | Train Loss: 0.5851, Train Acc: 0.6985 | Val Loss: 0.6099
2026-08-28 20:57:46,265 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6673, AUC = 0.7494

 31%|███       | 31/100 [04:27<09:55,  8.63s/it]2026-08-28 20:57:54,794 - INFO - Epoch 32 | Train Loss: 0.5642, Train Acc: 0.7219 | Val Loss: 0.6087
2026-08-28 20:57:54,794 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6831, AUC = 0.7438

 32%|███▏      | 32/100 [04:35<09:44,  8.60s/it]2026-08-28 20:58:03,381 - INFO - Epoch 33 | Train Loss: 0.5762, Train Acc: 0.7136 | Val Loss: 0.6144
2026-08-28 20:58:03,381 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6960, AUC = 0.7508

 33%|███▎      | 33/100 [04:44<09:35,  8.59s/it]2026-08-28 20:58:11,976 - INFO - Epoch 34 | Train Loss: 0.5652, Train Acc: 0.7018 | Val Loss: 0.6109
2026-08-28 20:58:11,976 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6651, AUC = 0.7335

 34%|███▍      | 34/100 [04:53<09:27,  8.59s/it]2026-08-28 20:58:20,585 - INFO - Epoch 35 | Train Loss: 0.5716, Train Acc: 0.7069 | Val Loss: 0.5857
2026-08-28 20:58:20,586 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7135, AUC = 0.7847

2026-08-28 20:58:20,586 - INFO - Loss IMPROVEMENT!

 35%|███▌      | 35/100 [05:01<09:19,  8.60s/it]2026-08-28 20:58:29,195 - INFO - Epoch 36 | Train Loss: 0.5531, Train Acc: 0.7119 | Val Loss: 0.6135
2026-08-28 20:58:29,195 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6799, AUC = 0.7443

 36%|███▌      | 36/100 [05:10<09:10,  8.60s/it]2026-08-28 20:58:37,721 - INFO - Epoch 37 | Train Loss: 0.5387, Train Acc: 0.7303 | Val Loss: 0.5957
2026-08-28 20:58:37,721 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6624, AUC = 0.7658

 37%|███▋      | 37/100 [05:18<09:00,  8.58s/it]2026-08-28 20:58:46,120 - INFO - Epoch 38 | Train Loss: 0.5407, Train Acc: 0.7253 | Val Loss: 0.5684
2026-08-28 20:58:46,121 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7181, AUC = 0.7867

2026-08-28 20:58:46,121 - INFO - Loss IMPROVEMENT!

 38%|███▊      | 38/100 [05:27<08:48,  8.53s/it]2026-08-28 20:58:54,728 - INFO - Epoch 39 | Train Loss: 0.5253, Train Acc: 0.7487 | Val Loss: 0.5821
2026-08-28 20:58:54,728 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6747, AUC = 0.7987

 39%|███▉      | 39/100 [05:35<08:41,  8.55s/it]2026-08-28 20:59:03,366 - INFO - Epoch 40 | Train Loss: 0.4899, Train Acc: 0.7705 | Val Loss: 0.5584
2026-08-28 20:59:03,366 - INFO - Val metrics: Acc = 0.7248 | F1 = 0.7248, AUC = 0.7912

2026-08-28 20:59:03,367 - INFO - Loss IMPROVEMENT!

 40%|████      | 40/100 [05:44<08:34,  8.58s/it]2026-08-28 20:59:11,968 - INFO - Epoch 41 | Train Loss: 0.5253, Train Acc: 0.7504 | Val Loss: 0.5536
2026-08-28 20:59:11,969 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7175, AUC = 0.8061

2026-08-28 20:59:11,969 - INFO - Loss IMPROVEMENT!

 41%|████      | 41/100 [05:53<08:26,  8.59s/it]2026-08-28 20:59:20,540 - INFO - Epoch 42 | Train Loss: 0.5205, Train Acc: 0.7454 | Val Loss: 0.5941
2026-08-28 20:59:20,540 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6530, AUC = 0.7856

 42%|████▏     | 42/100 [06:01<08:17,  8.58s/it]2026-08-28 20:59:29,108 - INFO - Epoch 43 | Train Loss: 0.5200, Train Acc: 0.7437 | Val Loss: 0.5839
2026-08-28 20:59:29,108 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.7112, AUC = 0.7789

 43%|████▎     | 43/100 [06:10<08:08,  8.58s/it]2026-08-28 20:59:37,663 - INFO - Epoch 44 | Train Loss: 0.4867, Train Acc: 0.7806 | Val Loss: 0.5567
2026-08-28 20:59:37,664 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7288, AUC = 0.7939

 44%|████▍     | 44/100 [06:18<07:59,  8.57s/it]2026-08-28 20:59:46,271 - INFO - Epoch 45 | Train Loss: 0.4620, Train Acc: 0.7822 | Val Loss: 0.6960
2026-08-28 20:59:46,271 - INFO - Val metrics: Acc = 0.5570 | F1 = 0.4811, AUC = 0.7413

 45%|████▌     | 45/100 [06:27<07:51,  8.58s/it]2026-08-28 20:59:54,823 - INFO - Epoch 46 | Train Loss: 0.4988, Train Acc: 0.7688 | Val Loss: 0.6211
2026-08-28 20:59:54,823 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.5809, AUC = 0.7876

 46%|████▌     | 46/100 [06:36<07:42,  8.57s/it]2026-08-28 21:00:03,456 - INFO - Epoch 47 | Train Loss: 0.4693, Train Acc: 0.7806 | Val Loss: 0.5044
2026-08-28 21:00:03,457 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7584, AUC = 0.8474

2026-08-28 21:00:03,457 - INFO - Loss IMPROVEMENT!

 47%|████▋     | 47/100 [06:44<07:35,  8.59s/it]2026-08-28 21:00:11,972 - INFO - Epoch 48 | Train Loss: 0.4700, Train Acc: 0.7806 | Val Loss: 0.5619
2026-08-28 21:00:11,972 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6791, AUC = 0.8232

 48%|████▊     | 48/100 [06:53<07:25,  8.57s/it]2026-08-28 21:00:20,696 - INFO - Epoch 49 | Train Loss: 0.4615, Train Acc: 0.7873 | Val Loss: 0.6062
2026-08-28 21:00:20,697 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.6138, AUC = 0.7894

 49%|████▉     | 49/100 [07:01<07:19,  8.61s/it]2026-08-28 21:00:29,363 - INFO - Epoch 50 | Train Loss: 0.4691, Train Acc: 0.7839 | Val Loss: 0.6172
2026-08-28 21:00:29,363 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6389, AUC = 0.7647

 50%|█████     | 50/100 [07:10<07:11,  8.63s/it]2026-08-28 21:00:38,002 - INFO - Epoch 51 | Train Loss: 0.4762, Train Acc: 0.7806 | Val Loss: 0.6134
2026-08-28 21:00:38,002 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6663, AUC = 0.7454

 51%|█████     | 51/100 [07:19<07:03,  8.63s/it]2026-08-28 21:00:46,557 - INFO - Epoch 52 | Train Loss: 0.4530, Train Acc: 0.7856 | Val Loss: 0.6100
2026-08-28 21:00:46,557 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6456, AUC = 0.8128

 52%|█████▏    | 52/100 [07:27<06:53,  8.61s/it]2026-08-28 21:00:55,140 - INFO - Epoch 53 | Train Loss: 0.4738, Train Acc: 0.7973 | Val Loss: 0.5617
2026-08-28 21:00:55,140 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6693, AUC = 0.8353

 53%|█████▎    | 53/100 [07:36<06:44,  8.60s/it]2026-08-28 21:01:03,702 - INFO - Epoch 54 | Train Loss: 0.4348, Train Acc: 0.8124 | Val Loss: 0.5310
2026-08-28 21:01:03,702 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7288, AUC = 0.8387

 54%|█████▍    | 54/100 [07:44<06:35,  8.59s/it]2026-08-28 21:01:12,350 - INFO - Epoch 55 | Train Loss: 0.4429, Train Acc: 0.8057 | Val Loss: 0.7058
2026-08-28 21:01:12,350 - INFO - Val metrics: Acc = 0.5906 | F1 = 0.5114, AUC = 0.8283

 55%|█████▌    | 55/100 [07:53<06:27,  8.61s/it]2026-08-28 21:01:20,996 - INFO - Epoch 56 | Train Loss: 0.4848, Train Acc: 0.7655 | Val Loss: 0.5255
2026-08-28 21:01:20,996 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7315, AUC = 0.8386

 56%|█████▌    | 56/100 [08:02<06:19,  8.62s/it]2026-08-28 21:01:29,596 - INFO - Epoch 57 | Train Loss: 0.4723, Train Acc: 0.7688 | Val Loss: 0.5169
2026-08-28 21:01:29,596 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7640, AUC = 0.8463

 57%|█████▋    | 57/100 [08:10<06:10,  8.61s/it]2026-08-28 21:01:38,202 - INFO - Epoch 58 | Train Loss: 0.4077, Train Acc: 0.8258 | Val Loss: 0.5376
2026-08-28 21:01:38,202 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7324, AUC = 0.8368

 58%|█████▊    | 58/100 [08:19<06:01,  8.61s/it]2026-08-28 21:01:46,794 - INFO - Epoch 59 | Train Loss: 0.4092, Train Acc: 0.8057 | Val Loss: 0.5785
2026-08-28 21:01:46,794 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6633, AUC = 0.8068

 59%|█████▉    | 59/100 [08:27<05:52,  8.61s/it]2026-08-28 21:01:55,384 - INFO - Epoch 60 | Train Loss: 0.4197, Train Acc: 0.8258 | Val Loss: 0.5728
2026-08-28 21:01:55,384 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6311, AUC = 0.8400

 60%|██████    | 60/100 [08:36<05:44,  8.60s/it]2026-08-28 21:02:03,985 - INFO - Epoch 61 | Train Loss: 0.3888, Train Acc: 0.8425 | Val Loss: 0.5499
2026-08-28 21:02:03,985 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6885, AUC = 0.8288

 61%|██████    | 61/100 [08:45<05:35,  8.60s/it]2026-08-28 21:02:12,610 - INFO - Epoch 62 | Train Loss: 0.3752, Train Acc: 0.8409 | Val Loss: 0.5526
2026-08-28 21:02:12,610 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7250, AUC = 0.8276

 62%|██████▏   | 62/100 [08:53<05:27,  8.61s/it]2026-08-28 21:02:21,122 - INFO - Epoch 63 | Train Loss: 0.3995, Train Acc: 0.8392 | Val Loss: 0.5278
2026-08-28 21:02:21,122 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7224, AUC = 0.8604

 63%|██████▎   | 63/100 [09:02<05:17,  8.58s/it]2026-08-28 21:02:29,710 - INFO - Epoch 64 | Train Loss: 0.3688, Train Acc: 0.8425 | Val Loss: 0.5768
2026-08-28 21:02:29,710 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6369, AUC = 0.8452

 64%|██████▍   | 64/100 [09:10<05:08,  8.58s/it]2026-08-28 21:02:38,242 - INFO - Epoch 65 | Train Loss: 0.4060, Train Acc: 0.8124 | Val Loss: 0.5356
2026-08-28 21:02:38,242 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7160, AUC = 0.8214

 65%|██████▌   | 65/100 [09:19<04:59,  8.57s/it]2026-08-28 21:02:46,952 - INFO - Epoch 66 | Train Loss: 0.3779, Train Acc: 0.8442 | Val Loss: 0.6009
2026-08-28 21:02:46,952 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6068, AUC = 0.8360

 66%|██████▌   | 66/100 [09:28<04:52,  8.61s/it]2026-08-28 21:02:55,633 - INFO - Epoch 67 | Train Loss: 0.3684, Train Acc: 0.8476 | Val Loss: 0.5422
2026-08-28 21:02:55,633 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6754, AUC = 0.8499

 67%|██████▋   | 67/100 [09:36<04:44,  8.63s/it]2026-08-28 21:03:04,320 - INFO - Epoch 68 | Train Loss: 0.3692, Train Acc: 0.8375 | Val Loss: 0.5039
2026-08-28 21:03:04,320 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7525, AUC = 0.8764

2026-08-28 21:03:04,321 - INFO - Loss IMPROVEMENT!

 68%|██████▊   | 68/100 [09:45<04:36,  8.65s/it]2026-08-28 21:03:12,896 - INFO - Epoch 69 | Train Loss: 0.3635, Train Acc: 0.8593 | Val Loss: 0.5749
2026-08-28 21:03:12,896 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6217, AUC = 0.8764

 69%|██████▉   | 69/100 [09:54<04:27,  8.63s/it]2026-08-28 21:03:21,566 - INFO - Epoch 70 | Train Loss: 0.3704, Train Acc: 0.8442 | Val Loss: 0.5965
2026-08-28 21:03:21,566 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6068, AUC = 0.8638

 70%|███████   | 70/100 [10:02<04:19,  8.64s/it]2026-08-28 21:03:30,131 - INFO - Epoch 71 | Train Loss: 0.3749, Train Acc: 0.8476 | Val Loss: 0.5559
2026-08-28 21:03:30,131 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6567, AUC = 0.8613

 71%|███████   | 71/100 [10:11<04:09,  8.62s/it]2026-08-28 21:03:38,719 - INFO - Epoch 72 | Train Loss: 0.3644, Train Acc: 0.8392 | Val Loss: 0.4852
2026-08-28 21:03:38,719 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7873, AUC = 0.8820

2026-08-28 21:03:38,719 - INFO - Loss IMPROVEMENT!

 72%|███████▏  | 72/100 [10:19<04:01,  8.61s/it]2026-08-28 21:03:47,372 - INFO - Epoch 73 | Train Loss: 0.3825, Train Acc: 0.8342 | Val Loss: 0.6349
2026-08-28 21:03:47,373 - INFO - Val metrics: Acc = 0.6242 | F1 = 0.5649, AUC = 0.8600

 73%|███████▎  | 73/100 [10:28<03:52,  8.62s/it]2026-08-28 21:03:55,932 - INFO - Epoch 74 | Train Loss: 0.3484, Train Acc: 0.8543 | Val Loss: 0.5281
2026-08-28 21:03:55,932 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7224, AUC = 0.8521

 74%|███████▍  | 74/100 [10:37<03:43,  8.60s/it]2026-08-28 21:04:04,587 - INFO - Epoch 75 | Train Loss: 0.3727, Train Acc: 0.8308 | Val Loss: 0.4923
2026-08-28 21:04:04,588 - INFO - Val metrics: Acc = 0.7718 | F1 = 0.7706, AUC = 0.8595

 75%|███████▌  | 75/100 [10:45<03:35,  8.62s/it]2026-08-28 21:04:13,159 - INFO - Epoch 76 | Train Loss: 0.3288, Train Acc: 0.8610 | Val Loss: 0.5206
2026-08-28 21:04:13,159 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7300, AUC = 0.8541

 76%|███████▌  | 76/100 [10:54<03:26,  8.60s/it]2026-08-28 21:04:21,777 - INFO - Epoch 77 | Train Loss: 0.3478, Train Acc: 0.8543 | Val Loss: 0.5716
2026-08-28 21:04:21,777 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.6895, AUC = 0.8299

 77%|███████▋  | 77/100 [11:02<03:17,  8.61s/it]2026-08-28 21:04:30,346 - INFO - Epoch 78 | Train Loss: 0.3212, Train Acc: 0.8677 | Val Loss: 0.5275
2026-08-28 21:04:30,346 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7238, AUC = 0.8519

 78%|███████▊  | 78/100 [11:11<03:09,  8.60s/it]2026-08-28 21:04:38,852 - INFO - Epoch 79 | Train Loss: 0.3375, Train Acc: 0.8526 | Val Loss: 0.5394
2026-08-28 21:04:38,853 - INFO - Val metrics: Acc = 0.7248 | F1 = 0.7162, AUC = 0.8517

 79%|███████▉  | 79/100 [11:20<02:59,  8.57s/it]2026-08-28 21:04:47,375 - INFO - Epoch 80 | Train Loss: 0.3373, Train Acc: 0.8492 | Val Loss: 0.5028
2026-08-28 21:04:47,375 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7261, AUC = 0.8677

 80%|████████  | 80/100 [11:28<02:51,  8.56s/it]2026-08-28 21:04:55,942 - INFO - Epoch 81 | Train Loss: 0.3306, Train Acc: 0.8492 | Val Loss: 0.4648
2026-08-28 21:04:55,942 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7901, AUC = 0.8786

2026-08-28 21:04:55,942 - INFO - Loss IMPROVEMENT!

 81%|████████  | 81/100 [11:37<02:42,  8.56s/it]2026-08-28 21:05:04,514 - INFO - Epoch 82 | Train Loss: 0.3324, Train Acc: 0.8509 | Val Loss: 0.5480
2026-08-28 21:05:04,514 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6773, AUC = 0.8465

 82%|████████▏ | 82/100 [11:45<02:34,  8.56s/it]2026-08-28 21:05:13,089 - INFO - Epoch 83 | Train Loss: 0.3347, Train Acc: 0.8660 | Val Loss: 0.5256
2026-08-28 21:05:13,089 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7086, AUC = 0.8605

 83%|████████▎ | 83/100 [11:54<02:25,  8.57s/it]2026-08-28 21:05:21,620 - INFO - Epoch 84 | Train Loss: 0.3041, Train Acc: 0.8677 | Val Loss: 0.6736
2026-08-28 21:05:21,620 - INFO - Val metrics: Acc = 0.6242 | F1 = 0.5597, AUC = 0.8614

 84%|████████▍ | 84/100 [12:02<02:16,  8.56s/it]2026-08-28 21:05:30,195 - INFO - Epoch 85 | Train Loss: 0.3121, Train Acc: 0.8811 | Val Loss: 0.6047
2026-08-28 21:05:30,195 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6307, AUC = 0.8562

 85%|████████▌ | 85/100 [12:11<02:08,  8.56s/it]2026-08-28 21:05:38,730 - INFO - Epoch 86 | Train Loss: 0.3517, Train Acc: 0.8543 | Val Loss: 0.5981
2026-08-28 21:05:38,731 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6483, AUC = 0.8254

 86%|████████▌ | 86/100 [12:19<01:59,  8.55s/it]2026-08-28 21:05:47,396 - INFO - Epoch 87 | Train Loss: 0.3527, Train Acc: 0.8610 | Val Loss: 0.6219
2026-08-28 21:05:47,396 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.5840, AUC = 0.8589

 87%|████████▋ | 87/100 [12:28<01:51,  8.59s/it]2026-08-28 21:05:55,978 - INFO - Epoch 88 | Train Loss: 0.3267, Train Acc: 0.8643 | Val Loss: 0.6822
2026-08-28 21:05:55,978 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.5491, AUC = 0.8641

 88%|████████▊ | 88/100 [12:37<01:43,  8.59s/it]2026-08-28 21:06:04,602 - INFO - Epoch 89 | Train Loss: 0.3213, Train Acc: 0.8777 | Val Loss: 0.4616
2026-08-28 21:06:04,602 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8172, AUC = 0.8969

2026-08-28 21:06:04,602 - INFO - Loss IMPROVEMENT!

 89%|████████▉ | 89/100 [12:45<01:34,  8.60s/it]2026-08-28 21:06:13,065 - INFO - Epoch 90 | Train Loss: 0.3134, Train Acc: 0.8509 | Val Loss: 0.5723
2026-08-28 21:06:13,065 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6660, AUC = 0.8551

 90%|█████████ | 90/100 [12:54<01:25,  8.56s/it]2026-08-28 21:06:21,532 - INFO - Epoch 91 | Train Loss: 0.3218, Train Acc: 0.8811 | Val Loss: 0.5923
2026-08-28 21:06:21,532 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6573, AUC = 0.8732

 91%|█████████ | 91/100 [13:02<01:16,  8.53s/it]2026-08-28 21:06:30,069 - INFO - Epoch 92 | Train Loss: 0.2998, Train Acc: 0.8744 | Val Loss: 0.4719
2026-08-28 21:06:30,069 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8036, AUC = 0.8814

 92%|█████████▏| 92/100 [13:11<01:08,  8.53s/it]2026-08-28 21:06:38,555 - INFO - Epoch 93 | Train Loss: 0.3876, Train Acc: 0.8308 | Val Loss: 0.5670
2026-08-28 21:06:38,556 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6660, AUC = 0.8654

 93%|█████████▎| 93/100 [13:19<00:59,  8.52s/it]2026-08-28 21:06:46,997 - INFO - Epoch 94 | Train Loss: 0.3327, Train Acc: 0.8660 | Val Loss: 0.5235
2026-08-28 21:06:46,997 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7318, AUC = 0.8786

 94%|█████████▍| 94/100 [13:28<00:50,  8.50s/it]2026-08-28 21:06:55,428 - INFO - Epoch 95 | Train Loss: 0.3226, Train Acc: 0.8593 | Val Loss: 0.5938
2026-08-28 21:06:55,428 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6515, AUC = 0.8479

 95%|█████████▌| 95/100 [13:36<00:42,  8.48s/it]2026-08-28 21:07:03,772 - INFO - Epoch 96 | Train Loss: 0.3208, Train Acc: 0.8693 | Val Loss: 0.6325
2026-08-28 21:07:03,772 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.5544, AUC = 0.8746

 96%|█████████▌| 96/100 [13:44<00:33,  8.44s/it]2026-08-28 21:07:12,256 - INFO - Epoch 97 | Train Loss: 0.3120, Train Acc: 0.8643 | Val Loss: 0.6948
2026-08-28 21:07:12,256 - INFO - Val metrics: Acc = 0.5839 | F1 = 0.5001, AUC = 0.8604

 97%|█████████▋| 97/100 [13:53<00:25,  8.45s/it]2026-08-28 21:07:20,669 - INFO - Epoch 98 | Train Loss: 0.3714, Train Acc: 0.8492 | Val Loss: 0.5938
2026-08-28 21:07:20,669 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6124, AUC = 0.8708

 98%|█████████▊| 98/100 [14:01<00:16,  8.44s/it]2026-08-28 21:07:29,156 - INFO - Epoch 99 | Train Loss: 0.3304, Train Acc: 0.8626 | Val Loss: 0.5360
2026-08-28 21:07:29,156 - INFO - Val metrics: Acc = 0.7248 | F1 = 0.7078, AUC = 0.8906

 99%|█████████▉| 99/100 [14:10<00:08,  8.45s/it]2026-08-28 21:07:37,695 - INFO - Epoch 100 | Train Loss: 0.2964, Train Acc: 0.8777 | Val Loss: 0.5760
2026-08-28 21:07:37,695 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6745, AUC = 0.8903

100%|██████████| 100/100 [14:18<00:00,  8.59s/it]
2026-08-28 21:07:37,696 - INFO -
--- FOLD 4 ---

2026-08-28 21:07:43,991 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 64, heads=4)
  (residual_proj1): Linear(in_features=10, out_features=256, bias=True)
  (residual_proj2): Linear(in_features=256, out_features=64, bias=True)
  (gat2): GATv2Conv(256, 64, heads=1)
  (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=2, bias=True)
  )
)

  0%|          | 0/100 [00:00<?, ?it/s]2026-08-28 21:07:52,503 - INFO - Epoch 01 | Train Loss: 0.8141, Train Acc: 0.5394 | Val Loss: 0.6672
2026-08-28 21:07:52,503 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6746, AUC = 0.7004

2026-08-28 21:07:52,503 - INFO - Loss IMPROVEMENT!

  1%|          | 1/100 [00:08<14:05,  8.54s/it]2026-08-28 21:08:01,077 - INFO - Epoch 02 | Train Loss: 0.6891, Train Acc: 0.5511 | Val Loss: 0.6962
2026-08-28 21:08:01,078 - INFO - Val metrics: Acc = 0.4899 | F1 = 0.3288, AUC = 0.6554

  2%|▏         | 2/100 [00:17<13:57,  8.54s/it]2026-08-28 21:08:09,597 - INFO - Epoch 03 | Train Loss: 0.7035, Train Acc: 0.5394 | Val Loss: 0.6840
2026-08-28 21:08:09,597 - INFO - Val metrics: Acc = 0.5235 | F1 = 0.4237, AUC = 0.7313

  3%|▎         | 3/100 [00:25<13:47,  8.53s/it]2026-08-28 21:08:18,131 - INFO - Epoch 04 | Train Loss: 0.6848, Train Acc: 0.5695 | Val Loss: 0.6901
2026-08-28 21:08:18,131 - INFO - Val metrics: Acc = 0.4966 | F1 = 0.3541, AUC = 0.7008

  4%|▍         | 4/100 [00:34<13:39,  8.53s/it]2026-08-28 21:08:26,587 - INFO - Epoch 05 | Train Loss: 0.6767, Train Acc: 0.5796 | Val Loss: 0.6669
2026-08-28 21:08:26,587 - INFO - Val metrics: Acc = 0.6040 | F1 = 0.5623, AUC = 0.7084

2026-08-28 21:08:26,587 - INFO - Loss IMPROVEMENT!

  5%|▌         | 5/100 [00:42<13:28,  8.51s/it]2026-08-28 21:08:35,139 - INFO - Epoch 06 | Train Loss: 0.6778, Train Acc: 0.5930 | Val Loss: 0.6700
2026-08-28 21:08:35,139 - INFO - Val metrics: Acc = 0.5503 | F1 = 0.4764, AUC = 0.7127

  6%|▌         | 6/100 [00:51<13:20,  8.52s/it]2026-08-28 21:08:43,626 - INFO - Epoch 07 | Train Loss: 0.6743, Train Acc: 0.5595 | Val Loss: 0.6612
2026-08-28 21:08:43,626 - INFO - Val metrics: Acc = 0.6242 | F1 = 0.5934, AUC = 0.7132

2026-08-28 21:08:43,626 - INFO - Loss IMPROVEMENT!

  7%|▋         | 7/100 [00:59<13:11,  8.51s/it]2026-08-28 21:08:52,095 - INFO - Epoch 08 | Train Loss: 0.6724, Train Acc: 0.6080 | Val Loss: 0.6772
2026-08-28 21:08:52,095 - INFO - Val metrics: Acc = 0.5168 | F1 = 0.4032, AUC = 0.7098

  8%|▊         | 8/100 [01:08<13:01,  8.50s/it]2026-08-28 21:09:00,679 - INFO - Epoch 09 | Train Loss: 0.6549, Train Acc: 0.6365 | Val Loss: 0.6455
2026-08-28 21:09:00,679 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6641, AUC = 0.7141

2026-08-28 21:09:00,679 - INFO - Loss IMPROVEMENT!

  9%|▉         | 9/100 [01:16<12:55,  8.53s/it]2026-08-28 21:09:09,168 - INFO - Epoch 10 | Train Loss: 0.6515, Train Acc: 0.6382 | Val Loss: 0.6368
2026-08-28 21:09:09,168 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6285, AUC = 0.7121

2026-08-28 21:09:09,168 - INFO - Loss IMPROVEMENT!

 10%|█         | 10/100 [01:25<12:46,  8.51s/it]2026-08-28 21:09:17,742 - INFO - Epoch 11 | Train Loss: 0.6375, Train Acc: 0.6516 | Val Loss: 0.6397
2026-08-28 21:09:17,742 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6474, AUC = 0.7096

 11%|█         | 11/100 [01:33<12:39,  8.53s/it]2026-08-28 21:09:26,297 - INFO - Epoch 12 | Train Loss: 0.6464, Train Acc: 0.6533 | Val Loss: 0.6491
2026-08-28 21:09:26,298 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6153, AUC = 0.7199

 12%|█▏        | 12/100 [01:42<12:31,  8.54s/it]2026-08-28 21:09:34,918 - INFO - Epoch 13 | Train Loss: 0.6487, Train Acc: 0.6214 | Val Loss: 0.6332
2026-08-28 21:09:34,919 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6491, AUC = 0.7141

2026-08-28 21:09:34,919 - INFO - Loss IMPROVEMENT!

 13%|█▎        | 13/100 [01:50<12:25,  8.57s/it]2026-08-28 21:09:43,528 - INFO - Epoch 14 | Train Loss: 0.6427, Train Acc: 0.6399 | Val Loss: 0.6513
2026-08-28 21:09:43,528 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6174, AUC = 0.7044

 14%|█▍        | 14/100 [01:59<12:17,  8.58s/it]2026-08-28 21:09:52,080 - INFO - Epoch 15 | Train Loss: 0.6458, Train Acc: 0.6382 | Val Loss: 0.6697
2026-08-28 21:09:52,080 - INFO - Val metrics: Acc = 0.5705 | F1 = 0.5135, AUC = 0.7091

 15%|█▌        | 15/100 [02:08<12:08,  8.57s/it]2026-08-28 21:10:00,711 - INFO - Epoch 16 | Train Loss: 0.6380, Train Acc: 0.6533 | Val Loss: 0.6738
2026-08-28 21:10:00,711 - INFO - Val metrics: Acc = 0.5638 | F1 = 0.4920, AUC = 0.7204

 16%|█▌        | 16/100 [02:16<12:01,  8.59s/it]2026-08-28 21:10:09,274 - INFO - Epoch 17 | Train Loss: 0.6267, Train Acc: 0.6566 | Val Loss: 0.6356
2026-08-28 21:10:09,274 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6547, AUC = 0.7233

 17%|█▋        | 17/100 [02:25<11:52,  8.58s/it]2026-08-28 21:10:17,826 - INFO - Epoch 18 | Train Loss: 0.6350, Train Acc: 0.6566 | Val Loss: 0.6331
2026-08-28 21:10:17,826 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6610, AUC = 0.7311

2026-08-28 21:10:17,826 - INFO - Loss IMPROVEMENT!

 18%|█▊        | 18/100 [02:33<11:43,  8.57s/it]2026-08-28 21:10:26,434 - INFO - Epoch 19 | Train Loss: 0.6254, Train Acc: 0.6683 | Val Loss: 0.6448
2026-08-28 21:10:26,434 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6452, AUC = 0.7350

 19%|█▉        | 19/100 [02:42<11:35,  8.58s/it]2026-08-28 21:10:34,966 - INFO - Epoch 20 | Train Loss: 0.6422, Train Acc: 0.6348 | Val Loss: 0.6324
2026-08-28 21:10:34,966 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6619, AUC = 0.7336

2026-08-28 21:10:34,966 - INFO - Loss IMPROVEMENT!

 20%|██        | 20/100 [02:50<11:25,  8.57s/it]2026-08-28 21:10:43,469 - INFO - Epoch 21 | Train Loss: 0.6164, Train Acc: 0.6767 | Val Loss: 0.6155
2026-08-28 21:10:43,469 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6682, AUC = 0.7433

2026-08-28 21:10:43,469 - INFO - Loss IMPROVEMENT!

 21%|██        | 21/100 [02:59<11:15,  8.55s/it]2026-08-28 21:10:52,052 - INFO - Epoch 22 | Train Loss: 0.6235, Train Acc: 0.6667 | Val Loss: 0.6157
2026-08-28 21:10:52,052 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6673, AUC = 0.7484

 22%|██▏       | 22/100 [03:08<11:07,  8.56s/it]2026-08-28 21:11:00,854 - INFO - Epoch 23 | Train Loss: 0.6167, Train Acc: 0.6784 | Val Loss: 0.5997
2026-08-28 21:11:00,854 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.7047, AUC = 0.7531

2026-08-28 21:11:00,854 - INFO - Loss IMPROVEMENT!

 23%|██▎       | 23/100 [03:16<11:04,  8.63s/it]2026-08-28 21:11:09,422 - INFO - Epoch 24 | Train Loss: 0.6111, Train Acc: 0.6817 | Val Loss: 0.6242
2026-08-28 21:11:09,422 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6392, AUC = 0.7552

 24%|██▍       | 24/100 [03:25<10:54,  8.61s/it]2026-08-28 21:11:18,070 - INFO - Epoch 25 | Train Loss: 0.6091, Train Acc: 0.6918 | Val Loss: 0.6139
2026-08-28 21:11:18,070 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6869, AUC = 0.7850

 25%|██▌       | 25/100 [03:34<10:46,  8.62s/it]2026-08-28 21:11:26,690 - INFO - Epoch 26 | Train Loss: 0.5969, Train Acc: 0.6767 | Val Loss: 0.5912
2026-08-28 21:11:26,690 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7180, AUC = 0.7733

2026-08-28 21:11:26,690 - INFO - Loss IMPROVEMENT!

 26%|██▌       | 26/100 [03:42<10:38,  8.62s/it]2026-08-28 21:11:35,224 - INFO - Epoch 27 | Train Loss: 0.5873, Train Acc: 0.6868 | Val Loss: 0.6258
2026-08-28 21:11:35,224 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6542, AUC = 0.7848

 27%|██▋       | 27/100 [03:51<10:27,  8.59s/it]2026-08-28 21:11:43,805 - INFO - Epoch 28 | Train Loss: 0.5931, Train Acc: 0.6700 | Val Loss: 0.6528
2026-08-28 21:11:43,805 - INFO - Val metrics: Acc = 0.5906 | F1 = 0.5233, AUC = 0.7833

 28%|██▊       | 28/100 [03:59<10:18,  8.59s/it]2026-08-28 21:11:52,414 - INFO - Epoch 29 | Train Loss: 0.6088, Train Acc: 0.6834 | Val Loss: 0.5865
2026-08-28 21:11:52,414 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.7040, AUC = 0.7806

2026-08-28 21:11:52,414 - INFO - Loss IMPROVEMENT!

 29%|██▉       | 29/100 [04:08<10:10,  8.60s/it]2026-08-28 21:12:01,073 - INFO - Epoch 30 | Train Loss: 0.5906, Train Acc: 0.7002 | Val Loss: 0.6152
2026-08-28 21:12:01,073 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6567, AUC = 0.7990

 30%|███       | 30/100 [04:17<10:03,  8.61s/it]2026-08-28 21:12:09,657 - INFO - Epoch 31 | Train Loss: 0.6003, Train Acc: 0.6784 | Val Loss: 0.5840
2026-08-28 21:12:09,657 - INFO - Val metrics: Acc = 0.7248 | F1 = 0.7240, AUC = 0.8019

2026-08-28 21:12:09,657 - INFO - Loss IMPROVEMENT!

 31%|███       | 31/100 [04:25<09:53,  8.61s/it]2026-08-28 21:12:18,246 - INFO - Epoch 32 | Train Loss: 0.5682, Train Acc: 0.7219 | Val Loss: 0.5677
2026-08-28 21:12:18,246 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7359, AUC = 0.8192

2026-08-28 21:12:18,246 - INFO - Loss IMPROVEMENT!

 32%|███▏      | 32/100 [04:34<09:44,  8.60s/it]2026-08-28 21:12:26,819 - INFO - Epoch 33 | Train Loss: 0.5500, Train Acc: 0.7253 | Val Loss: 0.5581
2026-08-28 21:12:26,819 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7166, AUC = 0.8147

2026-08-28 21:12:26,819 - INFO - Loss IMPROVEMENT!

 33%|███▎      | 33/100 [04:42<09:35,  8.59s/it]2026-08-28 21:12:35,570 - INFO - Epoch 34 | Train Loss: 0.5553, Train Acc: 0.7052 | Val Loss: 0.5487
2026-08-28 21:12:35,570 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7584, AUC = 0.8347

2026-08-28 21:12:35,570 - INFO - Loss IMPROVEMENT!

 34%|███▍      | 34/100 [04:51<09:30,  8.64s/it]2026-08-28 21:12:44,139 - INFO - Epoch 35 | Train Loss: 0.5677, Train Acc: 0.7018 | Val Loss: 0.6323
2026-08-28 21:12:44,139 - INFO - Val metrics: Acc = 0.5973 | F1 = 0.5338, AUC = 0.8367

 35%|███▌      | 35/100 [05:00<09:20,  8.62s/it]2026-08-28 21:12:52,696 - INFO - Epoch 36 | Train Loss: 0.5831, Train Acc: 0.6951 | Val Loss: 0.5623
2026-08-28 21:12:52,697 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7480, AUC = 0.8358

 36%|███▌      | 36/100 [05:08<09:10,  8.60s/it]2026-08-28 21:13:01,134 - INFO - Epoch 37 | Train Loss: 0.5734, Train Acc: 0.7052 | Val Loss: 0.5430
2026-08-28 21:13:01,134 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7647, AUC = 0.8376

2026-08-28 21:13:01,134 - INFO - Loss IMPROVEMENT!

 37%|███▋      | 37/100 [05:17<08:58,  8.55s/it]2026-08-28 21:13:09,700 - INFO - Epoch 38 | Train Loss: 0.5360, Train Acc: 0.7437 | Val Loss: 0.5404
2026-08-28 21:13:09,701 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7544, AUC = 0.8484

2026-08-28 21:13:09,701 - INFO - Loss IMPROVEMENT!

 38%|███▊      | 38/100 [05:25<08:50,  8.56s/it]2026-08-28 21:13:18,176 - INFO - Epoch 39 | Train Loss: 0.5459, Train Acc: 0.7219 | Val Loss: 0.5310
2026-08-28 21:13:18,176 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7771, AUC = 0.8744

2026-08-28 21:13:18,176 - INFO - Loss IMPROVEMENT!

 39%|███▉      | 39/100 [05:34<08:40,  8.53s/it]2026-08-28 21:13:26,744 - INFO - Epoch 40 | Train Loss: 0.5178, Train Acc: 0.7504 | Val Loss: 0.5287
2026-08-28 21:13:26,745 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7525, AUC = 0.8857

2026-08-28 21:13:26,745 - INFO - Loss IMPROVEMENT!

 40%|████      | 40/100 [05:42<08:32,  8.54s/it]2026-08-28 21:13:35,378 - INFO - Epoch 41 | Train Loss: 0.4926, Train Acc: 0.7571 | Val Loss: 0.5200
2026-08-28 21:13:35,378 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8041, AUC = 0.8801

2026-08-28 21:13:35,378 - INFO - Loss IMPROVEMENT!

 41%|████      | 41/100 [05:51<08:25,  8.57s/it]2026-08-28 21:13:43,884 - INFO - Epoch 42 | Train Loss: 0.5252, Train Acc: 0.7521 | Val Loss: 0.5149
2026-08-28 21:13:43,884 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8052, AUC = 0.8798

2026-08-28 21:13:43,884 - INFO - Loss IMPROVEMENT!

 42%|████▏     | 42/100 [05:59<08:15,  8.55s/it]2026-08-28 21:13:52,466 - INFO - Epoch 43 | Train Loss: 0.5112, Train Acc: 0.7471 | Val Loss: 0.5090
2026-08-28 21:13:52,466 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8054, AUC = 0.8751

2026-08-28 21:13:52,466 - INFO - Loss IMPROVEMENT!

 43%|████▎     | 43/100 [06:08<08:07,  8.56s/it]2026-08-28 21:14:01,004 - INFO - Epoch 44 | Train Loss: 0.5090, Train Acc: 0.7370 | Val Loss: 0.5047
2026-08-28 21:14:01,004 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8111, AUC = 0.8937

2026-08-28 21:14:01,004 - INFO - Loss IMPROVEMENT!

 44%|████▍     | 44/100 [06:17<07:59,  8.55s/it]2026-08-28 21:14:09,411 - INFO - Epoch 45 | Train Loss: 0.5131, Train Acc: 0.7554 | Val Loss: 0.5181
2026-08-28 21:14:09,411 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7830, AUC = 0.8855

 45%|████▌     | 45/100 [06:25<07:47,  8.51s/it]2026-08-28 21:14:17,990 - INFO - Epoch 46 | Train Loss: 0.4838, Train Acc: 0.7655 | Val Loss: 0.4988
2026-08-28 21:14:17,990 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7913, AUC = 0.9000

2026-08-28 21:14:17,990 - INFO - Loss IMPROVEMENT!

 46%|████▌     | 46/100 [06:34<07:40,  8.53s/it]2026-08-28 21:14:26,701 - INFO - Epoch 47 | Train Loss: 0.5036, Train Acc: 0.7571 | Val Loss: 0.4964
2026-08-28 21:14:26,701 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7817, AUC = 0.8859

2026-08-28 21:14:26,702 - INFO - Loss IMPROVEMENT!

 47%|████▋     | 47/100 [06:42<07:35,  8.59s/it]2026-08-28 21:14:35,353 - INFO - Epoch 48 | Train Loss: 0.4839, Train Acc: 0.7588 | Val Loss: 0.4977
2026-08-28 21:14:35,353 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8387, AUC = 0.9054

 48%|████▊     | 48/100 [06:51<07:27,  8.60s/it]2026-08-28 21:14:44,067 - INFO - Epoch 49 | Train Loss: 0.5019, Train Acc: 0.7672 | Val Loss: 0.5017
2026-08-28 21:14:44,067 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8188, AUC = 0.8834

 49%|████▉     | 49/100 [07:00<07:20,  8.64s/it]2026-08-28 21:14:52,775 - INFO - Epoch 50 | Train Loss: 0.5180, Train Acc: 0.7554 | Val Loss: 0.4786
2026-08-28 21:14:52,775 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8254, AUC = 0.9045

2026-08-28 21:14:52,775 - INFO - Loss IMPROVEMENT!

 50%|█████     | 50/100 [07:08<07:12,  8.66s/it]2026-08-28 21:15:01,454 - INFO - Epoch 51 | Train Loss: 0.4626, Train Acc: 0.7906 | Val Loss: 0.4489
2026-08-28 21:15:01,454 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8255, AUC = 0.9236

2026-08-28 21:15:01,454 - INFO - Loss IMPROVEMENT!

 51%|█████     | 51/100 [07:17<07:04,  8.67s/it]2026-08-28 21:15:10,151 - INFO - Epoch 52 | Train Loss: 0.4609, Train Acc: 0.7873 | Val Loss: 0.4787
2026-08-28 21:15:10,151 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8245, AUC = 0.9207

 52%|█████▏    | 52/100 [07:26<06:56,  8.67s/it]2026-08-28 21:15:18,883 - INFO - Epoch 53 | Train Loss: 0.4446, Train Acc: 0.7923 | Val Loss: 0.4292
2026-08-28 21:15:18,883 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8387, AUC = 0.9158

2026-08-28 21:15:18,883 - INFO - Loss IMPROVEMENT!

 53%|█████▎    | 53/100 [07:34<06:48,  8.69s/it]2026-08-28 21:15:27,588 - INFO - Epoch 54 | Train Loss: 0.4175, Train Acc: 0.8174 | Val Loss: 0.4469
2026-08-28 21:15:27,588 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8584, AUC = 0.9304

 54%|█████▍    | 54/100 [07:43<06:39,  8.69s/it]2026-08-28 21:15:36,276 - INFO - Epoch 55 | Train Loss: 0.4556, Train Acc: 0.7990 | Val Loss: 0.5027
2026-08-28 21:15:36,276 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8588, AUC = 0.9364

 55%|█████▌    | 55/100 [07:52<06:31,  8.69s/it]2026-08-28 21:15:44,950 - INFO - Epoch 56 | Train Loss: 0.4742, Train Acc: 0.7755 | Val Loss: 0.4849
2026-08-28 21:15:44,950 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8518, AUC = 0.9302

 56%|█████▌    | 56/100 [08:00<06:22,  8.69s/it]2026-08-28 21:15:53,689 - INFO - Epoch 57 | Train Loss: 0.4111, Train Acc: 0.8023 | Val Loss: 0.4323
2026-08-28 21:15:53,689 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8045, AUC = 0.9131

 57%|█████▋    | 57/100 [08:09<06:14,  8.70s/it]2026-08-28 21:16:02,788 - INFO - Epoch 58 | Train Loss: 0.4414, Train Acc: 0.7973 | Val Loss: 0.4772
2026-08-28 21:16:02,788 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8002, AUC = 0.9146

 58%|█████▊    | 58/100 [08:18<06:10,  8.82s/it]2026-08-28 21:16:13,275 - INFO - Epoch 59 | Train Loss: 0.4251, Train Acc: 0.7990 | Val Loss: 0.4967
2026-08-28 21:16:13,275 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7459, AUC = 0.9239

 59%|█████▉    | 59/100 [08:29<06:22,  9.32s/it]2026-08-28 21:16:21,644 - INFO - Epoch 60 | Train Loss: 0.4352, Train Acc: 0.8040 | Val Loss: 0.5079
2026-08-28 21:16:21,644 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.6779, AUC = 0.9146

 60%|██████    | 60/100 [08:37<06:01,  9.04s/it]2026-08-28 21:16:30,006 - INFO - Epoch 61 | Train Loss: 0.4278, Train Acc: 0.8107 | Val Loss: 0.4768
2026-08-28 21:16:30,006 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8251, AUC = 0.9211

 61%|██████    | 61/100 [08:46<05:44,  8.83s/it]2026-08-28 21:16:38,239 - INFO - Epoch 62 | Train Loss: 0.4252, Train Acc: 0.7889 | Val Loss: 0.4317
2026-08-28 21:16:38,239 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8515, AUC = 0.9165

 62%|██████▏   | 62/100 [08:54<05:28,  8.65s/it]2026-08-28 21:16:46,564 - INFO - Epoch 63 | Train Loss: 0.3736, Train Acc: 0.8543 | Val Loss: 0.4319
2026-08-28 21:16:46,564 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8010, AUC = 0.9337

 63%|██████▎   | 63/100 [09:02<05:16,  8.55s/it]2026-08-28 21:16:54,772 - INFO - Epoch 64 | Train Loss: 0.3969, Train Acc: 0.8308 | Val Loss: 0.4824
2026-08-28 21:16:54,772 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7505, AUC = 0.9461

 64%|██████▍   | 64/100 [09:10<05:04,  8.45s/it]2026-08-28 21:17:03,298 - INFO - Epoch 65 | Train Loss: 0.3786, Train Acc: 0.8358 | Val Loss: 0.4111
2026-08-28 21:17:03,298 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8590, AUC = 0.9456

2026-08-28 21:17:03,298 - INFO - Loss IMPROVEMENT!

 65%|██████▌   | 65/100 [09:19<04:56,  8.48s/it]2026-08-28 21:17:11,942 - INFO - Epoch 66 | Train Loss: 0.3787, Train Acc: 0.8291 | Val Loss: 0.4223
2026-08-28 21:17:11,942 - INFO - Val metrics: Acc = 0.8725 | F1 = 0.8723, AUC = 0.9409

 66%|██████▌   | 66/100 [09:27<04:49,  8.52s/it]2026-08-28 21:17:20,514 - INFO - Epoch 67 | Train Loss: 0.3502, Train Acc: 0.8409 | Val Loss: 0.4133
2026-08-28 21:17:20,514 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8590, AUC = 0.9358

 67%|██████▋   | 67/100 [09:36<04:41,  8.54s/it]2026-08-28 21:17:29,096 - INFO - Epoch 68 | Train Loss: 0.3536, Train Acc: 0.8459 | Val Loss: 0.4174
2026-08-28 21:17:29,097 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8656, AUC = 0.9281

 68%|██████▊   | 68/100 [09:45<04:33,  8.55s/it]2026-08-28 21:17:37,715 - INFO - Epoch 69 | Train Loss: 0.3299, Train Acc: 0.8576 | Val Loss: 0.4301
2026-08-28 21:17:37,715 - INFO - Val metrics: Acc = 0.8456 | F1 = 0.8434, AUC = 0.9436

 69%|██████▉   | 69/100 [09:53<04:25,  8.57s/it]2026-08-28 21:17:46,465 - INFO - Epoch 70 | Train Loss: 0.3583, Train Acc: 0.8442 | Val Loss: 0.4547
2026-08-28 21:17:46,465 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.7993, AUC = 0.9371

 70%|███████   | 70/100 [10:02<04:18,  8.63s/it]2026-08-28 21:17:55,136 - INFO - Epoch 71 | Train Loss: 0.3404, Train Acc: 0.8643 | Val Loss: 0.4695
2026-08-28 21:17:55,136 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7678, AUC = 0.9275

 71%|███████   | 71/100 [10:11<04:10,  8.64s/it]2026-08-28 21:18:03,846 - INFO - Epoch 72 | Train Loss: 0.3463, Train Acc: 0.8559 | Val Loss: 0.4549
2026-08-28 21:18:03,847 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8002, AUC = 0.9311

 72%|███████▏  | 72/100 [10:19<04:02,  8.66s/it]2026-08-28 21:18:12,410 - INFO - Epoch 73 | Train Loss: 0.3645, Train Acc: 0.8492 | Val Loss: 0.4673
2026-08-28 21:18:12,410 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8140, AUC = 0.9342

 73%|███████▎  | 73/100 [10:28<03:53,  8.63s/it]2026-08-28 21:18:21,043 - INFO - Epoch 74 | Train Loss: 0.3480, Train Acc: 0.8660 | Val Loss: 0.4707
2026-08-28 21:18:21,043 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7459, AUC = 0.9367

 74%|███████▍  | 74/100 [10:37<03:44,  8.63s/it]2026-08-28 21:18:29,704 - INFO - Epoch 75 | Train Loss: 0.3425, Train Acc: 0.8358 | Val Loss: 0.4257
2026-08-28 21:18:29,704 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8581, AUC = 0.9299

 75%|███████▌  | 75/100 [10:45<03:36,  8.64s/it]2026-08-28 21:18:38,377 - INFO - Epoch 76 | Train Loss: 0.3254, Train Acc: 0.8727 | Val Loss: 0.4429
2026-08-28 21:18:38,377 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8504, AUC = 0.9396

 76%|███████▌  | 76/100 [10:54<03:27,  8.65s/it]2026-08-28 21:18:46,976 - INFO - Epoch 77 | Train Loss: 0.3112, Train Acc: 0.8811 | Val Loss: 0.4208
2026-08-28 21:18:46,976 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8512, AUC = 0.9349

 77%|███████▋  | 77/100 [11:02<03:18,  8.63s/it]2026-08-28 21:18:55,523 - INFO - Epoch 78 | Train Loss: 0.3321, Train Acc: 0.8492 | Val Loss: 0.4940
2026-08-28 21:18:55,524 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7522, AUC = 0.9286

 78%|███████▊  | 78/100 [11:11<03:09,  8.61s/it]2026-08-28 21:19:04,236 - INFO - Epoch 79 | Train Loss: 0.3200, Train Acc: 0.8727 | Val Loss: 0.4698
2026-08-28 21:19:04,236 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7664, AUC = 0.9439

 79%|███████▉  | 79/100 [11:20<03:01,  8.64s/it]2026-08-28 21:19:13,003 - INFO - Epoch 80 | Train Loss: 0.3260, Train Acc: 0.8526 | Val Loss: 0.4375
2026-08-28 21:19:13,004 - INFO - Val metrics: Acc = 0.8322 | F1 = 0.8269, AUC = 0.9441

 80%|████████  | 80/100 [11:29<02:53,  8.68s/it]2026-08-28 21:19:21,677 - INFO - Epoch 81 | Train Loss: 0.3095, Train Acc: 0.8677 | Val Loss: 0.4649
2026-08-28 21:19:21,677 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7664, AUC = 0.9405

 81%|████████  | 81/100 [11:37<02:44,  8.68s/it]2026-08-28 21:19:30,402 - INFO - Epoch 82 | Train Loss: 0.2872, Train Acc: 0.9012 | Val Loss: 0.4931
2026-08-28 21:19:30,403 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.6838, AUC = 0.9380

 82%|████████▏ | 82/100 [11:46<02:36,  8.69s/it]2026-08-28 21:19:39,072 - INFO - Epoch 83 | Train Loss: 0.2622, Train Acc: 0.8894 | Val Loss: 0.4032
2026-08-28 21:19:39,072 - INFO - Val metrics: Acc = 0.8725 | F1 = 0.8713, AUC = 0.9369

2026-08-28 21:19:39,072 - INFO - Loss IMPROVEMENT!

 83%|████████▎ | 83/100 [11:55<02:27,  8.69s/it]2026-08-28 21:19:47,749 - INFO - Epoch 84 | Train Loss: 0.2979, Train Acc: 0.8811 | Val Loss: 0.4277
2026-08-28 21:19:47,749 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8357, AUC = 0.9389

 84%|████████▍ | 84/100 [12:03<02:18,  8.68s/it]2026-08-28 21:19:56,395 - INFO - Epoch 85 | Train Loss: 0.2955, Train Acc: 0.8844 | Val Loss: 0.4249
2026-08-28 21:19:56,395 - INFO - Val metrics: Acc = 0.8456 | F1 = 0.8422, AUC = 0.9405

 85%|████████▌ | 85/100 [12:12<02:10,  8.67s/it]2026-08-28 21:20:05,058 - INFO - Epoch 86 | Train Loss: 0.2854, Train Acc: 0.8811 | Val Loss: 0.4142
2026-08-28 21:20:05,058 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8499, AUC = 0.9403

 86%|████████▌ | 86/100 [12:21<02:01,  8.67s/it]2026-08-28 21:20:13,663 - INFO - Epoch 87 | Train Loss: 0.2928, Train Acc: 0.8811 | Val Loss: 0.4590
2026-08-28 21:20:13,663 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7664, AUC = 0.9382

 87%|████████▋ | 87/100 [12:29<01:52,  8.65s/it]2026-08-28 21:20:22,347 - INFO - Epoch 88 | Train Loss: 0.2812, Train Acc: 0.8844 | Val Loss: 0.3901
2026-08-28 21:20:22,347 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8578, AUC = 0.9420

2026-08-28 21:20:22,347 - INFO - Loss IMPROVEMENT!

 88%|████████▊ | 88/100 [12:38<01:43,  8.66s/it]2026-08-28 21:20:31,067 - INFO - Epoch 89 | Train Loss: 0.2869, Train Acc: 0.8777 | Val Loss: 0.4321
2026-08-28 21:20:31,067 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8343, AUC = 0.9448

 89%|████████▉ | 89/100 [12:47<01:35,  8.68s/it]2026-08-28 21:20:39,667 - INFO - Epoch 90 | Train Loss: 0.2583, Train Acc: 0.8995 | Val Loss: 0.4127
2026-08-28 21:20:39,667 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8565, AUC = 0.9450

 90%|█████████ | 90/100 [12:55<01:26,  8.65s/it]2026-08-28 21:20:48,240 - INFO - Epoch 91 | Train Loss: 0.2977, Train Acc: 0.8710 | Val Loss: 0.4005
2026-08-28 21:20:48,240 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8644, AUC = 0.9405

 91%|█████████ | 91/100 [13:04<01:17,  8.63s/it]2026-08-28 21:20:56,922 - INFO - Epoch 92 | Train Loss: 0.3015, Train Acc: 0.8811 | Val Loss: 0.4215
2026-08-28 21:20:56,923 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8559, AUC = 0.9465

 92%|█████████▏| 92/100 [13:12<01:09,  8.65s/it]2026-08-28 21:21:05,562 - INFO - Epoch 93 | Train Loss: 0.2910, Train Acc: 0.8727 | Val Loss: 0.4106
2026-08-28 21:21:05,562 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8565, AUC = 0.9461

 93%|█████████▎| 93/100 [13:21<01:00,  8.64s/it]2026-08-28 21:21:14,397 - INFO - Epoch 94 | Train Loss: 0.2818, Train Acc: 0.8878 | Val Loss: 0.4139
2026-08-28 21:21:14,397 - INFO - Val metrics: Acc = 0.8456 | F1 = 0.8415, AUC = 0.9472

 94%|█████████▍| 94/100 [13:30<00:52,  8.70s/it]2026-08-28 21:21:23,044 - INFO - Epoch 95 | Train Loss: 0.2454, Train Acc: 0.8961 | Val Loss: 0.4850
2026-08-28 21:21:23,044 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7262, AUC = 0.9427

 95%|█████████▌| 95/100 [13:39<00:43,  8.68s/it]2026-08-28 21:21:31,811 - INFO - Epoch 96 | Train Loss: 0.2884, Train Acc: 0.8811 | Val Loss: 0.4611
2026-08-28 21:21:31,811 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7664, AUC = 0.9402

 96%|█████████▌| 96/100 [13:47<00:34,  8.71s/it]2026-08-28 21:21:40,388 - INFO - Epoch 97 | Train Loss: 0.2889, Train Acc: 0.8945 | Val Loss: 0.5099
2026-08-28 21:21:40,388 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.6838, AUC = 0.9329

 97%|█████████▋| 97/100 [13:56<00:26,  8.67s/it]2026-08-28 21:21:49,018 - INFO - Epoch 98 | Train Loss: 0.2708, Train Acc: 0.8911 | Val Loss: 0.4480
2026-08-28 21:21:49,019 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7742, AUC = 0.9418

 98%|█████████▊| 98/100 [14:05<00:17,  8.66s/it]2026-08-28 21:21:57,691 - INFO - Epoch 99 | Train Loss: 0.2844, Train Acc: 0.8794 | Val Loss: 0.4258
2026-08-28 21:21:57,691 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8122, AUC = 0.9429

 99%|█████████▉| 99/100 [14:13<00:08,  8.66s/it]2026-08-28 21:22:06,295 - INFO - Epoch 100 | Train Loss: 0.2903, Train Acc: 0.8744 | Val Loss: 0.4765
2026-08-28 21:22:06,295 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7179, AUC = 0.9405

100%|██████████| 100/100 [14:22<00:00,  8.62s/it]
2026-08-28 21:22:06,295 - INFO -
--- FOLD 5 ---

2026-08-28 21:22:13,024 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 64, heads=4)
  (residual_proj1): Linear(in_features=10, out_features=256, bias=True)
  (residual_proj2): Linear(in_features=256, out_features=64, bias=True)
  (gat2): GATv2Conv(256, 64, heads=1)
  (bn1): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=2, bias=True)
  )
)

  0%|          | 0/100 [00:00<?, ?it/s]2026-08-28 21:22:21,782 - INFO - Epoch 01 | Train Loss: 0.7273, Train Acc: 0.5461 | Val Loss: 0.6688
2026-08-28 21:22:21,783 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6130, AUC = 0.6274

2026-08-28 21:22:21,783 - INFO - Loss IMPROVEMENT!

  1%|          | 1/100 [00:08<14:29,  8.78s/it]2026-08-28 21:22:30,563 - INFO - Epoch 02 | Train Loss: 0.6826, Train Acc: 0.5511 | Val Loss: 0.6693
2026-08-28 21:22:30,563 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.5852, AUC = 0.6370

  2%|▏         | 2/100 [00:17<14:19,  8.77s/it]2026-08-28 21:22:39,199 - INFO - Epoch 03 | Train Loss: 0.6779, Train Acc: 0.5913 | Val Loss: 0.6592
2026-08-28 21:22:39,200 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.6093, AUC = 0.6570

2026-08-28 21:22:39,200 - INFO - Loss IMPROVEMENT!

  3%|▎         | 3/100 [00:26<14:04,  8.71s/it]2026-08-28 21:22:47,922 - INFO - Epoch 04 | Train Loss: 0.6697, Train Acc: 0.6315 | Val Loss: 0.6602
2026-08-28 21:22:47,922 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6150, AUC = 0.6528

  4%|▍         | 4/100 [00:34<13:56,  8.71s/it]2026-08-28 21:22:56,665 - INFO - Epoch 05 | Train Loss: 0.6720, Train Acc: 0.5980 | Val Loss: 0.6657
2026-08-28 21:22:56,665 - INFO - Val metrics: Acc = 0.6242 | F1 = 0.6114, AUC = 0.6608

  5%|▌         | 5/100 [00:43<13:48,  8.72s/it]2026-08-28 21:23:05,250 - INFO - Epoch 06 | Train Loss: 0.6513, Train Acc: 0.6332 | Val Loss: 0.6582
2026-08-28 21:23:05,250 - INFO - Val metrics: Acc = 0.5973 | F1 = 0.5973, AUC = 0.6509

2026-08-28 21:23:05,250 - INFO - Loss IMPROVEMENT!

  6%|▌         | 6/100 [00:52<13:35,  8.68s/it]2026-08-28 21:23:13,880 - INFO - Epoch 07 | Train Loss: 0.6687, Train Acc: 0.5879 | Val Loss: 0.6513
2026-08-28 21:23:13,880 - INFO - Val metrics: Acc = 0.6242 | F1 = 0.6228, AUC = 0.6743

2026-08-28 21:23:13,880 - INFO - Loss IMPROVEMENT!

  7%|▋         | 7/100 [01:00<13:25,  8.66s/it]2026-08-28 21:23:22,568 - INFO - Epoch 08 | Train Loss: 0.6664, Train Acc: 0.6198 | Val Loss: 0.6605
2026-08-28 21:23:22,568 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6401, AUC = 0.6763

  8%|▊         | 8/100 [01:09<13:17,  8.67s/it]2026-08-28 21:23:31,219 - INFO - Epoch 09 | Train Loss: 0.6567, Train Acc: 0.5930 | Val Loss: 0.6576
2026-08-28 21:23:31,219 - INFO - Val metrics: Acc = 0.6040 | F1 = 0.5895, AUC = 0.6752

  9%|▉         | 9/100 [01:18<13:08,  8.66s/it]2026-08-28 21:23:39,832 - INFO - Epoch 10 | Train Loss: 0.6535, Train Acc: 0.6248 | Val Loss: 0.6592
2026-08-28 21:23:39,833 - INFO - Val metrics: Acc = 0.6242 | F1 = 0.6203, AUC = 0.6709

 10%|█         | 10/100 [01:26<12:58,  8.65s/it]2026-08-28 21:23:48,481 - INFO - Epoch 11 | Train Loss: 0.6379, Train Acc: 0.6348 | Val Loss: 0.6521
2026-08-28 21:23:48,481 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6572, AUC = 0.6723

 11%|█         | 11/100 [01:35<12:49,  8.65s/it]2026-08-28 21:23:57,124 - INFO - Epoch 12 | Train Loss: 0.6316, Train Acc: 0.6683 | Val Loss: 0.6488
2026-08-28 21:23:57,124 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6572, AUC = 0.6696

2026-08-28 21:23:57,124 - INFO - Loss IMPROVEMENT!

 12%|█▏        | 12/100 [01:44<12:41,  8.65s/it]2026-08-28 21:24:05,851 - INFO - Epoch 13 | Train Loss: 0.6543, Train Acc: 0.6231 | Val Loss: 0.6634
2026-08-28 21:24:05,851 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6157, AUC = 0.6865

 13%|█▎        | 13/100 [01:52<12:34,  8.67s/it]2026-08-28 21:24:14,588 - INFO - Epoch 14 | Train Loss: 0.6312, Train Acc: 0.6633 | Val Loss: 0.6717
2026-08-28 21:24:14,588 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6193, AUC = 0.6839

 14%|█▍        | 14/100 [02:01<12:27,  8.69s/it]2026-08-28 21:24:23,273 - INFO - Epoch 15 | Train Loss: 0.6555, Train Acc: 0.6332 | Val Loss: 0.6476
2026-08-28 21:24:23,273 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6509, AUC = 0.6947

2026-08-28 21:24:23,274 - INFO - Loss IMPROVEMENT!

 15%|█▌        | 15/100 [02:10<12:18,  8.69s/it]2026-08-28 21:24:31,942 - INFO - Epoch 16 | Train Loss: 0.6442, Train Acc: 0.6315 | Val Loss: 0.6490
2026-08-28 21:24:31,943 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.6043, AUC = 0.6916

 16%|█▌        | 16/100 [02:18<12:09,  8.68s/it]2026-08-28 21:24:40,566 - INFO - Epoch 17 | Train Loss: 0.6368, Train Acc: 0.6533 | Val Loss: 0.6535
2026-08-28 21:24:40,566 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6626, AUC = 0.6939

 17%|█▋        | 17/100 [02:27<11:59,  8.66s/it]2026-08-28 21:24:49,253 - INFO - Epoch 18 | Train Loss: 0.6284, Train Acc: 0.6482 | Val Loss: 0.6552
2026-08-28 21:24:49,253 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6589, AUC = 0.6916

 18%|█▊        | 18/100 [02:36<11:51,  8.67s/it]2026-08-28 21:24:57,972 - INFO - Epoch 19 | Train Loss: 0.6344, Train Acc: 0.6332 | Val Loss: 0.6524
2026-08-28 21:24:57,972 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6697, AUC = 0.6903

 19%|█▉        | 19/100 [02:44<11:43,  8.69s/it]2026-08-28 21:25:06,668 - INFO - Epoch 20 | Train Loss: 0.6393, Train Acc: 0.6399 | Val Loss: 0.6616
2026-08-28 21:25:06,668 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6470, AUC = 0.6981

 20%|██        | 20/100 [02:53<11:35,  8.69s/it]2026-08-28 21:25:15,360 - INFO - Epoch 21 | Train Loss: 0.6399, Train Acc: 0.6667 | Val Loss: 0.6426
2026-08-28 21:25:15,360 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6754, AUC = 0.7076

2026-08-28 21:25:15,360 - INFO - Loss IMPROVEMENT!

 21%|██        | 21/100 [03:02<11:26,  8.69s/it]2026-08-28 21:25:24,032 - INFO - Epoch 22 | Train Loss: 0.6216, Train Acc: 0.6566 | Val Loss: 0.6406
2026-08-28 21:25:24,032 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6901, AUC = 0.6994

2026-08-28 21:25:24,032 - INFO - Loss IMPROVEMENT!

 22%|██▏       | 22/100 [03:11<11:17,  8.69s/it]2026-08-28 21:25:32,782 - INFO - Epoch 23 | Train Loss: 0.6287, Train Acc: 0.6817 | Val Loss: 0.6678
2026-08-28 21:25:32,782 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6293, AUC = 0.7004

 23%|██▎       | 23/100 [03:19<11:10,  8.70s/it]2026-08-28 21:25:41,485 - INFO - Epoch 24 | Train Loss: 0.6268, Train Acc: 0.6616 | Val Loss: 0.6351
2026-08-28 21:25:41,485 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6706, AUC = 0.7037

2026-08-28 21:25:41,485 - INFO - Loss IMPROVEMENT!

 24%|██▍       | 24/100 [03:28<11:01,  8.71s/it]2026-08-28 21:25:50,190 - INFO - Epoch 25 | Train Loss: 0.6263, Train Acc: 0.6499 | Val Loss: 0.6651
2026-08-28 21:25:50,190 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6248, AUC = 0.7078

 25%|██▌       | 25/100 [03:37<10:52,  8.70s/it]2026-08-28 21:25:58,806 - INFO - Epoch 26 | Train Loss: 0.6352, Train Acc: 0.6432 | Val Loss: 0.6448
2026-08-28 21:25:58,806 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6841, AUC = 0.7017

 26%|██▌       | 26/100 [03:45<10:42,  8.68s/it]2026-08-28 21:26:07,374 - INFO - Epoch 27 | Train Loss: 0.6327, Train Acc: 0.6566 | Val Loss: 0.6427
2026-08-28 21:26:07,374 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6771, AUC = 0.6988

 27%|██▋       | 27/100 [03:54<10:31,  8.64s/it]2026-08-28 21:26:16,136 - INFO - Epoch 28 | Train Loss: 0.6179, Train Acc: 0.6650 | Val Loss: 0.6529
2026-08-28 21:26:16,136 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6271, AUC = 0.7183

 28%|██▊       | 28/100 [04:03<10:24,  8.68s/it]2026-08-28 21:26:24,785 - INFO - Epoch 29 | Train Loss: 0.6190, Train Acc: 0.6884 | Val Loss: 0.6381
2026-08-28 21:26:24,786 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6881, AUC = 0.7109

 29%|██▉       | 29/100 [04:11<10:15,  8.67s/it]2026-08-28 21:26:33,685 - INFO - Epoch 30 | Train Loss: 0.6086, Train Acc: 0.6683 | Val Loss: 0.6363
2026-08-28 21:26:33,685 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6778, AUC = 0.7112

 30%|███       | 30/100 [04:20<10:11,  8.74s/it]2026-08-28 21:26:42,397 - INFO - Epoch 31 | Train Loss: 0.6379, Train Acc: 0.6616 | Val Loss: 0.6471
2026-08-28 21:26:42,397 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6913, AUC = 0.7222

 31%|███       | 31/100 [04:29<10:02,  8.73s/it]2026-08-28 21:26:51,079 - INFO - Epoch 32 | Train Loss: 0.6067, Train Acc: 0.6918 | Val Loss: 0.6388
2026-08-28 21:26:51,079 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6775, AUC = 0.7206

 32%|███▏      | 32/100 [04:38<09:52,  8.72s/it]2026-08-28 21:26:59,728 - INFO - Epoch 33 | Train Loss: 0.6182, Train Acc: 0.6616 | Val Loss: 0.6247
2026-08-28 21:26:59,728 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7178, AUC = 0.7304

2026-08-28 21:26:59,728 - INFO - Loss IMPROVEMENT!

 33%|███▎      | 33/100 [04:46<09:42,  8.70s/it]2026-08-28 21:27:08,493 - INFO - Epoch 34 | Train Loss: 0.6134, Train Acc: 0.6951 | Val Loss: 0.6203
2026-08-28 21:27:08,493 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6975, AUC = 0.7331

2026-08-28 21:27:08,493 - INFO - Loss IMPROVEMENT!

 34%|███▍      | 34/100 [04:55<09:35,  8.72s/it]2026-08-28 21:27:17,247 - INFO - Epoch 35 | Train Loss: 0.6031, Train Acc: 0.6817 | Val Loss: 0.6052
2026-08-28 21:27:17,248 - INFO - Val metrics: Acc = 0.7248 | F1 = 0.7240, AUC = 0.7386

2026-08-28 21:27:17,248 - INFO - Loss IMPROVEMENT!

 35%|███▌      | 35/100 [05:04<09:27,  8.73s/it]2026-08-28 21:27:25,875 - INFO - Epoch 36 | Train Loss: 0.5785, Train Acc: 0.7085 | Val Loss: 0.6005
2026-08-28 21:27:25,875 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.7046, AUC = 0.7502

2026-08-28 21:27:25,875 - INFO - Loss IMPROVEMENT!

 36%|███▌      | 36/100 [05:12<09:16,  8.70s/it]2026-08-28 21:27:34,584 - INFO - Epoch 37 | Train Loss: 0.6208, Train Acc: 0.6750 | Val Loss: 0.6134
2026-08-28 21:27:34,584 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7315, AUC = 0.7520

 37%|███▋      | 37/100 [05:21<09:08,  8.70s/it]2026-08-28 21:27:43,365 - INFO - Epoch 38 | Train Loss: 0.6002, Train Acc: 0.6918 | Val Loss: 0.6143
2026-08-28 21:27:43,365 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.6975, AUC = 0.7626

 38%|███▊      | 38/100 [05:30<09:00,  8.72s/it]2026-08-28 21:27:52,205 - INFO - Epoch 39 | Train Loss: 0.5812, Train Acc: 0.6968 | Val Loss: 0.5964
2026-08-28 21:27:52,205 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.7061, AUC = 0.7695

2026-08-28 21:27:52,205 - INFO - Loss IMPROVEMENT!

 39%|███▉      | 39/100 [05:39<08:54,  8.76s/it]2026-08-28 21:28:01,007 - INFO - Epoch 40 | Train Loss: 0.5882, Train Acc: 0.6884 | Val Loss: 0.6027
2026-08-28 21:28:01,007 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7371, AUC = 0.7731

 40%|████      | 40/100 [05:47<08:46,  8.77s/it]2026-08-28 21:28:09,769 - INFO - Epoch 41 | Train Loss: 0.5832, Train Acc: 0.7035 | Val Loss: 0.6316
2026-08-28 21:28:09,769 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6136, AUC = 0.7763

 41%|████      | 41/100 [05:56<08:37,  8.77s/it]2026-08-28 21:28:18,375 - INFO - Epoch 42 | Train Loss: 0.5772, Train Acc: 0.6935 | Val Loss: 0.5969
2026-08-28 21:28:18,375 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.7061, AUC = 0.7826

 42%|████▏     | 42/100 [06:05<08:25,  8.72s/it]2026-08-28 21:28:27,125 - INFO - Epoch 43 | Train Loss: 0.5896, Train Acc: 0.6868 | Val Loss: 0.6038
2026-08-28 21:28:27,125 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6899, AUC = 0.7812

 43%|████▎     | 43/100 [06:14<08:17,  8.73s/it]2026-08-28 21:28:35,825 - INFO - Epoch 44 | Train Loss: 0.6065, Train Acc: 0.6616 | Val Loss: 0.6106
2026-08-28 21:28:35,825 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.6998, AUC = 0.7805

 44%|████▍     | 44/100 [06:22<08:08,  8.72s/it]2026-08-28 21:28:44,487 - INFO - Epoch 45 | Train Loss: 0.5857, Train Acc: 0.6884 | Val Loss: 0.5921
2026-08-28 21:28:44,487 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.7050, AUC = 0.7922

2026-08-28 21:28:44,487 - INFO - Loss IMPROVEMENT!

 45%|████▌     | 45/100 [06:31<07:58,  8.70s/it]2026-08-28 21:28:53,259 - INFO - Epoch 46 | Train Loss: 0.5650, Train Acc: 0.7035 | Val Loss: 0.5648
2026-08-28 21:28:53,259 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7651, AUC = 0.7981

2026-08-28 21:28:53,259 - INFO - Loss IMPROVEMENT!

 46%|████▌     | 46/100 [06:40<07:51,  8.73s/it]2026-08-28 21:29:02,003 - INFO - Epoch 47 | Train Loss: 0.5570, Train Acc: 0.7303 | Val Loss: 0.5730
2026-08-28 21:29:02,003 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6935, AUC = 0.7981

 47%|████▋     | 47/100 [06:48<07:42,  8.73s/it]2026-08-28 21:29:10,837 - INFO - Epoch 48 | Train Loss: 0.5594, Train Acc: 0.7236 | Val Loss: 0.5880
2026-08-28 21:29:10,838 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6791, AUC = 0.7974

 48%|████▊     | 48/100 [06:57<07:35,  8.76s/it]2026-08-28 21:29:19,571 - INFO - Epoch 49 | Train Loss: 0.5460, Train Acc: 0.7370 | Val Loss: 0.5978
2026-08-28 21:29:19,572 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6732, AUC = 0.8138

 49%|████▉     | 49/100 [07:06<07:26,  8.75s/it]2026-08-28 21:29:28,384 - INFO - Epoch 50 | Train Loss: 0.5438, Train Acc: 0.7219 | Val Loss: 0.5636
2026-08-28 21:29:28,384 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7608, AUC = 0.8378

2026-08-28 21:29:28,384 - INFO - Loss IMPROVEMENT!

 50%|█████     | 50/100 [07:15<07:18,  8.77s/it]2026-08-28 21:29:37,174 - INFO - Epoch 51 | Train Loss: 0.5185, Train Acc: 0.7404 | Val Loss: 0.5840
2026-08-28 21:29:37,174 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6601, AUC = 0.8354

 51%|█████     | 51/100 [07:24<07:10,  8.78s/it]2026-08-28 21:29:45,782 - INFO - Epoch 52 | Train Loss: 0.5386, Train Acc: 0.7119 | Val Loss: 0.5316
2026-08-28 21:29:45,782 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7848, AUC = 0.8518

2026-08-28 21:29:45,782 - INFO - Loss IMPROVEMENT!

 52%|█████▏    | 52/100 [07:32<06:58,  8.73s/it]2026-08-28 21:29:54,470 - INFO - Epoch 53 | Train Loss: 0.5191, Train Acc: 0.7454 | Val Loss: 0.5066
2026-08-28 21:29:54,471 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7986, AUC = 0.8672

2026-08-28 21:29:54,471 - INFO - Loss IMPROVEMENT!

 53%|█████▎    | 53/100 [07:41<06:49,  8.72s/it]2026-08-28 21:30:03,274 - INFO - Epoch 54 | Train Loss: 0.5412, Train Acc: 0.7370 | Val Loss: 0.5107
2026-08-28 21:30:03,274 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7916, AUC = 0.8657

 54%|█████▍    | 54/100 [07:50<06:42,  8.74s/it]2026-08-28 21:30:12,028 - INFO - Epoch 55 | Train Loss: 0.5073, Train Acc: 0.7454 | Val Loss: 0.5467
2026-08-28 21:30:12,028 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7210, AUC = 0.8908

 55%|█████▌    | 55/100 [07:59<06:33,  8.74s/it]2026-08-28 21:30:20,791 - INFO - Epoch 56 | Train Loss: 0.5017, Train Acc: 0.7521 | Val Loss: 0.4880
2026-08-28 21:30:20,791 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7844, AUC = 0.8682

2026-08-28 21:30:20,791 - INFO - Loss IMPROVEMENT!

 56%|█████▌    | 56/100 [08:07<06:25,  8.75s/it]2026-08-28 21:30:29,543 - INFO - Epoch 57 | Train Loss: 0.5638, Train Acc: 0.7387 | Val Loss: 0.5729
2026-08-28 21:30:29,544 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6793, AUC = 0.8731

 57%|█████▋    | 57/100 [08:16<06:16,  8.75s/it]2026-08-28 21:30:38,343 - INFO - Epoch 58 | Train Loss: 0.5381, Train Acc: 0.7370 | Val Loss: 0.4777
2026-08-28 21:30:38,344 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8051, AUC = 0.9174

2026-08-28 21:30:38,344 - INFO - Loss IMPROVEMENT!

 58%|█████▊    | 58/100 [08:25<06:08,  8.77s/it]2026-08-28 21:30:47,085 - INFO - Epoch 59 | Train Loss: 0.4678, Train Acc: 0.7789 | Val Loss: 0.4911
2026-08-28 21:30:47,085 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7759, AUC = 0.8787

 59%|█████▉    | 59/100 [08:34<05:59,  8.76s/it]2026-08-28 21:30:55,843 - INFO - Epoch 60 | Train Loss: 0.5098, Train Acc: 0.7705 | Val Loss: 0.4652
2026-08-28 21:30:55,843 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8251, AUC = 0.8994

2026-08-28 21:30:55,844 - INFO - Loss IMPROVEMENT!

 60%|██████    | 60/100 [08:42<05:50,  8.76s/it]2026-08-28 21:31:04,656 - INFO - Epoch 61 | Train Loss: 0.4690, Train Acc: 0.7672 | Val Loss: 0.4641
2026-08-28 21:31:04,656 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8518, AUC = 0.9176

2026-08-28 21:31:04,657 - INFO - Loss IMPROVEMENT!

 61%|██████    | 61/100 [08:51<05:42,  8.78s/it]2026-08-28 21:31:13,890 - INFO - Epoch 62 | Train Loss: 0.4702, Train Acc: 0.7755 | Val Loss: 0.4424
2026-08-28 21:31:13,890 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8386, AUC = 0.9142

2026-08-28 21:31:13,890 - INFO - Loss IMPROVEMENT!

 62%|██████▏   | 62/100 [09:00<05:38,  8.91s/it]2026-08-28 21:31:23,130 - INFO - Epoch 63 | Train Loss: 0.5044, Train Acc: 0.7588 | Val Loss: 0.5650
2026-08-28 21:31:23,130 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6660, AUC = 0.8726

 63%|██████▎   | 63/100 [09:10<05:33,  9.01s/it]2026-08-28 21:31:32,564 - INFO - Epoch 64 | Train Loss: 0.4583, Train Acc: 0.7889 | Val Loss: 0.4339
2026-08-28 21:31:32,564 - INFO - Val metrics: Acc = 0.8792 | F1 = 0.8792, AUC = 0.9284

2026-08-28 21:31:32,564 - INFO - Loss IMPROVEMENT!

 64%|██████▍   | 64/100 [09:19<05:28,  9.14s/it]2026-08-28 21:31:41,530 - INFO - Epoch 65 | Train Loss: 0.4483, Train Acc: 0.7956 | Val Loss: 0.4776
2026-08-28 21:31:41,530 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8025, AUC = 0.9079

 65%|██████▌   | 65/100 [09:28<05:17,  9.08s/it]2026-08-28 21:31:50,386 - INFO - Epoch 66 | Train Loss: 0.4465, Train Acc: 0.7889 | Val Loss: 0.4341
2026-08-28 21:31:50,386 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8657, AUC = 0.9362

 66%|██████▌   | 66/100 [09:37<05:06,  9.02s/it]2026-08-28 21:31:59,094 - INFO - Epoch 67 | Train Loss: 0.4433, Train Acc: 0.7940 | Val Loss: 0.4538
2026-08-28 21:31:59,094 - INFO - Val metrics: Acc = 0.8456 | F1 = 0.8454, AUC = 0.9165

 67%|██████▋   | 67/100 [09:46<04:54,  8.92s/it]2026-08-28 21:32:07,893 - INFO - Epoch 68 | Train Loss: 0.4522, Train Acc: 0.7990 | Val Loss: 0.5253
2026-08-28 21:32:07,894 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6691, AUC = 0.9191

 68%|██████▊   | 68/100 [09:54<04:44,  8.89s/it]2026-08-28 21:32:16,902 - INFO - Epoch 69 | Train Loss: 0.4495, Train Acc: 0.8090 | Val Loss: 0.4524
2026-08-28 21:32:16,903 - INFO - Val metrics: Acc = 0.8792 | F1 = 0.8792, AUC = 0.9441

 69%|██████▉   | 69/100 [10:03<04:36,  8.92s/it]2026-08-28 21:32:25,755 - INFO - Epoch 70 | Train Loss: 0.4221, Train Acc: 0.8023 | Val Loss: 0.4853
2026-08-28 21:32:25,755 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8083, AUC = 0.9146

 70%|███████   | 70/100 [10:12<04:27,  8.90s/it]2026-08-28 21:32:34,470 - INFO - Epoch 71 | Train Loss: 0.4234, Train Acc: 0.8074 | Val Loss: 0.4458
2026-08-28 21:32:34,471 - INFO - Val metrics: Acc = 0.8322 | F1 = 0.8311, AUC = 0.9126

 71%|███████   | 71/100 [10:21<04:16,  8.85s/it]2026-08-28 21:32:43,237 - INFO - Epoch 72 | Train Loss: 0.4027, Train Acc: 0.8174 | Val Loss: 0.5543
2026-08-28 21:32:43,238 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6445, AUC = 0.9331

 72%|███████▏  | 72/100 [10:30<04:07,  8.82s/it]2026-08-28 21:32:51,905 - INFO - Epoch 73 | Train Loss: 0.4040, Train Acc: 0.8074 | Val Loss: 0.4965
2026-08-28 21:32:51,906 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7488, AUC = 0.9499

 73%|███████▎  | 73/100 [10:38<03:56,  8.78s/it]2026-08-28 21:33:00,601 - INFO - Epoch 74 | Train Loss: 0.3947, Train Acc: 0.8074 | Val Loss: 0.4698
2026-08-28 21:33:00,601 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7443, AUC = 0.9367

 74%|███████▍  | 74/100 [10:47<03:47,  8.75s/it]2026-08-28 21:33:09,326 - INFO - Epoch 75 | Train Loss: 0.4362, Train Acc: 0.8007 | Val Loss: 0.6938
2026-08-28 21:33:09,326 - INFO - Val metrics: Acc = 0.5369 | F1 = 0.3958, AUC = 0.8866

 75%|███████▌  | 75/100 [10:56<03:38,  8.74s/it]2026-08-28 21:33:18,185 - INFO - Epoch 76 | Train Loss: 0.4433, Train Acc: 0.8007 | Val Loss: 0.4865
2026-08-28 21:33:18,185 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7907, AUC = 0.9241

 76%|███████▌  | 76/100 [11:05<03:30,  8.78s/it]2026-08-28 21:33:26,929 - INFO - Epoch 77 | Train Loss: 0.3887, Train Acc: 0.8308 | Val Loss: 0.5260
2026-08-28 21:33:26,929 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7179, AUC = 0.9153

 77%|███████▋  | 77/100 [11:13<03:21,  8.77s/it]2026-08-28 21:33:35,743 - INFO - Epoch 78 | Train Loss: 0.3763, Train Acc: 0.8409 | Val Loss: 0.4035
2026-08-28 21:33:35,743 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8650, AUC = 0.9461

2026-08-28 21:33:35,743 - INFO - Loss IMPROVEMENT!

 78%|███████▊  | 78/100 [11:22<03:13,  8.78s/it]2026-08-28 21:33:44,420 - INFO - Epoch 79 | Train Loss: 0.3611, Train Acc: 0.8492 | Val Loss: 0.5500
2026-08-28 21:33:44,420 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6480, AUC = 0.9194

 79%|███████▉  | 79/100 [11:31<03:03,  8.75s/it]2026-08-28 21:33:53,180 - INFO - Epoch 80 | Train Loss: 0.3657, Train Acc: 0.8425 | Val Loss: 0.5066
2026-08-28 21:33:53,180 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.6925, AUC = 0.9277

 80%|████████  | 80/100 [11:40<02:55,  8.75s/it]2026-08-28 21:34:01,916 - INFO - Epoch 81 | Train Loss: 0.3424, Train Acc: 0.8459 | Val Loss: 0.4813
2026-08-28 21:34:01,916 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7425, AUC = 0.9308

 81%|████████  | 81/100 [11:48<02:46,  8.75s/it]2026-08-28 21:34:10,696 - INFO - Epoch 82 | Train Loss: 0.3343, Train Acc: 0.8526 | Val Loss: 0.4912
2026-08-28 21:34:10,696 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7262, AUC = 0.9284

 82%|████████▏ | 82/100 [11:57<02:37,  8.76s/it]2026-08-28 21:34:19,497 - INFO - Epoch 83 | Train Loss: 0.3603, Train Acc: 0.8576 | Val Loss: 0.4870
2026-08-28 21:34:19,497 - INFO - Val metrics: Acc = 0.7718 | F1 = 0.7585, AUC = 0.9277

 83%|████████▎ | 83/100 [12:06<02:29,  8.77s/it]2026-08-28 21:34:28,131 - INFO - Epoch 84 | Train Loss: 0.2911, Train Acc: 0.8811 | Val Loss: 0.3960
2026-08-28 21:34:28,132 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8499, AUC = 0.9367

2026-08-28 21:34:28,132 - INFO - Loss IMPROVEMENT!

 84%|████████▍ | 84/100 [12:15<02:19,  8.73s/it]2026-08-28 21:34:36,848 - INFO - Epoch 85 | Train Loss: 0.3633, Train Acc: 0.8509 | Val Loss: 0.5195
2026-08-28 21:34:36,848 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.6838, AUC = 0.9346

 85%|████████▌ | 85/100 [12:23<02:10,  8.73s/it]2026-08-28 21:34:45,568 - INFO - Epoch 86 | Train Loss: 0.3152, Train Acc: 0.8693 | Val Loss: 0.3803
2026-08-28 21:34:45,568 - INFO - Val metrics: Acc = 0.8859 | F1 = 0.8854, AUC = 0.9506

2026-08-28 21:34:45,568 - INFO - Loss IMPROVEMENT!

 86%|████████▌ | 86/100 [12:32<02:02,  8.73s/it]2026-08-28 21:34:54,136 - INFO - Epoch 87 | Train Loss: 0.3232, Train Acc: 0.8543 | Val Loss: 0.4432
2026-08-28 21:34:54,137 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7895, AUC = 0.9506

 87%|████████▋ | 87/100 [12:41<01:52,  8.68s/it]2026-08-28 21:35:02,831 - INFO - Epoch 88 | Train Loss: 0.3436, Train Acc: 0.8476 | Val Loss: 0.4836
2026-08-28 21:35:02,831 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7363, AUC = 0.9373

 88%|████████▊ | 88/100 [12:49<01:44,  8.68s/it]2026-08-28 21:35:11,494 - INFO - Epoch 89 | Train Loss: 0.3472, Train Acc: 0.8559 | Val Loss: 0.4545
2026-08-28 21:35:11,494 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.7982, AUC = 0.9429

 89%|████████▉ | 89/100 [12:58<01:35,  8.68s/it]2026-08-28 21:35:20,173 - INFO - Epoch 90 | Train Loss: 0.3108, Train Acc: 0.8693 | Val Loss: 0.5896
2026-08-28 21:35:20,173 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.5755, AUC = 0.9329

 90%|█████████ | 90/100 [13:07<01:26,  8.68s/it]2026-08-28 21:35:28,816 - INFO - Epoch 91 | Train Loss: 0.3405, Train Acc: 0.8459 | Val Loss: 0.4651
2026-08-28 21:35:28,816 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7895, AUC = 0.9439

 91%|█████████ | 91/100 [13:15<01:18,  8.67s/it]2026-08-28 21:35:37,512 - INFO - Epoch 92 | Train Loss: 0.3399, Train Acc: 0.8492 | Val Loss: 0.4456
2026-08-28 21:35:37,512 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8357, AUC = 0.9405

 92%|█████████▏| 92/100 [13:24<01:09,  8.68s/it]2026-08-28 21:35:46,028 - INFO - Epoch 93 | Train Loss: 0.2750, Train Acc: 0.8794 | Val Loss: 0.5509
2026-08-28 21:35:46,028 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6351, AUC = 0.9371

 93%|█████████▎| 93/100 [13:33<01:00,  8.63s/it]2026-08-28 21:35:54,613 - INFO - Epoch 94 | Train Loss: 0.2873, Train Acc: 0.8928 | Val Loss: 0.5519
2026-08-28 21:35:54,613 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.6721, AUC = 0.9402

 94%|█████████▍| 94/100 [13:41<00:51,  8.62s/it]2026-08-28 21:36:03,281 - INFO - Epoch 95 | Train Loss: 0.3100, Train Acc: 0.8626 | Val Loss: 0.5272
2026-08-28 21:36:03,281 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6538, AUC = 0.9439

 95%|█████████▌| 95/100 [13:50<00:43,  8.63s/it]2026-08-28 21:36:11,857 - INFO - Epoch 96 | Train Loss: 0.2919, Train Acc: 0.8811 | Val Loss: 0.4589
2026-08-28 21:36:11,857 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7505, AUC = 0.9450

 96%|█████████▌| 96/100 [13:58<00:34,  8.61s/it]2026-08-28 21:36:20,551 - INFO - Epoch 97 | Train Loss: 0.3289, Train Acc: 0.8576 | Val Loss: 0.4652
2026-08-28 21:36:20,551 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7363, AUC = 0.9434

 97%|█████████▋| 97/100 [14:07<00:25,  8.64s/it]2026-08-28 21:36:29,150 - INFO - Epoch 98 | Train Loss: 0.2943, Train Acc: 0.8777 | Val Loss: 0.4658
2026-08-28 21:36:29,150 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7755, AUC = 0.9346

 98%|█████████▊| 98/100 [14:16<00:17,  8.63s/it]2026-08-28 21:36:37,759 - INFO - Epoch 99 | Train Loss: 0.2875, Train Acc: 0.8844 | Val Loss: 0.5154
2026-08-28 21:36:37,759 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6572, AUC = 0.9306

 99%|█████████▉| 99/100 [14:24<00:08,  8.62s/it]2026-08-28 21:36:46,330 - INFO - Epoch 100 | Train Loss: 0.2763, Train Acc: 0.8894 | Val Loss: 0.5044
2026-08-28 21:36:46,330 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.6925, AUC = 0.9320

100%|██████████| 100/100 [14:33<00:00,  8.73s/it]

"""








"""
GATv2Conv anziché GATConv: Viene usata la versione "v2" di GAT (GATv2Conv).
La GAT classica soffre di una limitazione teorica (il meccanismo di attenzione collassa a una
semplice media pesata se le query non sono dinamiche); GATv2 risolve questo bug rendendo
l'attenzione strettamente dipendente dalla combinazione di feature di entrambi i nodi connessi.

Layer Normalization & Residual Connections (nn.LayerNorm): Le reti basate su grafi e meccanismi 
di attenzione a molte teste sono soggette a instabilità durante il backpropagation.
L'aggiunta di una proiezione residua (residual_proj) e della Layer Normalization impedisce al 
gradiente di svanire.

Flessibilità Nodo vs Grafo:

Se il tuo obiettivo è fare Node Classification (es. identificare quali geni sono "driver" della
metastasi), la rete restituisce node_embeddings di dimensione [Num_Geni, Hidden_Dim].

Se il tuo obiettivo è fare Graph Classification (es. distinguere se il campione del paziente è un
primario o una metastasi), il layer global_mean_pool comprime la matrice dei geni in un unico 
vettore rappresentativo del paziente.
"""