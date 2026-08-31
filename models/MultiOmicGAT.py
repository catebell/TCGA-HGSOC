import torch
import torch.nn.functional as F

from torch.nn import Dropout, Module, Linear, Sequential, ReLU
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, GraphNorm, InstanceNorm


# GATv2Conv: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html
# solves static attention problem mentioned in the original paper

class MultiOmicGAT(Module):
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
        self.residual_proj1 = Linear(in_node_features, hidden_dim * heads)
        self.residual_proj2 = Linear(hidden_dim * heads, hidden_dim)

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
        self.norm1 = GraphNorm(hidden_dim * heads)
        self.norm2 = GraphNorm(hidden_dim)

        self.num_clinical_features = num_clinical_features

        if num_clinical_features > 0:
            self.clinical_encoder = Sequential(
                Linear(num_clinical_features, 32),
                ReLU(),
                Dropout(p=dropout)
            )

            fusion_dim = hidden_dim * 2 + 32
            total_emb_dim = fusion_dim
        else:
            total_emb_dim = hidden_dim * 2

        # graph classification/prediction
        self.classifier = Sequential(
            Linear(total_emb_dim, hidden_dim // 2),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_dim // 2, out_channels)
        )

        self.dropout_rate = dropout
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, batch=None, clinical_x=None):
        """
        x: Tensor [N_nodes, in_node_features] -> Genes features matrix
        edge_index: Tensor [2, E] -> Adjacency matrix
        batch: Tensor [N_nodes] -> to combine more graphs, observe them together
        """

        # --- LAYER 1 ---
        h_res = self.residual_proj1(x)  # Residual connection
        h = self.gat1(x, edge_index)
        h = self.norm1(h)
        h = h + h_res  # TODO pre-norm vs post-norm?
        h = F.elu(h)  # ELU activation (GAT standard)

        # --- LAYER 2 ---
        h_res = self.residual_proj2(h)
        h = self.gat2(h, edge_index)
        h = self.norm2(h)
        h = h + h_res  # TODO pre-norm vs post-norm?
        h = F.elu(h)

        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        graph_emb = torch.cat([mean_pool, max_pool], dim=1)  # [Batch_Size, hidden_dim * 2]

        graph_emb = self.dropout(graph_emb)

        if self.num_clinical_features > 0 and clinical_x is not None:
            clin_emb = self.clinical_encoder(clinical_x)
            fused_emb = torch.cat([graph_emb, clin_emb], dim=1) # [Batch_Size, hidden_dim * 2 + 32]
            out = self.classifier(fused_emb)
            return out, fused_emb
        else:
            out = self.classifier(graph_emb)
            return out, graph_emb


"""
2026-08-31 18:55:50,597 - INFO - Device: cuda
2026-08-31 18:55:50,597 - INFO - Starting training using target feature: disease_code

2026-08-31 18:55:50,607 - INFO - Train + Val Dataset init...
2026-08-31 18:55:50,623 - INFO - Test Dataset init...
2026-08-31 18:55:50,727 - INFO - Clinical Features found : 3
 ['age_at_initial_pathologic_diagnosis', 'days_to_last_followup', 'postoperative_rx_tx']
2026-08-31 18:55:57,218 - INFO - Unique labels found in dataset: tensor([0, 1])
2026-08-31 18:55:57,346 - INFO - classes_counts:
2026-08-31 18:55:57,346 - INFO - tensor([368, 378])
2026-08-31 18:55:57,347 - INFO - weights:
2026-08-31 18:55:57,347 - INFO - tensor([1.0136, 0.9868], device='cuda:0')
2026-08-31 18:55:57,462 - INFO -
--- FOLD 1 ---

2026-08-31 18:56:04,018 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 64, heads=4)
  (residual_proj1): Linear(in_features=10, out_features=256, bias=True)
  (residual_proj2): Linear(in_features=256, out_features=64, bias=True)
  (gat2): GATv2Conv(256, 64, heads=1)
  (bn1): GraphNorm(256)
  (bn2): GraphNorm(64)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=2, bias=True)
  )
  (dropout): Dropout(p=0.3, inplace=False)
)

  0%|          | 0/100 [00:00<?, ?it/s]2026-08-31 18:56:13,839 - INFO - Epoch 01 | Train Loss: 0.8063, Train Acc: 0.5000 | Val Loss: 0.6591
2026-08-31 18:56:13,839 - INFO - Val metrics: Acc = 0.6067 | F1 = 0.6015, AUC = 0.6613

2026-08-31 18:56:13,840 - INFO - Loss IMPROVEMENT!

  1%|          | 1/100 [00:09<16:12,  9.83s/it]2026-08-31 18:56:23,326 - INFO - Epoch 02 | Train Loss: 0.7042, Train Acc: 0.5419 | Val Loss: 0.6805
2026-08-31 18:56:23,326 - INFO - Val metrics: Acc = 0.6267 | F1 = 0.6111, AUC = 0.6671

  2%|▏         | 2/100 [00:19<15:43,  9.62s/it]2026-08-31 18:56:32,884 - INFO - Epoch 03 | Train Loss: 0.6805, Train Acc: 0.5487 | Val Loss: 0.6690
2026-08-31 18:56:32,884 - INFO - Val metrics: Acc = 0.6333 | F1 = 0.6274, AUC = 0.6778

  3%|▎         | 3/100 [00:28<15:30,  9.59s/it]2026-08-31 18:56:42,363 - INFO - Epoch 04 | Train Loss: 0.6745, Train Acc: 0.5839 | Val Loss: 0.6779
2026-08-31 18:56:42,363 - INFO - Val metrics: Acc = 0.6267 | F1 = 0.6212, AUC = 0.6803

  4%|▍         | 4/100 [00:38<15:16,  9.55s/it]2026-08-31 18:56:51,803 - INFO - Epoch 05 | Train Loss: 0.6801, Train Acc: 0.5940 | Val Loss: 0.6645
2026-08-31 18:56:51,804 - INFO - Val metrics: Acc = 0.6133 | F1 = 0.5924, AUC = 0.6844

  5%|▌         | 5/100 [00:47<15:03,  9.51s/it]2026-08-31 18:57:01,505 - INFO - Epoch 06 | Train Loss: 0.6704, Train Acc: 0.5889 | Val Loss: 0.6555
2026-08-31 18:57:01,505 - INFO - Val metrics: Acc = 0.6133 | F1 = 0.6127, AUC = 0.6903

2026-08-31 18:57:01,505 - INFO - Loss IMPROVEMENT!

  6%|▌         | 6/100 [00:57<15:00,  9.58s/it]2026-08-31 18:57:11,353 - INFO - Epoch 07 | Train Loss: 0.6638, Train Acc: 0.5956 | Val Loss: 0.6539
2026-08-31 18:57:11,353 - INFO - Val metrics: Acc = 0.5933 | F1 = 0.5835, AUC = 0.6876

2026-08-31 18:57:11,354 - INFO - Loss IMPROVEMENT!

  7%|▋         | 7/100 [01:07<14:58,  9.67s/it]2026-08-31 18:57:20,902 - INFO - Epoch 08 | Train Loss: 0.6862, Train Acc: 0.5805 | Val Loss: 0.6628
2026-08-31 18:57:20,902 - INFO - Val metrics: Acc = 0.5733 | F1 = 0.5502, AUC = 0.6957

  8%|▊         | 8/100 [01:16<14:45,  9.63s/it]2026-08-31 18:57:30,458 - INFO - Epoch 09 | Train Loss: 0.6747, Train Acc: 0.5990 | Val Loss: 0.6667
2026-08-31 18:57:30,458 - INFO - Val metrics: Acc = 0.6600 | F1 = 0.6588, AUC = 0.6993

  9%|▉         | 9/100 [01:26<14:34,  9.60s/it]2026-08-31 18:57:40,219 - INFO - Epoch 10 | Train Loss: 0.6668, Train Acc: 0.6174 | Val Loss: 0.6576
2026-08-31 18:57:40,219 - INFO - Val metrics: Acc = 0.6333 | F1 = 0.6329, AUC = 0.6991

 10%|█         | 10/100 [01:36<14:28,  9.65s/it]2026-08-31 18:57:49,861 - INFO - Epoch 11 | Train Loss: 0.6726, Train Acc: 0.6023 | Val Loss: 0.6553
2026-08-31 18:57:49,861 - INFO - Val metrics: Acc = 0.6467 | F1 = 0.6409, AUC = 0.6981

 11%|█         | 11/100 [01:45<14:18,  9.65s/it]2026-08-31 18:57:59,679 - INFO - Epoch 12 | Train Loss: 0.6574, Train Acc: 0.6376 | Val Loss: 0.6517
2026-08-31 18:57:59,680 - INFO - Val metrics: Acc = 0.6467 | F1 = 0.6440, AUC = 0.6956

2026-08-31 18:57:59,680 - INFO - Loss IMPROVEMENT!

 12%|█▏        | 12/100 [01:55<14:13,  9.70s/it]2026-08-31 18:58:09,686 - INFO - Epoch 13 | Train Loss: 0.6627, Train Acc: 0.6174 | Val Loss: 0.6463
2026-08-31 18:58:09,686 - INFO - Val metrics: Acc = 0.6600 | F1 = 0.6556, AUC = 0.7045

2026-08-31 18:58:09,686 - INFO - Loss IMPROVEMENT!

 13%|█▎        | 13/100 [02:05<14:12,  9.80s/it]2026-08-31 18:58:19,130 - INFO - Epoch 14 | Train Loss: 0.6280, Train Acc: 0.6560 | Val Loss: 0.6382
2026-08-31 18:58:19,130 - INFO - Val metrics: Acc = 0.6400 | F1 = 0.6399, AUC = 0.7071

2026-08-31 18:58:19,130 - INFO - Loss IMPROVEMENT!

 14%|█▍        | 14/100 [02:15<13:53,  9.69s/it]2026-08-31 18:58:28,278 - INFO - Epoch 15 | Train Loss: 0.6408, Train Acc: 0.6527 | Val Loss: 0.6285
2026-08-31 18:58:28,278 - INFO - Val metrics: Acc = 0.6333 | F1 = 0.6325, AUC = 0.7118

2026-08-31 18:58:28,278 - INFO - Loss IMPROVEMENT!

 15%|█▌        | 15/100 [02:24<13:29,  9.53s/it]2026-08-31 18:58:38,048 - INFO - Epoch 16 | Train Loss: 0.6550, Train Acc: 0.6292 | Val Loss: 0.6444
2026-08-31 18:58:38,048 - INFO - Val metrics: Acc = 0.6800 | F1 = 0.6701, AUC = 0.7150

 16%|█▌        | 16/100 [02:34<13:26,  9.60s/it]2026-08-31 18:58:47,680 - INFO - Epoch 17 | Train Loss: 0.6427, Train Acc: 0.6577 | Val Loss: 0.6381
2026-08-31 18:58:47,680 - INFO - Val metrics: Acc = 0.6667 | F1 = 0.6546, AUC = 0.7143

 17%|█▋        | 17/100 [02:43<13:17,  9.61s/it]2026-08-31 18:58:57,183 - INFO - Epoch 18 | Train Loss: 0.6326, Train Acc: 0.6477 | Val Loss: 0.6371
2026-08-31 18:58:57,183 - INFO - Val metrics: Acc = 0.6400 | F1 = 0.6399, AUC = 0.7013

 18%|█▊        | 18/100 [02:53<13:05,  9.58s/it]2026-08-31 18:59:06,468 - INFO - Epoch 19 | Train Loss: 0.6313, Train Acc: 0.6544 | Val Loss: 0.6314
2026-08-31 18:59:06,468 - INFO - Val metrics: Acc = 0.6267 | F1 = 0.6267, AUC = 0.6991

 19%|█▉        | 19/100 [03:02<12:48,  9.49s/it]2026-08-31 18:59:15,691 - INFO - Epoch 20 | Train Loss: 0.6458, Train Acc: 0.6426 | Val Loss: 0.6311
2026-08-31 18:59:15,691 - INFO - Val metrics: Acc = 0.6467 | F1 = 0.6440, AUC = 0.7119

 20%|██        | 20/100 [03:11<12:32,  9.41s/it]2026-08-31 18:59:25,008 - INFO - Epoch 21 | Train Loss: 0.6485, Train Acc: 0.6477 | Val Loss: 0.6317
2026-08-31 18:59:25,008 - INFO - Val metrics: Acc = 0.6533 | F1 = 0.6518, AUC = 0.7143

 21%|██        | 21/100 [03:20<12:21,  9.38s/it]2026-08-31 18:59:34,156 - INFO - Epoch 22 | Train Loss: 0.6463, Train Acc: 0.6124 | Val Loss: 0.6387
2026-08-31 18:59:34,156 - INFO - Val metrics: Acc = 0.7067 | F1 = 0.6927, AUC = 0.7216

 22%|██▏       | 22/100 [03:30<12:06,  9.31s/it]2026-08-31 18:59:43,335 - INFO - Epoch 23 | Train Loss: 0.6481, Train Acc: 0.6359 | Val Loss: 0.6406
2026-08-31 18:59:43,335 - INFO - Val metrics: Acc = 0.6267 | F1 = 0.6264, AUC = 0.7118

 23%|██▎       | 23/100 [03:39<11:53,  9.27s/it]2026-08-31 18:59:52,414 - INFO - Epoch 24 | Train Loss: 0.6535, Train Acc: 0.6191 | Val Loss: 0.6353
2026-08-31 18:59:52,414 - INFO - Val metrics: Acc = 0.6400 | F1 = 0.6377, AUC = 0.7117

 24%|██▍       | 24/100 [03:48<11:40,  9.21s/it]2026-08-31 19:00:02,188 - INFO - Epoch 25 | Train Loss: 0.6322, Train Acc: 0.6628 | Val Loss: 0.6218
2026-08-31 19:00:02,188 - INFO - Val metrics: Acc = 0.6400 | F1 = 0.6399, AUC = 0.7208

2026-08-31 19:00:02,188 - INFO - Loss IMPROVEMENT!

 25%|██▌       | 25/100 [03:58<11:43,  9.38s/it]2026-08-31 19:00:11,846 - INFO - Epoch 26 | Train Loss: 0.6325, Train Acc: 0.6544 | Val Loss: 0.6275
2026-08-31 19:00:11,846 - INFO - Val metrics: Acc = 0.6467 | F1 = 0.6459, AUC = 0.7223

 26%|██▌       | 26/100 [04:07<11:40,  9.46s/it]2026-08-31 19:00:21,720 - INFO - Epoch 27 | Train Loss: 0.6268, Train Acc: 0.6426 | Val Loss: 0.6232
2026-08-31 19:00:21,720 - INFO - Val metrics: Acc = 0.6200 | F1 = 0.6192, AUC = 0.7224

 27%|██▋       | 27/100 [04:17<11:39,  9.59s/it]2026-08-31 19:00:31,370 - INFO - Epoch 28 | Train Loss: 0.6404, Train Acc: 0.6393 | Val Loss: 0.6211
2026-08-31 19:00:31,370 - INFO - Val metrics: Acc = 0.6600 | F1 = 0.6599, AUC = 0.7198

2026-08-31 19:00:31,370 - INFO - Loss IMPROVEMENT!

 28%|██▊       | 28/100 [04:27<11:31,  9.61s/it]2026-08-31 19:00:40,422 - INFO - Epoch 29 | Train Loss: 0.6309, Train Acc: 0.6644 | Val Loss: 0.6202
2026-08-31 19:00:40,423 - INFO - Val metrics: Acc = 0.6533 | F1 = 0.6503, AUC = 0.7244

2026-08-31 19:00:40,423 - INFO - Loss IMPROVEMENT!

 29%|██▉       | 29/100 [04:36<11:10,  9.44s/it]2026-08-31 19:00:49,756 - INFO - Epoch 30 | Train Loss: 0.6388, Train Acc: 0.6393 | Val Loss: 0.6271
2026-08-31 19:00:49,756 - INFO - Val metrics: Acc = 0.6333 | F1 = 0.6332, AUC = 0.7167

 30%|███       | 30/100 [04:45<10:58,  9.41s/it]2026-08-31 19:00:59,720 - INFO - Epoch 31 | Train Loss: 0.6245, Train Acc: 0.6779 | Val Loss: 0.6202
2026-08-31 19:00:59,720 - INFO - Val metrics: Acc = 0.6600 | F1 = 0.6600, AUC = 0.7214

 31%|███       | 31/100 [04:55<11:00,  9.57s/it]2026-08-31 19:01:09,691 - INFO - Epoch 32 | Train Loss: 0.6279, Train Acc: 0.6695 | Val Loss: 0.6159
2026-08-31 19:01:09,692 - INFO - Val metrics: Acc = 0.6333 | F1 = 0.6332, AUC = 0.7294

2026-08-31 19:01:09,692 - INFO - Loss IMPROVEMENT!

 32%|███▏      | 32/100 [05:05<10:59,  9.70s/it]2026-08-31 19:01:19,045 - INFO - Epoch 33 | Train Loss: 0.6295, Train Acc: 0.6678 | Val Loss: 0.6163
2026-08-31 19:01:19,045 - INFO - Val metrics: Acc = 0.6200 | F1 = 0.6196, AUC = 0.7320

 33%|███▎      | 33/100 [05:15<10:42,  9.59s/it]2026-08-31 19:01:28,294 - INFO - Epoch 34 | Train Loss: 0.6310, Train Acc: 0.6711 | Val Loss: 0.6282
2026-08-31 19:01:28,294 - INFO - Val metrics: Acc = 0.6133 | F1 = 0.6048, AUC = 0.7361

 34%|███▍      | 34/100 [05:24<10:26,  9.49s/it]2026-08-31 19:01:37,350 - INFO - Epoch 35 | Train Loss: 0.6288, Train Acc: 0.6611 | Val Loss: 0.6500
2026-08-31 19:01:37,350 - INFO - Val metrics: Acc = 0.6067 | F1 = 0.5781, AUC = 0.7272

 35%|███▌      | 35/100 [05:33<10:08,  9.36s/it]2026-08-31 19:01:46,768 - INFO - Epoch 36 | Train Loss: 0.6199, Train Acc: 0.6560 | Val Loss: 0.6070
2026-08-31 19:01:46,768 - INFO - Val metrics: Acc = 0.6467 | F1 = 0.6465, AUC = 0.7454

2026-08-31 19:01:46,768 - INFO - Loss IMPROVEMENT!

 36%|███▌      | 36/100 [05:42<10:00,  9.38s/it]2026-08-31 19:01:55,999 - INFO - Epoch 37 | Train Loss: 0.6182, Train Acc: 0.6795 | Val Loss: 0.6053
2026-08-31 19:01:55,999 - INFO - Val metrics: Acc = 0.6800 | F1 = 0.6798, AUC = 0.7493

2026-08-31 19:01:55,999 - INFO - Loss IMPROVEMENT!

 37%|███▋      | 37/100 [05:51<09:48,  9.33s/it]2026-08-31 19:02:05,293 - INFO - Epoch 38 | Train Loss: 0.6209, Train Acc: 0.6594 | Val Loss: 0.6135
2026-08-31 19:02:05,293 - INFO - Val metrics: Acc = 0.6400 | F1 = 0.6347, AUC = 0.7431

 38%|███▊      | 38/100 [06:01<09:37,  9.32s/it]2026-08-31 19:02:14,507 - INFO - Epoch 39 | Train Loss: 0.6058, Train Acc: 0.6930 | Val Loss: 0.6118
2026-08-31 19:02:14,507 - INFO - Val metrics: Acc = 0.6533 | F1 = 0.6457, AUC = 0.7468

 39%|███▉      | 39/100 [06:10<09:26,  9.29s/it]2026-08-31 19:02:23,700 - INFO - Epoch 40 | Train Loss: 0.5880, Train Acc: 0.6879 | Val Loss: 0.5959
2026-08-31 19:02:23,700 - INFO - Val metrics: Acc = 0.6667 | F1 = 0.6652, AUC = 0.7544

2026-08-31 19:02:23,700 - INFO - Loss IMPROVEMENT!

 40%|████      | 40/100 [06:19<09:15,  9.26s/it]2026-08-31 19:02:32,991 - INFO - Epoch 41 | Train Loss: 0.5987, Train Acc: 0.6695 | Val Loss: 0.6026
2026-08-31 19:02:32,991 - INFO - Val metrics: Acc = 0.6600 | F1 = 0.6545, AUC = 0.7552

 41%|████      | 41/100 [06:28<09:06,  9.27s/it]2026-08-31 19:02:42,062 - INFO - Epoch 42 | Train Loss: 0.6125, Train Acc: 0.6862 | Val Loss: 0.6051
2026-08-31 19:02:42,062 - INFO - Val metrics: Acc = 0.6600 | F1 = 0.6468, AUC = 0.7776

 42%|████▏     | 42/100 [06:38<08:54,  9.21s/it]2026-08-31 19:02:51,416 - INFO - Epoch 43 | Train Loss: 0.5963, Train Acc: 0.7047 | Val Loss: 0.5637
2026-08-31 19:02:51,416 - INFO - Val metrics: Acc = 0.7200 | F1 = 0.7188, AUC = 0.7820

2026-08-31 19:02:51,416 - INFO - Loss IMPROVEMENT!

 43%|████▎     | 43/100 [06:47<08:47,  9.25s/it]2026-08-31 19:03:01,073 - INFO - Epoch 44 | Train Loss: 0.5725, Train Acc: 0.7081 | Val Loss: 0.5723
2026-08-31 19:03:01,073 - INFO - Val metrics: Acc = 0.6867 | F1 = 0.6835, AUC = 0.7847

 44%|████▍     | 44/100 [06:57<08:44,  9.37s/it]2026-08-31 19:03:10,526 - INFO - Epoch 45 | Train Loss: 0.5860, Train Acc: 0.6980 | Val Loss: 0.5879
2026-08-31 19:03:10,526 - INFO - Val metrics: Acc = 0.6600 | F1 = 0.6574, AUC = 0.7710

 45%|████▌     | 45/100 [07:06<08:36,  9.40s/it]2026-08-31 19:03:19,844 - INFO - Epoch 46 | Train Loss: 0.5824, Train Acc: 0.6980 | Val Loss: 0.5490
2026-08-31 19:03:19,844 - INFO - Val metrics: Acc = 0.7533 | F1 = 0.7524, AUC = 0.8096

2026-08-31 19:03:19,844 - INFO - Loss IMPROVEMENT!

 46%|████▌     | 46/100 [07:15<08:26,  9.38s/it]2026-08-31 19:03:29,115 - INFO - Epoch 47 | Train Loss: 0.5851, Train Acc: 0.6997 | Val Loss: 0.5449
2026-08-31 19:03:29,115 - INFO - Val metrics: Acc = 0.7400 | F1 = 0.7400, AUC = 0.8105

2026-08-31 19:03:29,116 - INFO - Loss IMPROVEMENT!

 47%|████▋     | 47/100 [07:25<08:15,  9.34s/it]2026-08-31 19:03:38,357 - INFO - Epoch 48 | Train Loss: 0.5611, Train Acc: 0.7064 | Val Loss: 0.5611
2026-08-31 19:03:38,357 - INFO - Val metrics: Acc = 0.6533 | F1 = 0.6457, AUC = 0.8083

 48%|████▊     | 48/100 [07:34<08:04,  9.31s/it]2026-08-31 19:03:47,573 - INFO - Epoch 49 | Train Loss: 0.5741, Train Acc: 0.6829 | Val Loss: 0.5454
2026-08-31 19:03:47,573 - INFO - Val metrics: Acc = 0.7467 | F1 = 0.7459, AUC = 0.8026

 49%|████▉     | 49/100 [07:43<07:53,  9.28s/it]2026-08-31 19:03:56,931 - INFO - Epoch 50 | Train Loss: 0.5449, Train Acc: 0.7232 | Val Loss: 0.5224
2026-08-31 19:03:56,931 - INFO - Val metrics: Acc = 0.7533 | F1 = 0.7532, AUC = 0.8321

2026-08-31 19:03:56,931 - INFO - Loss IMPROVEMENT!

 50%|█████     | 50/100 [07:52<07:45,  9.31s/it]2026-08-31 19:04:06,208 - INFO - Epoch 51 | Train Loss: 0.5437, Train Acc: 0.7232 | Val Loss: 0.5427
2026-08-31 19:04:06,208 - INFO - Val metrics: Acc = 0.7600 | F1 = 0.7500, AUC = 0.8282

 51%|█████     | 51/100 [08:02<07:35,  9.30s/it]2026-08-31 19:04:15,620 - INFO - Epoch 52 | Train Loss: 0.5520, Train Acc: 0.7064 | Val Loss: 0.5157
2026-08-31 19:04:15,620 - INFO - Val metrics: Acc = 0.7533 | F1 = 0.7532, AUC = 0.8330

2026-08-31 19:04:15,620 - INFO - Loss IMPROVEMENT!

 52%|█████▏    | 52/100 [08:11<07:27,  9.33s/it]2026-08-31 19:04:24,981 - INFO - Epoch 53 | Train Loss: 0.5232, Train Acc: 0.7500 | Val Loss: 0.4628
2026-08-31 19:04:24,982 - INFO - Val metrics: Acc = 0.7867 | F1 = 0.7867, AUC = 0.8658

2026-08-31 19:04:24,982 - INFO - Loss IMPROVEMENT!

 53%|█████▎    | 53/100 [08:20<07:19,  9.34s/it]2026-08-31 19:04:34,298 - INFO - Epoch 54 | Train Loss: 0.5352, Train Acc: 0.7315 | Val Loss: 0.4916
2026-08-31 19:04:34,298 - INFO - Val metrics: Acc = 0.7867 | F1 = 0.7863, AUC = 0.8469

 54%|█████▍    | 54/100 [08:30<07:09,  9.33s/it]2026-08-31 19:04:43,652 - INFO - Epoch 55 | Train Loss: 0.5170, Train Acc: 0.7668 | Val Loss: 0.4663
2026-08-31 19:04:43,652 - INFO - Val metrics: Acc = 0.7867 | F1 = 0.7848, AUC = 0.8702

 55%|█████▌    | 55/100 [08:39<07:00,  9.34s/it]2026-08-31 19:04:52,935 - INFO - Epoch 56 | Train Loss: 0.5648, Train Acc: 0.7148 | Val Loss: 0.5121
2026-08-31 19:04:52,935 - INFO - Val metrics: Acc = 0.7733 | F1 = 0.7732, AUC = 0.8556

 56%|█████▌    | 56/100 [08:48<06:50,  9.32s/it]2026-08-31 19:05:02,403 - INFO - Epoch 57 | Train Loss: 0.5054, Train Acc: 0.7517 | Val Loss: 0.4682
2026-08-31 19:05:02,403 - INFO - Val metrics: Acc = 0.7933 | F1 = 0.7933, AUC = 0.8700

 57%|█████▋    | 57/100 [08:58<06:42,  9.37s/it]2026-08-31 19:05:11,740 - INFO - Epoch 58 | Train Loss: 0.5160, Train Acc: 0.7634 | Val Loss: 0.4543
2026-08-31 19:05:11,741 - INFO - Val metrics: Acc = 0.8200 | F1 = 0.8199, AUC = 0.8734

2026-08-31 19:05:11,741 - INFO - Loss IMPROVEMENT!

 58%|█████▊    | 58/100 [09:07<06:33,  9.36s/it]2026-08-31 19:05:20,993 - INFO - Epoch 59 | Train Loss: 0.5011, Train Acc: 0.7517 | Val Loss: 0.4693
2026-08-31 19:05:20,993 - INFO - Val metrics: Acc = 0.7800 | F1 = 0.7783, AUC = 0.8590

 59%|█████▉    | 59/100 [09:16<06:22,  9.33s/it]2026-08-31 19:06:07,446 - INFO - Epoch 64 | Train Loss: 0.4790, Train Acc: 0.7601 | Val Loss: 0.4438
2026-08-31 19:06:07,446 - INFO - Val metrics: Acc = 0.7667 | F1 = 0.7620, AUC = 0.9022

 64%|██████▍   | 64/100 [10:03<05:35,  9.31s/it]2026-08-31 19:06:16,494 - INFO - Epoch 65 | Train Loss: 0.4299, Train Acc: 0.8020 | Val Loss: 0.3964
2026-08-31 19:06:16,494 - INFO - Val metrics: Acc = 0.8000 | F1 = 0.7987, AUC = 0.9155

 65%|██████▌   | 65/100 [10:12<05:23,  9.23s/it]2026-08-31 19:06:25,824 - INFO - Epoch 66 | Train Loss: 0.4416, Train Acc: 0.7852 | Val Loss: 0.3702
2026-08-31 19:06:25,824 - INFO - Val metrics: Acc = 0.8467 | F1 = 0.8461, AUC = 0.9198

2026-08-31 19:06:25,824 - INFO - Loss IMPROVEMENT!

 66%|██████▌   | 66/100 [10:21<05:14,  9.26s/it]2026-08-31 19:06:36,213 - INFO - Epoch 67 | Train Loss: 0.4654, Train Acc: 0.7634 | Val Loss: 0.3643
2026-08-31 19:06:36,213 - INFO - Val metrics: Acc = 0.8533 | F1 = 0.8531, AUC = 0.9312

2026-08-31 19:06:36,214 - INFO - Loss IMPROVEMENT!

 67%|██████▋   | 67/100 [10:32<05:16,  9.60s/it]2026-08-31 19:06:45,253 - INFO - Epoch 68 | Train Loss: 0.4492, Train Acc: 0.8037 | Val Loss: 0.4157
2026-08-31 19:06:45,253 - INFO - Val metrics: Acc = 0.8000 | F1 = 0.7982, AUC = 0.8976

 68%|██████▊   | 68/100 [10:41<05:01,  9.43s/it]2026-08-31 19:06:54,565 - INFO - Epoch 69 | Train Loss: 0.4635, Train Acc: 0.7668 | Val Loss: 0.4364
2026-08-31 19:06:54,565 - INFO - Val metrics: Acc = 0.7800 | F1 = 0.7756, AUC = 0.9148

 69%|██████▉   | 69/100 [10:50<04:51,  9.40s/it]2026-08-31 19:07:03,986 - INFO - Epoch 70 | Train Loss: 0.4406, Train Acc: 0.7903 | Val Loss: 0.4183
2026-08-31 19:07:03,986 - INFO - Val metrics: Acc = 0.8067 | F1 = 0.8011, AUC = 0.9152

 70%|███████   | 70/100 [10:59<04:42,  9.40s/it]2026-08-31 19:07:13,436 - INFO - Epoch 71 | Train Loss: 0.4376, Train Acc: 0.7919 | Val Loss: 0.4047
2026-08-31 19:07:13,436 - INFO - Val metrics: Acc = 0.8067 | F1 = 0.8042, AUC = 0.9173

 71%|███████   | 71/100 [11:09<04:33,  9.42s/it]2026-08-31 19:07:22,539 - INFO - Epoch 72 | Train Loss: 0.4148, Train Acc: 0.8154 | Val Loss: 0.3787
2026-08-31 19:07:22,539 - INFO - Val metrics: Acc = 0.8267 | F1 = 0.8256, AUC = 0.9358

 72%|███████▏  | 72/100 [11:18<04:21,  9.32s/it]2026-08-31 19:07:31,812 - INFO - Epoch 73 | Train Loss: 0.4627, Train Acc: 0.7852 | Val Loss: 0.3396
2026-08-31 19:07:31,812 - INFO - Val metrics: Acc = 0.8600 | F1 = 0.8600, AUC = 0.9369

2026-08-31 19:07:31,813 - INFO - Loss IMPROVEMENT!

 73%|███████▎  | 73/100 [11:27<04:11,  9.31s/it]2026-08-31 19:07:41,100 - INFO - Epoch 74 | Train Loss: 0.4116, Train Acc: 0.8054 | Val Loss: 0.4151
2026-08-31 19:07:41,100 - INFO - Val metrics: Acc = 0.7933 | F1 = 0.7900, AUC = 0.9193

 74%|███████▍  | 74/100 [11:37<04:01,  9.30s/it]2026-08-31 19:07:50,293 - INFO - Epoch 75 | Train Loss: 0.4384, Train Acc: 0.8205 | Val Loss: 0.3712
2026-08-31 19:07:50,293 - INFO - Val metrics: Acc = 0.8733 | F1 = 0.8733, AUC = 0.9308

 75%|███████▌  | 75/100 [11:46<03:51,  9.27s/it]2026-08-31 19:07:59,785 - INFO - Epoch 76 | Train Loss: 0.4060, Train Acc: 0.8121 | Val Loss: 0.3466
2026-08-31 19:07:59,785 - INFO - Val metrics: Acc = 0.8467 | F1 = 0.8463, AUC = 0.9275

 76%|███████▌  | 76/100 [11:55<03:44,  9.34s/it]2026-08-31 19:08:09,030 - INFO - Epoch 77 | Train Loss: 0.4493, Train Acc: 0.7836 | Val Loss: 0.3313
2026-08-31 19:08:09,031 - INFO - Val metrics: Acc = 0.8667 | F1 = 0.8666, AUC = 0.9431

2026-08-31 19:08:09,031 - INFO - Loss IMPROVEMENT!

 77%|███████▋  | 77/100 [12:05<03:34,  9.31s/it]2026-08-31 19:08:18,225 - INFO - Epoch 78 | Train Loss: 0.4189, Train Acc: 0.8138 | Val Loss: 0.3336
2026-08-31 19:08:18,225 - INFO - Val metrics: Acc = 0.9000 | F1 = 0.9000, AUC = 0.9419

 78%|███████▊  | 78/100 [12:14<03:24,  9.27s/it]2026-08-31 19:08:27,630 - INFO - Epoch 79 | Train Loss: 0.3893, Train Acc: 0.8104 | Val Loss: 0.3217
2026-08-31 19:08:27,630 - INFO - Val metrics: Acc = 0.8600 | F1 = 0.8592, AUC = 0.9582

2026-08-31 19:08:27,630 - INFO - Loss IMPROVEMENT!

 79%|███████▉  | 79/100 [12:23<03:15,  9.32s/it]2026-08-31 19:08:37,282 - INFO - Epoch 80 | Train Loss: 0.4427, Train Acc: 0.7919 | Val Loss: 0.3493
2026-08-31 19:08:37,282 - INFO - Val metrics: Acc = 0.8933 | F1 = 0.8933, AUC = 0.9531

 80%|████████  | 80/100 [12:33<03:08,  9.41s/it]2026-08-31 19:08:46,646 - INFO - Epoch 81 | Train Loss: 0.3781, Train Acc: 0.8540 | Val Loss: 0.2975
2026-08-31 19:08:46,646 - INFO - Val metrics: Acc = 0.9000 | F1 = 0.9000, AUC = 0.9445

2026-08-31 19:08:46,646 - INFO - Loss IMPROVEMENT!

 81%|████████  | 81/100 [12:42<02:58,  9.40s/it]2026-08-31 19:08:55,779 - INFO - Epoch 82 | Train Loss: 0.4123, Train Acc: 0.8070 | Val Loss: 0.2847
2026-08-31 19:08:55,779 - INFO - Val metrics: Acc = 0.9000 | F1 = 0.9000, AUC = 0.9516

2026-08-31 19:08:55,779 - INFO - Loss IMPROVEMENT!

 82%|████████▏ | 82/100 [12:51<02:47,  9.32s/it]2026-08-31 19:09:05,007 - INFO - Epoch 83 | Train Loss: 0.4162, Train Acc: 0.8188 | Val Loss: 0.3068
2026-08-31 19:09:05,008 - INFO - Val metrics: Acc = 0.8733 | F1 = 0.8729, AUC = 0.9500

 83%|████████▎ | 83/100 [13:00<02:37,  9.29s/it]2026-08-31 19:09:14,300 - INFO - Epoch 84 | Train Loss: 0.3970, Train Acc: 0.8255 | Val Loss: 0.2989
2026-08-31 19:09:14,300 - INFO - Val metrics: Acc = 0.9200 | F1 = 0.9199, AUC = 0.9587

 84%|████████▍ | 84/100 [13:10<02:28,  9.29s/it]^C
(BioEnv) [cbelluti@ailb-login-02 5_training]$ tail -f script.err

2026-08-31 19:10:46,220 - INFO - Loss IMPROVEMENT!

 94%|█████████▍| 94/100 [14:42<00:55,  9.18s/it]2026-08-31 19:10:55,375 - INFO - Epoch 95 | Train Loss: 0.3531, Train Acc: 0.8356 | Val Loss: 0.2784
2026-08-31 19:10:55,376 - INFO - Val metrics: Acc = 0.8533 | F1 = 0.8531, AUC = 0.9582

 95%|█████████▌| 95/100 [14:51<00:45,  9.17s/it]2026-08-31 19:11:04,521 - INFO - Epoch 96 | Train Loss: 0.3359, Train Acc: 0.8507 | Val Loss: 0.2915
2026-08-31 19:11:04,521 - INFO - Val metrics: Acc = 0.9000 | F1 = 0.8999, AUC = 0.9518

 96%|█████████▌| 96/100 [15:00<00:36,  9.16s/it]^C
(BioEnv) [cbelluti@ailb-login-02 5_training]$ squeue -u cbelluti,
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             95610 all_usr_p training cbelluti  R      16:13      1 germano
(BioEnv) [cbelluti@ailb-login-02 5_training]$ tail -f script.err

2026-08-31 19:10:46,220 - INFO - Loss IMPROVEMENT!

 94%|█████████▍| 94/100 [14:42<00:55,  9.18s/it]2026-08-31 19:10:55,375 - INFO - Epoch 95 | Train Loss: 0.3531, Train Acc: 0.8356 | Val Loss: 0.2784
2026-08-31 19:10:55,376 - INFO - Val metrics: Acc = 0.8533 | F1 = 0.8531, AUC = 0.9582

 95%|█████████▌| 95/100 [14:51<00:45,  9.17s/it]2026-08-31 19:11:04,521 - INFO - Epoch 96 | Train Loss: 0.3359, Train Acc: 0.8507 | Val Loss: 0.2915
2026-08-31 19:11:04,521 - INFO - Val metrics: Acc = 0.9000 | F1 = 0.8999, AUC = 0.9518

 96%|█████████▌| 96/100 [15:00<00:36,  9.16s/it]2026-08-31 19:11:48,429 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 64, heads=4)
  (residual_proj1): Linear(in_features=10, out_features=256, bias=True)
  (residual_proj2): Linear(in_features=256, out_features=64, bias=True)
  (gat2): GATv2Conv(256, 64, heads=1)
  (bn1): GraphNorm(256)
  (bn2): GraphNorm(64)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=2, bias=True)
  )
  (dropout): Dropout(p=0.3, inplace=False)
)

  0%|          | 0/100 [00:00<?, ?it/s]2026-08-31 19:11:58,258 - INFO - Epoch 01 | Train Loss: 0.7548, Train Acc: 0.5042 | Val Loss: 0.6845
2026-08-31 19:11:58,258 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.6047, AUC = 0.6544

2026-08-31 19:11:58,259 - INFO - Loss IMPROVEMENT!

  1%|          | 1/100 [00:09<16:13,  9.84s/it]2026-08-31 19:12:08,469 - INFO - Epoch 02 | Train Loss: 0.7114, Train Acc: 0.5159 | Val Loss: 0.6883
2026-08-31 19:12:08,469 - INFO - Val metrics: Acc = 0.5101 | F1 = 0.3497, AUC = 0.6670

  2%|▏         | 2/100 [00:20<16:25, 10.05s/it]2026-08-31 19:12:18,995 - INFO - Epoch 03 | Train Loss: 0.7056, Train Acc: 0.5075 | Val Loss: 0.6733
2026-08-31 19:12:18,995 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.6099, AUC = 0.6795

2026-08-31 19:12:18,995 - INFO - Loss IMPROVEMENT!

  3%|▎         | 3/100 [00:30<16:36, 10.27s/it]2026-08-31 19:12:28,511 - INFO - Epoch 04 | Train Loss: 0.6823, Train Acc: 0.5846 | Val Loss: 0.6812
2026-08-31 19:12:28,511 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6531, AUC = 0.6841

  4%|▍         | 4/100 [00:40<15:57,  9.97s/it]2026-08-31 19:12:37,671 - INFO - Epoch 05 | Train Loss: 0.6668, Train Acc: 0.6248 | Val Loss: 0.6670
2026-08-31 19:12:37,671 - INFO - Val metrics: Acc = 0.5772 | F1 = 0.5659, AUC = 0.6890

2026-08-31 19:12:37,671 - INFO - Loss IMPROVEMENT!

  5%|▌         | 5/100 [00:49<15:19,  9.68s/it]2026-08-31 19:12:46,811 - INFO - Epoch 06 | Train Loss: 0.6824, Train Acc: 0.5712 | Val Loss: 0.6723
2026-08-31 19:12:46,812 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6713, AUC = 0.6908

  6%|▌         | 6/100 [00:58<14:52,  9.49s/it]2026-08-31 19:12:56,241 - INFO - Epoch 07 | Train Loss: 0.6706, Train Acc: 0.5712 | Val Loss: 0.6646
2026-08-31 19:12:56,242 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6365, AUC = 0.6901

2026-08-31 19:12:56,242 - INFO - Loss IMPROVEMENT!

  7%|▋         | 7/100 [01:07<14:41,  9.48s/it]2026-08-31 19:13:05,694 - INFO - Epoch 08 | Train Loss: 0.6603, Train Acc: 0.6248 | Val Loss: 0.6609
2026-08-31 19:13:05,694 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6052, AUC = 0.6946

2026-08-31 19:13:05,694 - INFO - Loss IMPROVEMENT!

  8%|▊         | 8/100 [01:17<14:31,  9.47s/it]2026-08-31 19:13:15,082 - INFO - Epoch 09 | Train Loss: 0.6595, Train Acc: 0.6600 | Val Loss: 0.6466
2026-08-31 19:13:15,082 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6767, AUC = 0.6962

2026-08-31 19:13:15,082 - INFO - Loss IMPROVEMENT!

  9%|▉         | 9/100 [01:26<14:19,  9.44s/it]2026-08-31 19:13:24,547 - INFO - Epoch 10 | Train Loss: 0.6556, Train Acc: 0.6365 | Val Loss: 0.6517
2026-08-31 19:13:24,547 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6624, AUC = 0.6890

 10%|█         | 10/100 [01:36<14:10,  9.45s/it]2026-08-31 19:13:33,956 - INFO - Epoch 11 | Train Loss: 0.6480, Train Acc: 0.6365 | Val Loss: 0.6419
2026-08-31 19:13:33,957 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6619, AUC = 0.6932

2026-08-31 19:13:33,957 - INFO - Loss IMPROVEMENT!

 11%|█         | 11/100 [01:45<13:59,  9.44s/it]2026-08-31 19:13:43,212 - INFO - Epoch 12 | Train Loss: 0.6583, Train Acc: 0.6231 | Val Loss: 0.6317
2026-08-31 19:13:43,212 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6577, AUC = 0.7004

2026-08-31 19:13:43,213 - INFO - Loss IMPROVEMENT!

 12%|█▏        | 12/100 [01:54<13:45,  9.38s/it]2026-08-31 19:13:52,692 - INFO - Epoch 13 | Train Loss: 0.6320, Train Acc: 0.6734 | Val Loss: 0.6446
2026-08-31 19:13:52,692 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6054, AUC = 0.7016

 13%|█▎        | 13/100 [02:04<13:38,  9.41s/it]2026-08-31 19:14:02,267 - INFO - Epoch 14 | Train Loss: 0.6232, Train Acc: 0.6650 | Val Loss: 0.6668
2026-08-31 19:14:02,267 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6068, AUC = 0.6977

 14%|█▍        | 14/100 [02:13<13:33,  9.46s/it]2026-08-31 19:14:11,861 - INFO - Epoch 15 | Train Loss: 0.6514, Train Acc: 0.6298 | Val Loss: 0.6552
2026-08-31 19:14:11,861 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6068, AUC = 0.7083

 15%|█▌        | 15/100 [02:23<13:27,  9.50s/it]2026-08-31 19:14:21,248 - INFO - Epoch 16 | Train Loss: 0.6492, Train Acc: 0.6332 | Val Loss: 0.6331
2026-08-31 19:14:21,248 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6777, AUC = 0.7121

 16%|█▌        | 16/100 [02:32<13:15,  9.47s/it]2026-08-31 19:14:30,553 - INFO - Epoch 17 | Train Loss: 0.6221, Train Acc: 0.6432 | Val Loss: 0.6554
2026-08-31 19:14:30,553 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6652, AUC = 0.7022

 17%|█▋        | 17/100 [02:42<13:01,  9.42s/it]2026-08-31 19:14:39,985 - INFO - Epoch 18 | Train Loss: 0.6310, Train Acc: 0.6482 | Val Loss: 0.6331
2026-08-31 19:14:39,985 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6778, AUC = 0.7059

 18%|█▊        | 18/100 [02:51<12:52,  9.42s/it]2026-08-31 19:14:49,356 - INFO - Epoch 19 | Train Loss: 0.6228, Train Acc: 0.6683 | Val Loss: 0.6386
2026-08-31 19:14:49,356 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6470, AUC = 0.7067

 19%|█▉        | 19/100 [03:00<12:41,  9.41s/it]2026-08-31 19:14:58,836 - INFO - Epoch 20 | Train Loss: 0.6304, Train Acc: 0.6834 | Val Loss: 0.6342
2026-08-31 19:14:58,836 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6713, AUC = 0.7160

 20%|██        | 20/100 [03:10<12:34,  9.43s/it]2026-08-31 19:15:08,065 - INFO - Epoch 21 | Train Loss: 0.6293, Train Acc: 0.6533 | Val Loss: 0.6302
2026-08-31 19:15:08,065 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6637, AUC = 0.7150

2026-08-31 19:15:08,066 - INFO - Loss IMPROVEMENT!

 21%|██        | 21/100 [03:19<12:20,  9.37s/it]2026-08-31 19:15:17,340 - INFO - Epoch 22 | Train Loss: 0.6294, Train Acc: 0.6583 | Val Loss: 0.6412
2026-08-31 19:15:17,340 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6470, AUC = 0.7067

 22%|██▏       | 22/100 [03:28<12:08,  9.34s/it]2026-08-31 19:15:26,782 - INFO - Epoch 23 | Train Loss: 0.6097, Train Acc: 0.6717 | Val Loss: 0.6440
2026-08-31 19:15:26,782 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6619, AUC = 0.7121

 23%|██▎       | 23/100 [03:38<12:01,  9.37s/it]2026-08-31 19:15:36,380 - INFO - Epoch 24 | Train Loss: 0.6320, Train Acc: 0.6616 | Val Loss: 0.6393
2026-08-31 19:15:36,380 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6352, AUC = 0.7209

 24%|██▍       | 24/100 [03:47<11:57,  9.44s/it]2026-08-31 19:15:45,814 - INFO - Epoch 25 | Train Loss: 0.6078, Train Acc: 0.6784 | Val Loss: 0.6213
2026-08-31 19:15:45,814 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6979, AUC = 0.7285

2026-08-31 19:15:45,814 - INFO - Loss IMPROVEMENT!

 25%|██▌       | 25/100 [03:57<11:47,  9.44s/it]2026-08-31 19:15:55,304 - INFO - Epoch 26 | Train Loss: 0.6094, Train Acc: 0.6734 | Val Loss: 0.6597
2026-08-31 19:15:55,304 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6022, AUC = 0.7191

 26%|██▌       | 26/100 [04:06<11:39,  9.45s/it]2026-08-31 19:16:04,800 - INFO - Epoch 27 | Train Loss: 0.6004, Train Acc: 0.6851 | Val Loss: 0.6027
2026-08-31 19:16:04,800 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7309, AUC = 0.7544

2026-08-31 19:16:04,800 - INFO - Loss IMPROVEMENT!

 27%|██▋       | 27/100 [04:16<11:31,  9.47s/it]2026-08-31 19:16:14,312 - INFO - Epoch 28 | Train Loss: 0.5917, Train Acc: 0.6834 | Val Loss: 0.6232
2026-08-31 19:16:14,312 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6624, AUC = 0.7497

 28%|██▊       | 28/100 [04:25<11:22,  9.48s/it]2026-08-31 19:16:23,842 - INFO - Epoch 29 | Train Loss: 0.6003, Train Acc: 0.6767 | Val Loss: 0.5991
2026-08-31 19:16:23,842 - INFO - Val metrics: Acc = 0.7047 | F1 = 0.7044, AUC = 0.7555

2026-08-31 19:16:23,842 - INFO - Loss IMPROVEMENT!

 29%|██▉       | 29/100 [04:35<11:14,  9.50s/it]2026-08-31 19:16:33,321 - INFO - Epoch 30 | Train Loss: 0.6154, Train Acc: 0.6951 | Val Loss: 0.5903
2026-08-31 19:16:33,321 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7166, AUC = 0.7715

2026-08-31 19:16:33,321 - INFO - Loss IMPROVEMENT!

 30%|███       | 30/100 [04:44<11:04,  9.49s/it]2026-08-31 19:16:42,748 - INFO - Epoch 31 | Train Loss: 0.5965, Train Acc: 0.7018 | Val Loss: 0.5887
2026-08-31 19:16:42,748 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7180, AUC = 0.7800

2026-08-31 19:16:42,748 - INFO - Loss IMPROVEMENT!

 31%|███       | 31/100 [04:54<10:53,  9.47s/it]2026-08-31 19:16:51,959 - INFO - Epoch 32 | Train Loss: 0.5692, Train Acc: 0.7052 | Val Loss: 0.5690
2026-08-31 19:16:51,960 - INFO - Val metrics: Acc = 0.7383 | F1 = 0.7381, AUC = 0.7832

2026-08-31 19:16:51,960 - INFO - Loss IMPROVEMENT!

 32%|███▏      | 32/100 [05:03<10:38,  9.39s/it]2026-08-31 19:17:01,443 - INFO - Epoch 33 | Train Loss: 0.5715, Train Acc: 0.6985 | Val Loss: 0.6132
2026-08-31 19:17:01,443 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6471, AUC = 0.7816

 33%|███▎      | 33/100 [05:13<10:31,  9.42s/it]2026-08-31 19:17:10,673 - INFO - Epoch 34 | Train Loss: 0.5625, Train Acc: 0.7152 | Val Loss: 0.5862
2026-08-31 19:17:10,673 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.7166, AUC = 0.7724

 34%|███▍      | 34/100 [05:22<10:17,  9.36s/it]2026-08-31 19:17:20,242 - INFO - Epoch 35 | Train Loss: 0.5723, Train Acc: 0.7219 | Val Loss: 0.5817
2026-08-31 19:17:20,242 - INFO - Val metrics: Acc = 0.7248 | F1 = 0.7175, AUC = 0.7998

 35%|███▌      | 35/100 [05:31<10:12,  9.42s/it]2026-08-31 19:17:29,825 - INFO - Epoch 36 | Train Loss: 0.5655, Train Acc: 0.6951 | Val Loss: 0.5652
2026-08-31 19:17:29,825 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7649, AUC = 0.8043

2026-08-31 19:17:29,825 - INFO - Loss IMPROVEMENT!

 36%|███▌      | 36/100 [05:41<10:06,  9.47s/it]2026-08-31 19:17:39,303 - INFO - Epoch 37 | Train Loss: 0.5840, Train Acc: 0.6968 | Val Loss: 0.5876
2026-08-31 19:17:39,303 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7288, AUC = 0.7932

 37%|███▋      | 37/100 [05:50<09:56,  9.47s/it]2026-08-31 19:17:49,971 - INFO - Epoch 38 | Train Loss: 0.5615, Train Acc: 0.7169 | Val Loss: 0.5574
2026-08-31 19:17:49,972 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7424, AUC = 0.8074

2026-08-31 19:17:49,972 - INFO - Loss IMPROVEMENT!

 38%|███▊      | 38/100 [06:01<10:09,  9.83s/it]2026-08-31 19:17:59,477 - INFO - Epoch 39 | Train Loss: 0.5451, Train Acc: 0.7102 | Val Loss: 0.5550
2026-08-31 19:17:59,477 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7581, AUC = 0.8063

2026-08-31 19:17:59,477 - INFO - Loss IMPROVEMENT!

 39%|███▉      | 39/100 [06:11<09:53,  9.74s/it]2026-08-31 19:18:08,859 - INFO - Epoch 40 | Train Loss: 0.5523, Train Acc: 0.7219 | Val Loss: 0.5480
2026-08-31 19:18:08,859 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7581, AUC = 0.8144

2026-08-31 19:18:08,859 - INFO - Loss IMPROVEMENT!

 40%|████      | 40/100 [06:20<09:37,  9.63s/it]2026-08-31 19:18:18,157 - INFO - Epoch 41 | Train Loss: 0.5487, Train Acc: 0.7270 | Val Loss: 0.5275
2026-08-31 19:18:18,158 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7984, AUC = 0.8339

2026-08-31 19:18:18,158 - INFO - Loss IMPROVEMENT!

 41%|████      | 41/100 [06:29<09:22,  9.53s/it]2026-08-31 19:18:27,381 - INFO - Epoch 42 | Train Loss: 0.5303, Train Acc: 0.7504 | Val Loss: 0.5336
2026-08-31 19:18:27,382 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7608, AUC = 0.8359

 42%|████▏     | 42/100 [06:38<09:07,  9.44s/it]2026-08-31 19:18:37,067 - INFO - Epoch 43 | Train Loss: 0.5216, Train Acc: 0.7487 | Val Loss: 0.5288
2026-08-31 19:18:37,067 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7640, AUC = 0.8305

 43%|████▎     | 43/100 [06:48<09:02,  9.51s/it]2026-08-31 19:18:47,639 - INFO - Epoch 44 | Train Loss: 0.5290, Train Acc: 0.7538 | Val Loss: 0.5252
2026-08-31 19:18:47,639 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7910, AUC = 0.8332

2026-08-31 19:18:47,639 - INFO - Loss IMPROVEMENT!

 44%|████▍     | 44/100 [06:59<09:10,  9.83s/it]2026-08-31 19:18:57,576 - INFO - Epoch 45 | Train Loss: 0.5238, Train Acc: 0.7420 | Val Loss: 0.4903
2026-08-31 19:18:57,577 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7785, AUC = 0.8524

2026-08-31 19:18:57,577 - INFO - Loss IMPROVEMENT!

 45%|████▌     | 45/100 [07:09<09:02,  9.86s/it]2026-08-31 19:19:07,239 - INFO - Epoch 46 | Train Loss: 0.4914, Train Acc: 0.7873 | Val Loss: 0.5024
2026-08-31 19:19:07,239 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7640, AUC = 0.8541

 46%|████▌     | 46/100 [07:18<08:49,  9.80s/it]2026-08-31 19:19:16,618 - INFO - Epoch 47 | Train Loss: 0.5248, Train Acc: 0.7554 | Val Loss: 0.5378
2026-08-31 19:19:16,618 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7349, AUC = 0.8668

 47%|████▋     | 47/100 [07:28<08:32,  9.67s/it]2026-08-31 19:19:26,065 - INFO - Epoch 48 | Train Loss: 0.5076, Train Acc: 0.7705 | Val Loss: 0.5268
2026-08-31 19:19:26,065 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7848, AUC = 0.8342

 48%|████▊     | 48/100 [07:37<08:19,  9.61s/it]2026-08-31 19:19:35,446 - INFO - Epoch 49 | Train Loss: 0.4760, Train Acc: 0.7873 | Val Loss: 0.5249
2026-08-31 19:19:35,446 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7716, AUC = 0.8688

 49%|████▉     | 49/100 [07:47<08:06,  9.54s/it]2026-08-31 19:19:44,926 - INFO - Epoch 50 | Train Loss: 0.4873, Train Acc: 0.7454 | Val Loss: 0.5043
2026-08-31 19:19:44,926 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7910, AUC = 0.8523

 50%|█████     | 50/100 [07:56<07:56,  9.52s/it]2026-08-31 19:19:54,336 - INFO - Epoch 51 | Train Loss: 0.5025, Train Acc: 0.7454 | Val Loss: 0.4938
2026-08-31 19:19:54,336 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8096, AUC = 0.8667

 51%|█████     | 51/100 [08:05<07:44,  9.49s/it]2026-08-31 19:20:03,826 - INFO - Epoch 52 | Train Loss: 0.4786, Train Acc: 0.7856 | Val Loss: 0.4781
2026-08-31 19:20:03,826 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7850, AUC = 0.8712

2026-08-31 19:20:03,826 - INFO - Loss IMPROVEMENT!

 52%|█████▏    | 52/100 [08:15<07:35,  9.49s/it]2026-08-31 19:20:13,156 - INFO - Epoch 53 | Train Loss: 0.4595, Train Acc: 0.7956 | Val Loss: 0.4733
2026-08-31 19:20:13,156 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7918, AUC = 0.8683

2026-08-31 19:20:13,156 - INFO - Loss IMPROVEMENT!

 53%|█████▎    | 53/100 [08:24<07:23,  9.44s/it]2026-08-31 19:20:22,466 - INFO - Epoch 54 | Train Loss: 0.4581, Train Acc: 0.7873 | Val Loss: 0.4573
2026-08-31 19:20:22,467 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7986, AUC = 0.8668

2026-08-31 19:20:22,467 - INFO - Loss IMPROVEMENT!

 54%|█████▍    | 54/100 [08:34<07:12,  9.40s/it]2026-08-31 19:20:31,687 - INFO - Epoch 55 | Train Loss: 0.4650, Train Acc: 0.7906 | Val Loss: 0.4635
2026-08-31 19:20:31,687 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7848, AUC = 0.8685

 55%|█████▌    | 55/100 [08:43<07:00,  9.35s/it]2026-08-31 19:20:40,777 - INFO - Epoch 56 | Train Loss: 0.4408, Train Acc: 0.7940 | Val Loss: 0.5008
2026-08-31 19:20:40,777 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7953, AUC = 0.8634

 56%|█████▌    | 56/100 [08:52<06:47,  9.27s/it]2026-08-31 19:20:50,123 - INFO - Epoch 57 | Train Loss: 0.4565, Train Acc: 0.7856 | Val Loss: 0.5342
2026-08-31 19:20:50,123 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7461, AUC = 0.8559

 57%|█████▋    | 57/100 [09:01<06:39,  9.29s/it]2026-08-31 19:20:59,480 - INFO - Epoch 58 | Train Loss: 0.4420, Train Acc: 0.7856 | Val Loss: 0.5072
2026-08-31 19:20:59,480 - INFO - Val metrics: Acc = 0.7651 | F1 = 0.7588, AUC = 0.8777

 58%|█████▊    | 58/100 [09:11<06:31,  9.31s/it]2026-08-31 19:21:08,856 - INFO - Epoch 59 | Train Loss: 0.4615, Train Acc: 0.7755 | Val Loss: 0.4938
2026-08-31 19:21:08,856 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7895, AUC = 0.8672

 59%|█████▉    | 59/100 [09:20<06:22,  9.33s/it]2026-08-31 19:21:18,117 - INFO - Epoch 60 | Train Loss: 0.4367, Train Acc: 0.8090 | Val Loss: 0.4886
2026-08-31 19:21:18,117 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7771, AUC = 0.8647

 60%|██████    | 60/100 [09:29<06:12,  9.31s/it]2026-08-31 19:21:27,499 - INFO - Epoch 61 | Train Loss: 0.4134, Train Acc: 0.8057 | Val Loss: 0.4366
2026-08-31 19:21:27,500 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8176, AUC = 0.8836

2026-08-31 19:21:27,500 - INFO - Loss IMPROVEMENT!

 61%|██████    | 61/100 [09:39<06:04,  9.33s/it]2026-08-31 19:21:36,907 - INFO - Epoch 62 | Train Loss: 0.4168, Train Acc: 0.8124 | Val Loss: 0.4542
2026-08-31 19:21:36,908 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8096, AUC = 0.8825

 62%|██████▏   | 62/100 [09:48<05:55,  9.35s/it]2026-08-31 19:21:46,116 - INFO - Epoch 63 | Train Loss: 0.4540, Train Acc: 0.7822 | Val Loss: 0.4749
2026-08-31 19:21:46,116 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7800, AUC = 0.8978

 63%|██████▎   | 63/100 [09:57<05:44,  9.31s/it]2026-08-31 19:21:55,304 - INFO - Epoch 64 | Train Loss: 0.3980, Train Acc: 0.8124 | Val Loss: 0.4309
2026-08-31 19:21:55,305 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8036, AUC = 0.8928

2026-08-31 19:21:55,305 - INFO - Loss IMPROVEMENT!

 64%|██████▍   | 64/100 [10:06<05:33,  9.28s/it]2026-08-31 19:22:04,620 - INFO - Epoch 65 | Train Loss: 0.3976, Train Acc: 0.8291 | Val Loss: 0.4416
2026-08-31 19:22:04,620 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7817, AUC = 0.8946

 65%|██████▌   | 65/100 [10:16<05:25,  9.29s/it]2026-08-31 19:22:13,912 - INFO - Epoch 66 | Train Loss: 0.3850, Train Acc: 0.8342 | Val Loss: 0.4692
2026-08-31 19:22:13,913 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7809, AUC = 0.8888

 66%|██████▌   | 66/100 [10:25<05:15,  9.29s/it]2026-08-31 19:22:23,119 - INFO - Epoch 67 | Train Loss: 0.4072, Train Acc: 0.8074 | Val Loss: 0.5639
2026-08-31 19:22:23,119 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7349, AUC = 0.8807

 67%|██████▋   | 67/100 [10:34<05:05,  9.26s/it]2026-08-31 19:22:32,378 - INFO - Epoch 68 | Train Loss: 0.3901, Train Acc: 0.8358 | Val Loss: 0.4462
2026-08-31 19:22:32,378 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7913, AUC = 0.8847

 68%|██████▊   | 68/100 [10:43<04:56,  9.26s/it]2026-08-31 19:22:41,691 - INFO - Epoch 69 | Train Loss: 0.4208, Train Acc: 0.8090 | Val Loss: 0.5685
2026-08-31 19:22:41,691 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7318, AUC = 0.8735

 69%|██████▉   | 69/100 [10:53<04:47,  9.28s/it]2026-08-31 19:22:50,881 - INFO - Epoch 70 | Train Loss: 0.3972, Train Acc: 0.8174 | Val Loss: 0.4816
2026-08-31 19:22:50,881 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7824, AUC = 0.8695

 70%|███████   | 70/100 [11:02<04:37,  9.25s/it]2026-08-31 19:23:00,031 - INFO - Epoch 71 | Train Loss: 0.3945, Train Acc: 0.8291 | Val Loss: 0.4560
2026-08-31 19:23:00,032 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7953, AUC = 0.8793

 71%|███████   | 71/100 [11:11<04:27,  9.22s/it]2026-08-31 19:23:09,276 - INFO - Epoch 72 | Train Loss: 0.3885, Train Acc: 0.8342 | Val Loss: 0.5244
2026-08-31 19:23:09,276 - INFO - Val metrics: Acc = 0.7315 | F1 = 0.7194, AUC = 0.8924

 72%|███████▏  | 72/100 [11:20<04:18,  9.23s/it]2026-08-31 19:23:19,723 - INFO - Epoch 73 | Train Loss: 0.4414, Train Acc: 0.7973 | Val Loss: 0.5182
2026-08-31 19:23:19,723 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7502, AUC = 0.8901

 73%|███████▎  | 73/100 [11:31<04:19,  9.59s/it]2026-08-31 19:23:28,818 - INFO - Epoch 74 | Train Loss: 0.4058, Train Acc: 0.8141 | Val Loss: 0.5102
2026-08-31 19:23:28,818 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7412, AUC = 0.9022

 74%|███████▍  | 74/100 [11:40<04:05,  9.44s/it]2026-08-31 19:23:37,828 - INFO - Epoch 75 | Train Loss: 0.4022, Train Acc: 0.8141 | Val Loss: 0.5113
2026-08-31 19:23:37,828 - INFO - Val metrics: Acc = 0.7919 | F1 = 0.7864, AUC = 0.8953

 75%|███████▌  | 75/100 [11:49<03:52,  9.31s/it]2026-08-31 19:23:46,979 - INFO - Epoch 76 | Train Loss: 0.3450, Train Acc: 0.8509 | Val Loss: 0.4110
2026-08-31 19:23:46,979 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8106, AUC = 0.8993

2026-08-31 19:23:46,979 - INFO - Loss IMPROVEMENT!

 76%|███████▌  | 76/100 [11:58<03:42,  9.27s/it]2026-08-31 19:23:55,839 - INFO - Epoch 77 | Train Loss: 0.3340, Train Acc: 0.8576 | Val Loss: 0.3823
2026-08-31 19:23:55,840 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8386, AUC = 0.9076

2026-08-31 19:23:55,840 - INFO - Loss IMPROVEMENT!

 77%|███████▋  | 77/100 [12:07<03:30,  9.15s/it]2026-08-31 19:24:05,770 - INFO - Epoch 78 | Train Loss: 0.3374, Train Acc: 0.8693 | Val Loss: 0.3758
2026-08-31 19:24:05,770 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8249, AUC = 0.9132

2026-08-31 19:24:05,770 - INFO - Loss IMPROVEMENT!

 78%|███████▊  | 78/100 [12:17<03:26,  9.38s/it]2026-08-31 19:24:15,648 - INFO - Epoch 79 | Train Loss: 0.3446, Train Acc: 0.8693 | Val Loss: 0.3878
2026-08-31 19:24:15,648 - INFO - Val metrics: Acc = 0.8456 | F1 = 0.8456, AUC = 0.9049

 79%|███████▉  | 79/100 [12:27<03:20,  9.53s/it]2026-08-31 19:24:25,574 - INFO - Epoch 80 | Train Loss: 0.3444, Train Acc: 0.8409 | Val Loss: 0.4212
2026-08-31 19:24:25,574 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8090, AUC = 0.9097

 80%|████████  | 80/100 [12:37<03:12,  9.65s/it]2026-08-31 19:24:35,511 - INFO - Epoch 81 | Train Loss: 0.3193, Train Acc: 0.8492 | Val Loss: 0.4721
2026-08-31 19:24:35,511 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8010, AUC = 0.9032

 81%|████████  | 81/100 [12:47<03:04,  9.73s/it]2026-08-31 19:24:44,751 - INFO - Epoch 82 | Train Loss: 0.3268, Train Acc: 0.8643 | Val Loss: 0.4381
2026-08-31 19:24:44,752 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8102, AUC = 0.8930

 82%|████████▏ | 82/100 [12:56<02:52,  9.59s/it]2026-08-31 19:24:54,070 - INFO - Epoch 83 | Train Loss: 0.3320, Train Acc: 0.8677 | Val Loss: 0.4118
2026-08-31 19:24:54,070 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8242, AUC = 0.8969

 83%|████████▎ | 83/100 [13:05<02:41,  9.51s/it]2026-08-31 19:25:03,185 - INFO - Epoch 84 | Train Loss: 0.3498, Train Acc: 0.8425 | Val Loss: 0.4464
2026-08-31 19:25:03,185 - INFO - Val metrics: Acc = 0.8054 | F1 = 0.8025, AUC = 0.9106

 84%|████████▍ | 84/100 [13:14<02:30,  9.39s/it]2026-08-31 19:25:12,390 - INFO - Epoch 85 | Train Loss: 0.3763, Train Acc: 0.8526 | Val Loss: 0.3944
2026-08-31 19:25:12,390 - INFO - Val metrics: Acc = 0.8322 | F1 = 0.8315, AUC = 0.9065

 85%|████████▌ | 85/100 [13:23<02:20,  9.33s/it]2026-08-31 19:25:21,583 - INFO - Epoch 86 | Train Loss: 0.3230, Train Acc: 0.8543 | Val Loss: 0.4106
2026-08-31 19:25:21,583 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8096, AUC = 0.9223

 86%|████████▌ | 86/100 [13:33<02:10,  9.29s/it]2026-08-31 19:25:31,032 - INFO - Epoch 87 | Train Loss: 0.3238, Train Acc: 0.8509 | Val Loss: 0.4005
2026-08-31 19:25:31,032 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8176, AUC = 0.9115

 87%|████████▋ | 87/100 [13:42<02:01,  9.34s/it]2026-08-31 19:25:40,406 - INFO - Epoch 88 | Train Loss: 0.3514, Train Acc: 0.8392 | Val Loss: 0.3987
2026-08-31 19:25:40,406 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8106, AUC = 0.9119

 88%|████████▊ | 88/100 [13:51<01:52,  9.35s/it]2026-08-31 19:25:49,743 - INFO - Epoch 89 | Train Loss: 0.3121, Train Acc: 0.8593 | Val Loss: 0.3925
2026-08-31 19:25:49,743 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8242, AUC = 0.9097

 89%|████████▉ | 89/100 [14:01<01:42,  9.35s/it]2026-08-31 19:25:59,056 - INFO - Epoch 90 | Train Loss: 0.2727, Train Acc: 0.8995 | Val Loss: 0.4144
2026-08-31 19:25:59,056 - INFO - Val metrics: Acc = 0.8121 | F1 = 0.8096, AUC = 0.9133

 90%|█████████ | 90/100 [14:10<01:33,  9.34s/it]2026-08-31 19:26:08,099 - INFO - Epoch 91 | Train Loss: 0.3461, Train Acc: 0.8492 | Val Loss: 0.4052
2026-08-31 19:26:08,099 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8167, AUC = 0.9153

 91%|█████████ | 91/100 [14:19<01:23,  9.25s/it]2026-08-31 19:26:17,406 - INFO - Epoch 92 | Train Loss: 0.2972, Train Acc: 0.8861 | Val Loss: 0.3970
2026-08-31 19:26:17,406 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8172, AUC = 0.9133

 92%|█████████▏| 92/100 [14:28<01:14,  9.27s/it]2026-08-31 19:26:26,650 - INFO - Epoch 93 | Train Loss: 0.2936, Train Acc: 0.8693 | Val Loss: 0.3786
2026-08-31 19:26:26,651 - INFO - Val metrics: Acc = 0.8456 | F1 = 0.8449, AUC = 0.9133

 93%|█████████▎| 93/100 [14:38<01:04,  9.26s/it]2026-08-31 19:26:35,936 - INFO - Epoch 94 | Train Loss: 0.2932, Train Acc: 0.8626 | Val Loss: 0.3897
2026-08-31 19:26:35,937 - INFO - Val metrics: Acc = 0.8322 | F1 = 0.8311, AUC = 0.9200

 94%|█████████▍| 94/100 [14:47<00:55,  9.27s/it]2026-08-31 19:26:45,138 - INFO - Epoch 95 | Train Loss: 0.2790, Train Acc: 0.8894 | Val Loss: 0.3896
2026-08-31 19:26:45,138 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8242, AUC = 0.9207

 95%|█████████▌| 95/100 [14:56<00:46,  9.25s/it]2026-08-31 19:26:54,415 - INFO - Epoch 96 | Train Loss: 0.2705, Train Acc: 0.8861 | Val Loss: 0.3604
2026-08-31 19:26:54,415 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8523, AUC = 0.9202

2026-08-31 19:26:54,415 - INFO - Loss IMPROVEMENT!

 96%|█████████▌| 96/100 [15:05<00:37,  9.26s/it]2026-08-31 19:27:03,639 - INFO - Epoch 97 | Train Loss: 0.3269, Train Acc: 0.8693 | Val Loss: 0.3945
2026-08-31 19:27:03,640 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8242, AUC = 0.9146

 97%|█████████▋| 97/100 [15:15<00:27,  9.25s/it]2026-08-31 19:27:12,977 - INFO - Epoch 98 | Train Loss: 0.2989, Train Acc: 0.8811 | Val Loss: 0.3917
2026-08-31 19:27:12,977 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8242, AUC = 0.9141

 98%|█████████▊| 98/100 [15:24<00:18,  9.27s/it]2026-08-31 19:27:22,343 - INFO - Epoch 99 | Train Loss: 0.2455, Train Acc: 0.8945 | Val Loss: 0.4094
2026-08-31 19:27:22,343 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8232, AUC = 0.9193

 99%|█████████▉| 99/100 [15:33<00:09,  9.30s/it]2026-08-31 19:27:31,642 - INFO - Epoch 100 | Train Loss: 0.3031, Train Acc: 0.8878 | Val Loss: 0.4151
2026-08-31 19:27:31,642 - INFO - Val metrics: Acc = 0.8322 | F1 = 0.8303, AUC = 0.9207

100%|██████████| 100/100 [15:43<00:00,  9.43s/it]
2026-08-31 19:27:31,643 - INFO -
--- FOLD 3 ---

2026-08-31 19:27:38,023 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 64, heads=4)
  (residual_proj1): Linear(in_features=10, out_features=256, bias=True)
  (residual_proj2): Linear(in_features=256, out_features=64, bias=True)
  (gat2): GATv2Conv(256, 64, heads=1)
  (bn1): GraphNorm(256)
  (bn2): GraphNorm(64)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=32, out_features=2, bias=True)
  )
  (dropout): Dropout(p=0.3, inplace=False)
)

  0%|          | 0/100 [00:00<?, ?it/s]2026-08-31 19:27:47,408 - INFO - Epoch 01 | Train Loss: 0.7877, Train Acc: 0.5310 | Val Loss: 0.6920
2026-08-31 19:27:47,408 - INFO - Val metrics: Acc = 0.5302 | F1 = 0.4279, AUC = 0.6079

2026-08-31 19:27:47,408 - INFO - Loss IMPROVEMENT!

  1%|          | 1/100 [00:09<15:31,  9.41s/it]2026-08-31 19:27:56,766 - INFO - Epoch 02 | Train Loss: 0.7029, Train Acc: 0.5444 | Val Loss: 0.6883
2026-08-31 19:27:56,767 - INFO - Val metrics: Acc = 0.5503 | F1 = 0.4823, AUC = 0.5906

2026-08-31 19:27:56,767 - INFO - Loss IMPROVEMENT!

  2%|▏         | 2/100 [00:18<15:21,  9.40s/it]2026-08-31 19:28:06,126 - INFO - Epoch 03 | Train Loss: 0.6818, Train Acc: 0.5544 | Val Loss: 0.6694
2026-08-31 19:28:06,126 - INFO - Val metrics: Acc = 0.6040 | F1 = 0.6022, AUC = 0.6461

2026-08-31 19:28:06,127 - INFO - Loss IMPROVEMENT!

  3%|▎         | 3/100 [00:28<15:07,  9.36s/it]2026-08-31 19:28:15,312 - INFO - Epoch 04 | Train Loss: 0.6876, Train Acc: 0.5695 | Val Loss: 0.6732
2026-08-31 19:28:15,312 - INFO - Val metrics: Acc = 0.5839 | F1 = 0.5565, AUC = 0.6587

  4%|▍         | 4/100 [00:37<14:51,  9.29s/it]2026-08-31 19:28:24,499 - INFO - Epoch 05 | Train Loss: 0.6774, Train Acc: 0.6047 | Val Loss: 0.6540
2026-08-31 19:28:24,499 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6168, AUC = 0.6719

2026-08-31 19:28:24,499 - INFO - Loss IMPROVEMENT!

  5%|▌         | 5/100 [00:46<14:39,  9.25s/it]2026-08-31 19:28:33,870 - INFO - Epoch 06 | Train Loss: 0.6739, Train Acc: 0.5762 | Val Loss: 0.6559
2026-08-31 19:28:33,870 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6378, AUC = 0.6672

  6%|▌         | 6/100 [00:55<14:33,  9.29s/it]2026-08-31 19:28:44,090 - INFO - Epoch 07 | Train Loss: 0.6639, Train Acc: 0.6097 | Val Loss: 0.6524
2026-08-31 19:28:44,090 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.6174, AUC = 0.6695

2026-08-31 19:28:44,090 - INFO - Loss IMPROVEMENT!

  7%|▋         | 7/100 [01:06<14:52,  9.60s/it]2026-08-31 19:28:54,624 - INFO - Epoch 08 | Train Loss: 0.6628, Train Acc: 0.6097 | Val Loss: 0.6655
2026-08-31 19:28:54,625 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.6056, AUC = 0.6622

  8%|▊         | 8/100 [01:16<15:10,  9.89s/it]2026-08-31 19:29:04,268 - INFO - Epoch 09 | Train Loss: 0.6589, Train Acc: 0.6231 | Val Loss: 0.6527
2026-08-31 19:29:04,268 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6673, AUC = 0.6739

  9%|▉         | 9/100 [01:26<14:53,  9.82s/it]2026-08-31 19:29:13,690 - INFO - Epoch 10 | Train Loss: 0.6615, Train Acc: 0.6214 | Val Loss: 0.6595
2026-08-31 19:29:13,690 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6248, AUC = 0.6818

 10%|█         | 10/100 [01:35<14:32,  9.69s/it]2026-08-31 19:29:23,129 - INFO - Epoch 11 | Train Loss: 0.6570, Train Acc: 0.6164 | Val Loss: 0.6438
2026-08-31 19:29:23,129 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6420, AUC = 0.6832

2026-08-31 19:29:23,129 - INFO - Loss IMPROVEMENT!

 11%|█         | 11/100 [01:45<14:15,  9.62s/it]2026-08-31 19:29:32,630 - INFO - Epoch 12 | Train Loss: 0.6728, Train Acc: 0.6114 | Val Loss: 0.6586
2026-08-31 19:29:32,630 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6502, AUC = 0.6609

 12%|█▏        | 12/100 [01:54<14:03,  9.58s/it]2026-08-31 19:29:41,897 - INFO - Epoch 13 | Train Loss: 0.6583, Train Acc: 0.6097 | Val Loss: 0.6498
2026-08-31 19:29:41,897 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6193, AUC = 0.6908

 13%|█▎        | 13/100 [02:03<13:45,  9.49s/it]2026-08-31 19:29:51,416 - INFO - Epoch 14 | Train Loss: 0.6550, Train Acc: 0.6533 | Val Loss: 0.6490
2026-08-31 19:29:51,416 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6433, AUC = 0.6944

 14%|█▍        | 14/100 [02:13<13:36,  9.50s/it]2026-08-31 19:30:00,802 - INFO - Epoch 15 | Train Loss: 0.6590, Train Acc: 0.6164 | Val Loss: 0.6536
2026-08-31 19:30:00,803 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.6233, AUC = 0.6932

 15%|█▌        | 15/100 [02:22<13:24,  9.46s/it]2026-08-31 19:30:10,267 - INFO - Epoch 16 | Train Loss: 0.6520, Train Acc: 0.6298 | Val Loss: 0.6610
2026-08-31 19:30:10,267 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6271, AUC = 0.6885

 16%|█▌        | 16/100 [02:32<13:14,  9.46s/it]2026-08-31 19:30:19,549 - INFO - Epoch 17 | Train Loss: 0.6378, Train Acc: 0.6834 | Val Loss: 0.6627
2026-08-31 19:30:19,549 - INFO - Val metrics: Acc = 0.6309 | F1 = 0.6153, AUC = 0.6816

 17%|█▋        | 17/100 [02:41<13:00,  9.41s/it]2026-08-31 19:30:28,647 - INFO - Epoch 18 | Train Loss: 0.6488, Train Acc: 0.6332 | Val Loss: 0.6479
2026-08-31 19:30:28,647 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6643, AUC = 0.6883

 18%|█▊        | 18/100 [02:50<12:43,  9.32s/it]2026-08-31 19:30:37,870 - INFO - Epoch 19 | Train Loss: 0.6388, Train Acc: 0.6332 | Val Loss: 0.6476
2026-08-31 19:30:37,870 - INFO - Val metrics: Acc = 0.6779 | F1 = 0.6713, AUC = 0.6952

 19%|█▉        | 19/100 [02:59<12:32,  9.29s/it]2026-08-31 19:30:47,137 - INFO - Epoch 20 | Train Loss: 0.6286, Train Acc: 0.6583 | Val Loss: 0.6520
2026-08-31 19:30:47,137 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6271, AUC = 0.7121

 20%|██        | 20/100 [03:09<12:22,  9.28s/it]2026-08-31 19:30:56,616 - INFO - Epoch 21 | Train Loss: 0.6329, Train Acc: 0.6566 | Val Loss: 0.6317
2026-08-31 19:30:56,616 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6872, AUC = 0.7115

2026-08-31 19:30:56,616 - INFO - Loss IMPROVEMENT!

 21%|██        | 21/100 [03:18<12:18,  9.34s/it]2026-08-31 19:31:05,981 - INFO - Epoch 22 | Train Loss: 0.6481, Train Acc: 0.6566 | Val Loss: 0.6449
2026-08-31 19:31:05,982 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6809, AUC = 0.6995

 22%|██▏       | 22/100 [03:27<12:09,  9.35s/it]2026-08-31 19:31:15,381 - INFO - Epoch 23 | Train Loss: 0.6330, Train Acc: 0.6499 | Val Loss: 0.6422
2026-08-31 19:31:15,382 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6331, AUC = 0.7121

 23%|██▎       | 23/100 [03:37<12:00,  9.36s/it]2026-08-31 19:31:24,523 - INFO - Epoch 24 | Train Loss: 0.6015, Train Acc: 0.6700 | Val Loss: 0.6357
2026-08-31 19:31:24,523 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6486, AUC = 0.7050

 24%|██▍       | 24/100 [03:46<11:46,  9.30s/it]2026-08-31 19:31:33,580 - INFO - Epoch 25 | Train Loss: 0.6166, Train Acc: 0.6700 | Val Loss: 0.6313
2026-08-31 19:31:33,580 - INFO - Val metrics: Acc = 0.6644 | F1 = 0.6610, AUC = 0.7175

2026-08-31 19:31:33,580 - INFO - Loss IMPROVEMENT!

 25%|██▌       | 25/100 [03:55<11:32,  9.23s/it]2026-08-31 19:31:42,923 - INFO - Epoch 26 | Train Loss: 0.6135, Train Acc: 0.6734 | Val Loss: 0.6120
2026-08-31 19:31:42,923 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6799, AUC = 0.7357

2026-08-31 19:31:42,924 - INFO - Loss IMPROVEMENT!

 26%|██▌       | 26/100 [04:04<11:25,  9.26s/it]2026-08-31 19:31:52,151 - INFO - Epoch 27 | Train Loss: 0.6275, Train Acc: 0.6415 | Val Loss: 0.6284
2026-08-31 19:31:52,151 - INFO - Val metrics: Acc = 0.6577 | F1 = 0.6470, AUC = 0.7256

 27%|██▋       | 27/100 [04:14<11:15,  9.25s/it]2026-08-31 19:32:01,441 - INFO - Epoch 28 | Train Loss: 0.5956, Train Acc: 0.6817 | Val Loss: 0.6065
2026-08-31 19:32:01,441 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6762, AUC = 0.7555

2026-08-31 19:32:01,441 - INFO - Loss IMPROVEMENT!

 28%|██▊       | 28/100 [04:23<11:06,  9.26s/it]2026-08-31 19:32:10,750 - INFO - Epoch 29 | Train Loss: 0.5854, Train Acc: 0.7085 | Val Loss: 0.6151
2026-08-31 19:32:10,750 - INFO - Val metrics: Acc = 0.6443 | F1 = 0.6364, AUC = 0.7371

 29%|██▉       | 29/100 [04:32<10:58,  9.28s/it]2026-08-31 19:32:20,010 - INFO - Epoch 30 | Train Loss: 0.6038, Train Acc: 0.6985 | Val Loss: 0.6244
2026-08-31 19:32:20,010 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6330, AUC = 0.7721

 30%|███       | 30/100 [04:41<10:48,  9.27s/it]2026-08-31 19:32:29,255 - INFO - Epoch 31 | Train Loss: 0.5783, Train Acc: 0.6817 | Val Loss: 0.5971
2026-08-31 19:32:29,255 - INFO - Val metrics: Acc = 0.6913 | F1 = 0.6823, AUC = 0.7780

2026-08-31 19:32:29,255 - INFO - Loss IMPROVEMENT!

 31%|███       | 31/100 [04:51<10:39,  9.27s/it]2026-08-31 19:32:38,606 - INFO - Epoch 32 | Train Loss: 0.5889, Train Acc: 0.6851 | Val Loss: 0.6240
2026-08-31 19:32:38,606 - INFO - Val metrics: Acc = 0.6376 | F1 = 0.6079, AUC = 0.7818

 32%|███▏      | 32/100 [05:00<10:31,  9.29s/it]2026-08-31 19:32:47,896 - INFO - Epoch 33 | Train Loss: 0.5739, Train Acc: 0.6985 | Val Loss: 0.6213
2026-08-31 19:32:47,896 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6483, AUC = 0.7908

 33%|███▎      | 33/100 [05:09<10:22,  9.29s/it]2026-08-31 19:32:57,223 - INFO - Epoch 34 | Train Loss: 0.5590, Train Acc: 0.7002 | Val Loss: 0.5912
2026-08-31 19:32:57,223 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6693, AUC = 0.7966

2026-08-31 19:32:57,223 - INFO - Loss IMPROVEMENT!

 34%|███▍      | 34/100 [05:19<10:13,  9.30s/it]2026-08-31 19:33:06,586 - INFO - Epoch 35 | Train Loss: 0.5953, Train Acc: 0.6968 | Val Loss: 0.5960
2026-08-31 19:33:06,586 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6730, AUC = 0.7982

 35%|███▌      | 35/100 [05:28<10:05,  9.32s/it]2026-08-31 19:33:15,737 - INFO - Epoch 36 | Train Loss: 0.5273, Train Acc: 0.7605 | Val Loss: 0.6322
2026-08-31 19:33:15,737 - INFO - Val metrics: Acc = 0.6510 | F1 = 0.6160, AUC = 0.8436

 36%|███▌      | 36/100 [05:37<09:53,  9.27s/it]2026-08-31 19:33:25,009 - INFO - Epoch 37 | Train Loss: 0.5766, Train Acc: 0.7102 | Val Loss: 0.6352
2026-08-31 19:33:25,009 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.5544, AUC = 0.8207

 37%|███▋      | 37/100 [05:46<09:43,  9.27s/it]2026-08-31 19:33:34,232 - INFO - Epoch 38 | Train Loss: 0.5319, Train Acc: 0.7471 | Val Loss: 0.6268
2026-08-31 19:33:34,232 - INFO - Val metrics: Acc = 0.6107 | F1 = 0.5544, AUC = 0.8472

 38%|███▊      | 38/100 [05:56<09:33,  9.26s/it]2026-08-31 19:33:43,476 - INFO - Epoch 39 | Train Loss: 0.5147, Train Acc: 0.7621 | Val Loss: 0.4872
2026-08-31 19:33:43,476 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7471, AUC = 0.8795

2026-08-31 19:33:43,476 - INFO - Loss IMPROVEMENT!

 39%|███▉      | 39/100 [06:05<09:24,  9.25s/it]2026-08-31 19:33:52,771 - INFO - Epoch 40 | Train Loss: 0.5649, Train Acc: 0.7236 | Val Loss: 0.5177
2026-08-31 19:33:52,771 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7451, AUC = 0.8650

 40%|████      | 40/100 [06:14<09:15,  9.26s/it]2026-08-31 19:34:02,130 - INFO - Epoch 41 | Train Loss: 0.5184, Train Acc: 0.7521 | Val Loss: 0.5655
2026-08-31 19:34:02,131 - INFO - Val metrics: Acc = 0.7114 | F1 = 0.6913, AUC = 0.8605

 41%|████      | 41/100 [06:24<09:08,  9.29s/it]2026-08-31 19:34:11,598 - INFO - Epoch 42 | Train Loss: 0.5180, Train Acc: 0.7638 | Val Loss: 0.5739
2026-08-31 19:34:11,598 - INFO - Val metrics: Acc = 0.6711 | F1 = 0.6428, AUC = 0.8888

 42%|████▏     | 42/100 [06:33<09:02,  9.35s/it]2026-08-31 19:34:22,121 - INFO - Epoch 43 | Train Loss: 0.5264, Train Acc: 0.7655 | Val Loss: 0.5076
2026-08-31 19:34:22,121 - INFO - Val metrics: Acc = 0.7517 | F1 = 0.7439, AUC = 0.8724

 43%|████▎     | 43/100 [06:44<09:12,  9.70s/it]2026-08-31 19:34:31,314 - INFO - Epoch 44 | Train Loss: 0.4937, Train Acc: 0.7538 | Val Loss: 0.4899
2026-08-31 19:34:31,315 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7363, AUC = 0.8874

 44%|████▍     | 44/100 [06:53<08:54,  9.55s/it]2026-08-31 19:34:40,641 - INFO - Epoch 45 | Train Loss: 0.4680, Train Acc: 0.7822 | Val Loss: 0.5672
2026-08-31 19:34:40,641 - INFO - Val metrics: Acc = 0.6846 | F1 = 0.6544, AUC = 0.9058

 45%|████▌     | 45/100 [07:02<08:41,  9.48s/it]2026-08-31 19:34:49,848 - INFO - Epoch 46 | Train Loss: 0.4746, Train Acc: 0.7856 | Val Loss: 0.6492
2026-08-31 19:34:49,848 - INFO - Val metrics: Acc = 0.6174 | F1 = 0.5596, AUC = 0.9031

 46%|████▌     | 46/100 [07:11<08:27,  9.40s/it]2026-08-31 19:34:59,196 - INFO - Epoch 47 | Train Loss: 0.4540, Train Acc: 0.7906 | Val Loss: 0.5270
2026-08-31 19:34:59,197 - INFO - Val metrics: Acc = 0.7181 | F1 = 0.6996, AUC = 0.9110

 47%|████▋     | 47/100 [07:21<08:17,  9.38s/it]2026-08-31 19:35:08,498 - INFO - Epoch 48 | Train Loss: 0.4786, Train Acc: 0.7722 | Val Loss: 0.4940
2026-08-31 19:35:08,498 - INFO - Val metrics: Acc = 0.7450 | F1 = 0.7318, AUC = 0.9148

 48%|████▊     | 48/100 [07:30<08:06,  9.36s/it]2026-08-31 19:35:17,758 - INFO - Epoch 49 | Train Loss: 0.4535, Train Acc: 0.7856 | Val Loss: 0.4571
2026-08-31 19:35:17,758 - INFO - Val metrics: Acc = 0.7852 | F1 = 0.7779, AUC = 0.9207

2026-08-31 19:35:17,758 - INFO - Loss IMPROVEMENT!

 49%|████▉     | 49/100 [07:39<07:55,  9.33s/it]2026-08-31 19:35:26,824 - INFO - Epoch 50 | Train Loss: 0.4529, Train Acc: 0.7839 | Val Loss: 0.4616
2026-08-31 19:35:26,825 - INFO - Val metrics: Acc = 0.7987 | F1 = 0.7937, AUC = 0.9241

 50%|█████     | 50/100 [07:48<07:42,  9.25s/it]2026-08-31 19:35:35,830 - INFO - Epoch 51 | Train Loss: 0.4963, Train Acc: 0.7638 | Val Loss: 0.5411
2026-08-31 19:35:35,830 - INFO - Val metrics: Acc = 0.6980 | F1 = 0.6745, AUC = 0.9005

 51%|█████     | 51/100 [07:57<07:29,  9.18s/it]2026-08-31 19:35:45,350 - INFO - Epoch 52 | Train Loss: 0.4670, Train Acc: 0.7772 | Val Loss: 0.3727
2026-08-31 19:35:45,350 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8657, AUC = 0.9301

2026-08-31 19:35:45,350 - INFO - Loss IMPROVEMENT!

 52%|█████▏    | 52/100 [08:07<07:25,  9.28s/it]2026-08-31 19:35:55,059 - INFO - Epoch 53 | Train Loss: 0.4324, Train Acc: 0.8074 | Val Loss: 0.4560
2026-08-31 19:35:55,059 - INFO - Val metrics: Acc = 0.7584 | F1 = 0.7489, AUC = 0.9265

 53%|█████▎    | 53/100 [08:17<07:22,  9.41s/it]2026-08-31 19:36:05,165 - INFO - Epoch 54 | Train Loss: 0.4188, Train Acc: 0.8174 | Val Loss: 0.3436
2026-08-31 19:36:05,165 - INFO - Val metrics: Acc = 0.8859 | F1 = 0.8859, AUC = 0.9407

2026-08-31 19:36:05,165 - INFO - Loss IMPROVEMENT!

 54%|█████▍    | 54/100 [08:27<07:22,  9.62s/it]2026-08-31 19:36:15,125 - INFO - Epoch 55 | Train Loss: 0.4203, Train Acc: 0.8157 | Val Loss: 0.4485
2026-08-31 19:36:15,125 - INFO - Val metrics: Acc = 0.7785 | F1 = 0.7716, AUC = 0.9308

 55%|█████▌    | 55/100 [08:37<07:17,  9.72s/it]2026-08-31 19:36:24,840 - INFO - Epoch 56 | Train Loss: 0.4276, Train Acc: 0.8090 | Val Loss: 0.4222
2026-08-31 19:36:24,841 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8161, AUC = 0.9223

 56%|█████▌    | 56/100 [08:46<07:07,  9.72s/it]2026-08-31 19:36:34,324 - INFO - Epoch 57 | Train Loss: 0.3943, Train Acc: 0.8308 | Val Loss: 0.3916
2026-08-31 19:36:34,324 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8237, AUC = 0.9268

 57%|█████▋    | 57/100 [08:56<06:54,  9.65s/it]2026-08-31 19:36:43,763 - INFO - Epoch 58 | Train Loss: 0.4201, Train Acc: 0.8291 | Val Loss: 0.4132
2026-08-31 19:36:43,763 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8167, AUC = 0.9247

 58%|█████▊    | 58/100 [09:05<06:42,  9.59s/it]2026-08-31 19:36:53,021 - INFO - Epoch 59 | Train Loss: 0.3940, Train Acc: 0.8258 | Val Loss: 0.3917
2026-08-31 19:36:53,021 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8161, AUC = 0.9333

 59%|█████▉    | 59/100 [09:14<06:28,  9.49s/it]2026-08-31 19:37:02,129 - INFO - Epoch 60 | Train Loss: 0.3715, Train Acc: 0.8275 | Val Loss: 0.3591
2026-08-31 19:37:02,129 - INFO - Val metrics: Acc = 0.8188 | F1 = 0.8155, AUC = 0.9515

 60%|██████    | 60/100 [09:24<06:14,  9.37s/it]2026-08-31 19:37:11,379 - INFO - Epoch 61 | Train Loss: 0.3859, Train Acc: 0.8392 | Val Loss: 0.3776
2026-08-31 19:37:11,379 - INFO - Val metrics: Acc = 0.8255 | F1 = 0.8226, AUC = 0.9368

 61%|██████    | 61/100 [09:33<06:04,  9.34s/it]2026-08-31 19:37:20,612 - INFO - Epoch 62 | Train Loss: 0.3749, Train Acc: 0.8291 | Val Loss: 0.3594
2026-08-31 19:37:20,612 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8380, AUC = 0.9389

 62%|██████▏   | 62/100 [09:42<05:53,  9.31s/it]2026-08-31 19:37:29,910 - INFO - Epoch 63 | Train Loss: 0.3428, Train Acc: 0.8459 | Val Loss: 0.3796
2026-08-31 19:37:29,910 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8581, AUC = 0.9326

 63%|██████▎   | 63/100 [09:51<05:44,  9.30s/it]2026-08-31 19:37:39,178 - INFO - Epoch 64 | Train Loss: 0.3654, Train Acc: 0.8409 | Val Loss: 0.3826
2026-08-31 19:37:39,178 - INFO - Val metrics: Acc = 0.8322 | F1 = 0.8303, AUC = 0.9317

 64%|██████▍   | 64/100 [10:01<05:34,  9.29s/it]2026-08-31 19:37:48,368 - INFO - Epoch 65 | Train Loss: 0.4030, Train Acc: 0.8141 | Val Loss: 0.3834
2026-08-31 19:37:48,368 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8363, AUC = 0.9389

 65%|██████▌   | 65/100 [10:10<05:24,  9.26s/it]2026-08-31 19:37:57,741 - INFO - Epoch 66 | Train Loss: 0.3528, Train Acc: 0.8509 | Val Loss: 0.3342
2026-08-31 19:37:57,741 - INFO - Val metrics: Acc = 0.8792 | F1 = 0.8791, AUC = 0.9400

2026-08-31 19:37:57,741 - INFO - Loss IMPROVEMENT!

 66%|██████▌   | 66/100 [10:19<05:16,  9.30s/it]2026-08-31 19:38:07,072 - INFO - Epoch 67 | Train Loss: 0.3261, Train Acc: 0.8693 | Val Loss: 0.3364
2026-08-31 19:38:07,072 - INFO - Val metrics: Acc = 0.8725 | F1 = 0.8724, AUC = 0.9391

 67%|██████▋   | 67/100 [10:29<05:07,  9.31s/it]2026-08-31 19:38:16,597 - INFO - Epoch 68 | Train Loss: 0.3314, Train Acc: 0.8677 | Val Loss: 0.3545
2026-08-31 19:38:16,597 - INFO - Val metrics: Acc = 0.8389 | F1 = 0.8386, AUC = 0.9368

 68%|██████▊   | 68/100 [10:38<04:59,  9.37s/it]2026-08-31 19:38:26,056 - INFO - Epoch 69 | Train Loss: 0.3048, Train Acc: 0.8794 | Val Loss: 0.3064
2026-08-31 19:38:26,057 - INFO - Val metrics: Acc = 0.8725 | F1 = 0.8725, AUC = 0.9483

2026-08-31 19:38:26,057 - INFO - Loss IMPROVEMENT!

 69%|██████▉   | 69/100 [10:48<04:51,  9.40s/it]2026-08-31 19:38:35,315 - INFO - Epoch 70 | Train Loss: 0.3028, Train Acc: 0.8760 | Val Loss: 0.3337
2026-08-31 19:38:35,315 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8653, AUC = 0.9414

 70%|███████   | 70/100 [10:57<04:40,  9.36s/it]2026-08-31 19:38:44,550 - INFO - Epoch 71 | Train Loss: 0.2961, Train Acc: 0.8844 | Val Loss: 0.3213
2026-08-31 19:38:44,550 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8655, AUC = 0.9420

 71%|███████   | 71/100 [11:06<04:30,  9.32s/it]2026-08-31 19:38:53,625 - INFO - Epoch 72 | Train Loss: 0.3473, Train Acc: 0.8526 | Val Loss: 0.3589
2026-08-31 19:38:53,625 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8578, AUC = 0.9348

 72%|███████▏  | 72/100 [11:15<04:18,  9.25s/it]2026-08-31 19:39:03,292 - INFO - Epoch 73 | Train Loss: 0.3206, Train Acc: 0.8610 | Val Loss: 0.3311
2026-08-31 19:39:03,292 - INFO - Val metrics: Acc = 0.8591 | F1 = 0.8587, AUC = 0.9400

 73%|███████▎  | 73/100 [11:25<04:13,  9.37s/it]2026-08-31 19:39:12,607 - INFO - Epoch 74 | Train Loss: 0.3230, Train Acc: 0.8543 | Val Loss: 0.3201
2026-08-31 19:39:12,608 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8656, AUC = 0.9431

 74%|███████▍  | 74/100 [11:34<04:03,  9.36s/it]2026-08-31 19:39:22,109 - INFO - Epoch 75 | Train Loss: 0.3178, Train Acc: 0.8660 | Val Loss: 0.3277
2026-08-31 19:39:22,110 - INFO - Val metrics: Acc = 0.8658 | F1 = 0.8655, AUC = 0.9407

 75%|███████▌  | 75/100 [11:44<03:54,  9.40s/it]2026-08-31 19:39:31,377 - INFO - Epoch 76 | Train Loss: 0.2992, Train Acc: 0.8727 | Val Loss: 0.3191
2026-08-31 19:39:31,377 - INFO - Val metrics: Acc = 0.8926 | F1 = 0.8924, AUC = 0.9458

 76%|███████▌  | 76/100 [11:53<03:44,  9.36s/it]2026-08-31 19:39:40,911 - INFO - Epoch 77 | Train Loss: 0.2781, Train Acc: 0.8878 | Val Loss: 0.3018
2026-08-31 19:39:40,911 - INFO - Val metrics: Acc = 0.8859 | F1 = 0.8859, AUC = 0.9465

2026-08-31 19:39:40,911 - INFO - Loss IMPROVEMENT!

 77%|███████▋  | 77/100 [12:02<03:36,  9.41s/it]2026-08-31 19:39:51,511 - INFO - Epoch 78 | Train Loss: 0.2890, Train Acc: 0.8811 | Val Loss: 0.3116
2026-08-31 19:39:51,511 - INFO - Val metrics: Acc = 0.8792 | F1 = 0.8792, AUC = 0.9479

 78%|███████▊  | 78/100 [12:13<03:34,  9.77s/it]2026-08-31 19:40:01,420 - INFO - Epoch 79 | Train Loss: 0.3150, Train Acc: 0.8509 | Val Loss: 0.3185
2026-08-31 19:40:01,421 - INFO - Val metrics: Acc = 0.8792 | F1 = 0.8785, AUC = 0.9490

 79%|███████▉  | 79/100 [12:23<03:26,  9.81s/it]2026-08-31 19:40:11,340 - INFO - Epoch 80 | Train Loss: 0.2852, Train Acc: 0.8844 | Val Loss: 0.3623
2026-08-31 19:40:11,340 - INFO - Val metrics: Acc = 0.8523 | F1 = 0.8515, AUC = 0.9418

 80%|████████  | 80/100 [12:33<03:16,  9.84s/it]2026-08-31 19:40:21,364 - INFO - Epoch 81 | Train Loss: 0.2721, Train Acc: 0.8777 | Val Loss: 0.2917
2026-08-31 19:40:21,364 - INFO - Val metrics: Acc = 0.8792 | F1 = 0.8791, AUC = 0.9541

2026-08-31 19:40:21,365 - INFO - Loss IMPROVEMENT!
"""