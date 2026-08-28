import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


# ==============================================================================
# 1. DEFINIZIONE DELLA LOSS DI COX (PROPORTIONAL HAZARDS)
# ==============================================================================
class CoxPHLoss(nn.Module):
    """
    Negative Log-Likelihood della Cox Proportional Hazards Model per dati censurati.
    - risk_pred: log-hazard predetti dal modello (shape: [N, 1])
    - time: giorni alla morte o all'ultimo follow-up (shape: [N])
    - event: 1 se il paziente è deceduto (evento osservato), 0 se censurato (shape: [N])
    """

    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk_pred, time, event):
        # Ordina i dati per tempo decrescente (richiesto per il calcolo efficiente del risk set)
        order = torch.argsort(time, descending=True)
        risk_pred = risk_pred[order].squeeze(-1)
        event = event[order]

        # Calcolo del log del denominatore cumulativo (Risk Set)
        # log(sum(exp(risk_pred_j))) per tutti i j con t_j >= t_i
        max_risk = torch.max(risk_pred)  # Stabilità numerica
        exp_risk = torch.exp(risk_pred - max_risk)
        log_risk_set = torch.log(torch.cumsum(exp_risk, dim=0)) + max_risk

        # Si calcola la loss solo per i pazienti per cui si è verificato l'evento (uncensored)
        uncensored_loss = risk_pred - log_risk_set
        loss = -torch.sum(uncensored_loss * event) / (torch.sum(event) + 1e-7)
        return loss


# ==============================================================================
# 2. ARCHITETTURA DEL MODELLO MULTI-OMICO GAT + CLINICAL
# ==============================================================================
class MultiOmicSurvivalGAT(nn.Module):
    def __init__(self, in_node_channels, in_clinical_channels, hidden_dim=64, num_heads=4, dropout=0.3):
        super(MultiOmicSurvivalGAT, self).__init__()

        # --- BLOCCO GAT (Genomica / Omica) ---
        # Layer 1: Converte le feature del nodo (RNA + Copy Number) in embedding
        self.gat1 = GATConv(in_node_channels, hidden_dim, heads=num_heads, concat=True, dropout=dropout)

        # Layer 2: Mantiene l'attenzione tra i geni vicini
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)

        # Linear layer di proiezione dell'embedding genomico del grafo
        self.gene_embedding_projection = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 per via del pooling Mean + Max

        # --- BLOCCO CLINICO ---
        self.clinical_encoder = nn.Sequential(
            nn.Linear(in_clinical_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- FUSION & PREDIZIONE HAZARD ---
        # Unisce l'embedding del grafo dei geni (hidden_dim) con le feature cliniche (32)
        fusion_dim = hidden_dim + 32

        self.hazard_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Output: 1 singolo valore scalare di log-hazard (rischio)
        )

    def forward(self, x, edge_index, batch, clinical_features):
        """
        x: [N_nodi_totali, num_node_features] -> Feature dei geni (RNA + CNV)
        edge_index: [2, E] -> Matrice di adiacenza (es. STRING PPI)
        batch: [N_nodi_totali] -> Mappatura nodi -> grafi/pazienti nel mini-batch
        clinical_features: [B, num_clinical_features] -> Proprietà del grafo (paziente)
        """
        # 1. Message Passing con Graph Attention
        h = F.elu(self.gat1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.elu(self.gat2(h, edge_index))

        # 2. Graph Pooling (Converte i vettori dei singoli geni in 1 singolo vettore per il paziente)
        # Combinare Mean e Max Pooling cattura sia il profilo di rete generale che le alterazioni estreme (es. oncogeni amplificati)
        graph_mean = global_mean_pool(h, batch)
        graph_max = global_max_pool(h, batch)
        graph_embedding = torch.cat([graph_mean, graph_max], dim=1)
        graph_embedding = F.relu(self.gene_embedding_projection(graph_embedding))  # [B, hidden_dim]

        # 3. Processamento Feature Cliniche
        clin_embedding = self.clinical_encoder(clinical_features)  # [B, 32]

        # 4. Multi-modal Fusion
        fused_features = torch.cat([graph_embedding, clin_embedding], dim=1)  # [B, hidden_dim + 32]

        # 5. Predizione del Rischio (Log-Hazard)
        log_hazard = self.hazard_predictor(fused_features)
        return log_hazard


# ==============================================================================
# 3. STRUTTURA DATI PYTORCH GEOMETRIC (DATA LOADER SETUP)
# ==============================================================================
def create_patient_pyg_data(gene_x, edge_index, clinical_x, days_to_event, vital_status):
    """
    Crea un oggetto Data di PyG per un singolo paziente.
    - gene_x: Tensor [Num_Geni, Num_Gene_Features] (es. RNA, LOH, Major_CN, ...)
    - edge_index: Tensor [2, Num_Archi]
    - clinical_x: Tensor [Num_Clinical_Features] (es. Età, Stadio, FIGO, ...)
    - days_to_event: int/float (giorni alla morte o all'ultimo follow-up)
    - vital_status: int (1 = Dead, 0 = Alive/Censored)
    """
    data = Data(
        x=gene_x,
        edge_index=edge_index,
        clinical=clinical_x.unsqueeze(0),  # Aggiunge dimensione batch [1, num_features]
        y_time=torch.tensor([days_to_event], dtype=torch.float32),
        y_event=torch.tensor([vital_status], dtype=torch.float32)
    )
    return data


# ==============================================================================
# 4. ESEMPIO COMPLETO DI TRAINING LOOP
# ==============================================================================
if __name__ == "__main__":
    # Dimensione delle feature
    NUM_GENI = 2000
    NUM_GENE_FEATURES = 8  # es: RNA_expr, Major_CN, Minor_CN, Total_CN, LOH, CnLOH, HomDel, AllelicImbalance
    NUM_CLINICAL_FEATURES = 5  # es: Age, Stage_II, Stage_III, Stage_IV, Primary_vs_Metastatic
    BATCH_SIZE = 16

    # Sintetizziamo una matrice di adiacenza del grafo (es. STRING)
    dummy_edge_index = torch.randint(0, NUM_GENI, (2, 5000))

    # Creiamo un mini-batch di test con 16 pazienti
    data_list = []
    for _ in range(BATCH_SIZE):
        gene_x = torch.randn((NUM_GENI, NUM_GENE_FEATURES))
        clin_x = torch.randn(NUM_CLINICAL_FEATURES)
        days = torch.randint(100, 3000, (1,)).item()
        event = torch.randint(0, 2, (1,)).item()

        data_list.append(create_patient_pyg_data(gene_x, dummy_edge_index, clin_x, days, event))

    # DataLoader PyG Gestisce il batching automatico dei grafi
    from torch_geometric.loader import DataLoader

    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=True)

    # Inizializzazione Modello e Loss
    model = MultiOmicSurvivalGAT(
        in_node_channels=NUM_GENE_FEATURES,
        in_clinical_channels=NUM_CLINICAL_FEATURES,
        hidden_dim=64
    )
    criterion = CoxPHLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Singolo Passaggio di Training
    model.train()
    for batch in loader:
        optimizer.zero_grad()

        # Forward Pass
        # batch.x, batch.edge_index, e batch.batch sono gestiti in automatico da PyG
        log_hazard = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            clinical_features=batch.clinical
        )

        # Calcolo Loss della sopravvivenza
        loss = criterion(log_hazard, batch.y_time, batch.y_event)

        loss.backward()
        optimizer.step()

        print(f"Training Cox-PH Loss: {loss.item():.4f}")

"""
Valutazione del Modello: Il C-Index
Non usare l'Accuracy o il Mean Squared Error per valutare questo modello. 
Per i modelli di sopravvivenza, la metrica standard è il Concordance Index (C-Index, valore tra 0.5 e 
1.0), che misura se il modello assegna correttamente un rischio più alto a chi muore prima:
"""

from lifelines.utils import concordance_index

# Durante il ciclo di validation:
c_index = concordance_index(
    event_times=all_times.cpu().numpy(),
    predicted_scores=-all_hazards.cpu().numpy(), # Il segno meno serve perché un hazard alto equivale a un tempo di vita breve
    event_observed=all_events.cpu().numpy()
)
print(f"Validation C-Index: {c_index:.3f}")