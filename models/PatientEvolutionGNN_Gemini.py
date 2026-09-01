import torch
import torch.nn as nn


class PatientEvolutionGNN(nn.Module):
    def __init__(self, tissue_gat_model, hidden_dim, num_clinical_features=0, out_channels=1):
        super(PatientEvolutionGNN, self).__init__()

        # Riutilizziamo la tua MultiOmicGAT per estrarre gli embedding dei tessuti
        self.tissue_gat = tissue_gat_model

        # Dimensione dell'embedding generato dal tuo dual pooling (hidden_dim * 2)
        tissue_emb_dim = hidden_dim * 2

        # Encoder Clinico
        self.has_clinical = num_clinical_features > 0
        if self.has_clinical:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(num_clinical_features, 32),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )
            clin_dim = 32
        else:
            clin_dim = 0

        # Vettore concatenato: [Z_Primario || Z_Recidiva || Delta_Evolutivo || Clinica]
        total_patient_dim = (tissue_emb_dim * 3) + clin_dim

        # Classifier Finale per il Paziente
        self.patient_classifier = nn.Sequential(
            nn.Linear(total_patient_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, data_primary, data_recurrence, clinical_x=None):
        # 1. Estraiamo gli embedding dai due grafi usando la TUA GAT
        # Passiamo i singoli grafi (senza la parte di classificazione interna della MultiOmicGAT)
        _, z_primary = self.tissue_gat(data_primary.x, data_primary.edge_index, data_primary.batch)
        _, z_recurrence = self.tissue_gat(data_recurrence.x, data_recurrence.edge_index, data_recurrence.batch)

        # 2. Vettore di transizione/distanza evolutiva (rappresenta il ramo R dall'albero filogenetico)
        delta_evolution = z_recurrence - z_primary

        # 3. Costruzione dell'Embedding Complessivo del Paziente
        components = [z_primary, z_recurrence, delta_evolution]

        if self.has_clinical and clinical_x is not None:
            clin_emb = self.clinical_encoder(clinical_x)
            components.append(clin_emb)

        z_patient = torch.cat(components, dim=1)

        # 4. Prediction finale per paziente in base all'evoluzione dei suoi tessuti
        out = self.patient_classifier(z_patient)

        return out, z_patient