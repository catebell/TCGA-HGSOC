import os

import joblib
import pandas as pd
import torch
from scipy.spatial.distance import cosine, euclidean
from torch_geometric.loader import DataLoader

import task_target
from models.MultiOmicGAT import MultiOmicGAT
from models.MultiOmicGATSurvival import MultiOmicGATSurvival
from PatientTissueDataset import PatientTissueDataset
from train_eval_utils import nan_imputation_and_features_normalization


""""" setup parameters """""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task = task_target.TASK
target_column = ""

if task == 'classification':
    target_column = task_target.TARGET_COL
    print(f"Starting pairs comparison using target feature: {target_column}\n")
elif task == 'survival':
    target_column = task_target.SURVIVAL_EVENT_COL  # placeholder, won't actually be used here
    print(f"Starting pairs comparison on survival prediction\n")

output_dir = os.path.join(".", "results")
os.makedirs(output_dir, exist_ok=True)
save_csv_path = os.path.join(output_dir, "patient_tissues_heterogeneity_metrics.csv")

path_to_tissues_pairs= os.path.join("..", "3_train_test_split", "tissues_pairs.csv")
path_to_clinical = os.path.join("..", "2_preprocessing", "preprocessed_clinical_data.tsv")
path_to_graphs_dir = os.path.join("..", "4_samples_graph_creation", "processed_graphs")

path_to_pairs_graphs = os.path.join(path_to_graphs_dir, "pairs")

model_dict_path = os.path.join("..", "5_training", "trained_models", "best_MultiOmicGat_fold1.pt")

df_pairs_mapping = pd.read_csv(path_to_tissues_pairs)
df_clinical = pd.read_csv(path_to_clinical, sep="\t")

prep_path = os.path.join("..", "5_training", "trained_models", "clinical_preprocessors.joblib")
preprocessors = joblib.load(prep_path)


""""" dataset initialization and clinical processing """""

# clinical df filtered to keep only patients with tissues pairs
paired_patient_ids = df_pairs_mapping["Patient_ID"].unique()
df_clinical_pairs = df_clinical[df_clinical["bcr_patient_barcode"].isin(paired_patient_ids)].copy()

scalers = preprocessors['scalers']
imputers = preprocessors['imputers']
categories = preprocessors['categories']

df_clinical_processed = nan_imputation_and_features_normalization(
    df=df_clinical_pairs,
    target_col=target_column,
    is_train=False,
    scalers=scalers,
    imputers=imputers,
    categories=categories,
)

pairs_dataset = PatientTissueDataset(
    graphs_dir=path_to_pairs_graphs,
    clinical_df=df_clinical_processed,
    task=task,
    target_col=target_column
)

pairs_loader = DataLoader(pairs_dataset, batch_size=1, shuffle=False)


print(f"Loading model: {model_dict_path}")
num_genes_features = pairs_dataset.num_features
num_clinical_features = len(pairs_dataset.clinical_feature_cols)

# model init with same training configuration
if task == 'classification':
    model = MultiOmicGAT(
Questi sono due paragrafi del contratto, cosa dice        in_node_features=num_genes_features,
        hidden_dim=64,
        out_channels=2,  # Adattare a seconda che l'head sia Survival (1) o Classification # todo check se è efficiente prenderlo automatizzato
        heads=4,
        dropout=0.3,
        num_clinical_features=0#num_clinical_features,
    ).to(device)

elif task == 'survival':
    model = MultiOmicGATSurvival(
        in_node_features=num_genes_features,
        hidden_dim=64,
        out_channels=1,
        heads=4,
        dropout=0.3,
        num_clinical_features=0#num_clinical_features,
    ).to(device)

model.load_state_dict(torch.load(model_dict_path, map_location=device))
model.eval()


print("Latent embeddings extraction...")
embeddings_list = []

with torch.no_grad():
    for data in pairs_loader:
        data = data.to(device)

        _, fused_emb = model(data.x, data.edge_index, data.batch, getattr(data, 'clinical_x', None))  # forward pass

        sample_id = data.sample_id[0]
        patient_id = "-".join(sample_id.split("-")[:3])
        tissue_code = sample_id.split("-")[3][:2]  # Es. '01' (Primary), '02' (Recurrent), '06' (Metastatic)

        embeddings_list.append({
            "patient_id": patient_id,
            "sample_id": sample_id,
            "tissue_code": tissue_code,
            "embedding": fused_emb.cpu().squeeze(0).numpy()
        })

df_emb = pd.DataFrame(embeddings_list)


""""" computing Tumoral Heterogeneity Shift per patient """""

print("Computing tumoral heterogeneity metrics (Pairwise Divergence)...")
comparison_results = []

grouped = df_emb.groupby("patient_id")

for patient_id, group in grouped:
    if len(group) >= 2:
        primary_samples = group[group["tissue_code"] == "01"]  # reference sample (primary)
        secondary_samples = group[group["tissue_code"] != "01"]  # secondary sample (recurrence/metastasis)

        if not primary_samples.empty and not secondary_samples.empty:
            sample_A = primary_samples.iloc[0]
            sample_B = secondary_samples.iloc[0]

            emb_A = sample_A["embedding"]
            emb_B = sample_B["embedding"]


            """"" distance metrics """""

            cos_dist = cosine(emb_A, emb_B)  # shift
            euc_dist = euclidean(emb_A, emb_B)  # distance magnitude

            df_clinical = df_clinical.set_index('bcr_patient_barcode')
            if task == 'classification':
                target_class = df_clinical.loc[patient_id, target_column]
            elif task == 'survival':
                target_class = None

            comparison_results.append({
                "patient_id": patient_id,
                "target_class": target_class,
                "primary_sample_id": sample_A["sample_id"],
                "secondary_sample_id": sample_B["sample_id"],
                "secondary_tissue_code": sample_B["tissue_code"],
                "cosine_distance": cos_dist,
                "euclidean_distance": euc_dist
            })

df_results = pd.DataFrame(comparison_results)

df_results.to_csv(save_csv_path, index=False)

print(f"\nAnalysis completed successfully on {len(df_results)} patients :)")
print(f"Results saved in: {save_csv_path}\n")
print(df_results.head())