import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import task_target
from utils import print_details, nan_imputation_and_features_normalization
from PatientTissueDataset import PatientTissueDataset
from MultiOmicGAT import MultiOmicGAT


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('execution.log', mode='w'),
        logging.StreamHandler()
    ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Device: " + str(device))


""""" set general parameters """""

target_column = task_target.TARGET_COL

logging.info(f"Starting training using target feature: {target_column}\n")

path_to_clinical = os.path.join("..", "2_preprocessing", "preprocessed_clinical_data.tsv")
path_to_clinical_test = os.path.join("..", "3_train_test_split", "processed_clinical.tsv")  # TODO levare

paths_graphs_splits = {
    "train_val": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "train_val"),
    "test": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "test")
}

dir_to_save = os.path.join("trained_models")
os.makedirs(dir_to_save, exist_ok=True)


""""" loading clinical data """""

df_clinical_preprocessed = pd.read_csv(path_to_clinical, sep="\t")

# reverse preprocessing formatting
df_clinical_preprocessed = df_clinical_preprocessed.replace(-1, np.nan)
df_clinical_preprocessed = df_clinical_preprocessed.dropna(subset=[target_column])
df_clinical_preprocessed[target_column] = df_clinical_preprocessed[target_column].astype(int)


""""" loading datasets """""

'''
logging.info("Train Dataset init...")
train_dataset = PatientTissueDataset(paths_graphs_splits.get("train"), df_clinical_preprocessed, target_col=target_column)

logging.info("Val Dataset init...")
val_dataset = PatientTissueDataset(paths_graphs_splits.get("val"), df_clinical_preprocessed, target_col=target_column)

full_train_dataset = train_dataset + val_dataset
'''

logging.info("Train + Val Dataset init...")
train_val_dataset = PatientTissueDataset(paths_graphs_splits.get("train_val"), df_clinical_preprocessed, target_col=target_column)

logging.info("Test Dataset init...")
df_clinical_test = pd.read_csv(path_to_clinical_test, sep="\t")
test_dataset = PatientTissueDataset(paths_graphs_splits.get("test"), df_clinical_test, target_col=target_column)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print_details(train_val_dataset)


""""" set model parameters """""

num_genes_features = 10  # multiomics feature vector
num_clinical_features = len(train_val_dataset.clinical_feature_cols)  # total clinical features without the one to predict
logging.info(f"Num Clinical Features found : {num_clinical_features}")

epochs = 50


""""" functions """""

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data in loader:  # data = batch of graphs
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch, data.clinical_x)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

        total += data.num_graphs

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch, data.clinical_x)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)  # class with the highest prob
            correct += (pred == data.y).sum().item()

            all_targets.extend(data.y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += data.num_graphs

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    num_classes_in_target = len(np.unique(all_targets))
    if num_classes_in_target == 2:
        auc_val = roc_auc_score(all_targets, all_probs[:, 1])
    elif num_classes_in_target > 2:  # Multi-Class AUC --> OvR strategy (One-vs-Rest)
        auc_val = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    else:
        auc_val = 0.0

    metrics = {
        'acc': np.mean(all_preds == all_targets),  # acc = correct/total
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'auc': auc_val,
    }

    return total_loss/total, metrics


""""" k-fold train loop """""

y_labels = torch.tensor([train_val_dataset[i].y.item() for i in range(len(train_val_dataset))])
unique_labels = torch.unique(y_labels)
logging.info(f"Unique labels found in dataset: {unique_labels}")
num_classes = len(unique_labels)

# different weights to classes based on number of samples

class_counts = torch.bincount(y_labels)
total_samples = len(y_labels)
class_weights = total_samples / (len(class_counts) * class_counts.float())
class_weights = class_weights.to(device)

logging.info("classes_counts: ")
logging.info(class_counts)

logging.info("weights: ")
logging.info(class_weights)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_dataset, y_labels)):
    logging.info(f"\n--- FOLD {fold + 1} ---\n")

    """"" clinical fold specific processing """""

    # retrieving patient ids of current train/val splits from graphs sample ids
    train_samples_ids = ["-".join(train_val_dataset[i].sample_id.split("-")[:3]) for i in train_idx]
    val_samples_ids = ["-".join(train_val_dataset[i].sample_id.split("-")[:3]) for i in val_idx]

    df_clinical_train = df_clinical_preprocessed[df_clinical_preprocessed['bcr_patient_barcode'].isin(train_samples_ids)]
    df_clinical_val = df_clinical_preprocessed[df_clinical_preprocessed['bcr_patient_barcode'].isin(val_samples_ids)]

    df_clinical_train_processed, scalers, imputers, categories = nan_imputation_and_features_normalization(
        df=df_clinical_train,
        is_train=True,
    )

    df_clinical_val_processed = nan_imputation_and_features_normalization(
        df=df_clinical_val,
        is_train=False,
        scalers=scalers,
        imputers=imputers,
        categories=categories,
    )

    df_clinical_processed = pd.DataFrame(pd.concat([df_clinical_train_processed, df_clinical_val_processed]))
    train_val_dataset.update_clinical_df(df_clinical_processed)


    """"" fold specific dataset separation """""

    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = MultiOmicGAT(
        in_features=num_genes_features,
        hidden_dim=64,
        out_channels=num_classes,
        heads=4,
        dropout=0.2,
        num_clinical_features=num_clinical_features,
    ).to(device)
    logging.info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)
        logging.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
        logging.info(f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

        if val_loss < best_val_loss:
            #logging.info(f"IMPROVEMENT! Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logging.info("IMPROVEMENT!\n")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(dir_to_save, f"best_MultiOmicGat_fold{fold+1}.pt"))
            early_stopping_counter = 0

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping, 20 epochs without improvement ---\n")
            break
        else:
            early_stopping_counter += 1
