import logging
import os
import numpy as np
import pandas as pd
import torch
import itertools

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader

import task_target
from PatientTissueDataset import PatientTissueDataset
from MultiOmicGAT import MultiOmicGAT

import warnings
warnings.filterwarnings("ignore")  # to temporarily not see warnings

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
torch.cuda.empty_cache()


""""" set general parameters """""

# params grid for GridSearch
params_grid = {
    'lr': [0.001, 0.0001],
    'hidden_channels': [32, 64, 128],
    'batch_size': [4, 8, 16],
    'dropout': [0.2, 0.3, 0.4, 0.5],
}

# generates all possible params combinations
keys, values = zip(*params_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


max_epochs = 100

target_column = task_target.TARGET_COL
logging.info(f"Starting training using target feature: {target_column}\n")

path_to_clinical = os.path.join("..", "2_preprocessing", "preprocessed_clinical_data.tsv")

paths_graphs_splits = {
    "train": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "train"),
    "val": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "val"),
}


""""" loading clinical data """""

df_clinical_preprocessed = pd.read_csv(path_to_clinical, sep="\t")

# reverse preprocessing formatting
df_clinical_preprocessed = df_clinical_preprocessed.replace(-1, np.nan)
df_clinical_preprocessed = df_clinical_preprocessed.dropna(subset=[target_column])
df_clinical_preprocessed[target_column] = df_clinical_preprocessed[target_column].astype(int)


""""" loading datasets """""

logging.info("Train Dataset init...")
train_dataset = PatientTissueDataset(paths_graphs_splits.get("train"), df_clinical_preprocessed, target_col=target_column)

logging.info("Val Dataset init...")
val_dataset = PatientTissueDataset(paths_graphs_splits.get("val"), df_clinical_preprocessed, target_col=target_column)


""""" set model parameters """""

num_genes_features = 10  # multiomics feature vector
num_clinical_features = len(train_dataset.clinical_feature_cols)  # total clinical features without the one to predict
logging.info(f"Num Clinical Features found : {num_clinical_features}")

y_labels = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
unique_labels = torch.unique(y_labels)
logging.info(f"Unique labels found in dataset: {unique_labels}")
num_classes = len(unique_labels)

class_counts = torch.bincount(y_labels)
total_samples = len(y_labels)
class_weights = total_samples / (len(class_counts) * class_counts.float())
class_weights = class_weights.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


""""" functions """""

def train(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    model.zero_grad()

    for data in loader:
        data = data.to(device)
        out, _ = model(data.x, data.edge_index, data.batch, data.clinical_x)
        loss = criterion(out, data.y)
        loss.backward()

        optimizer.step()  # parameters update based on gradients
        optimizer.zero_grad()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

        total += data.num_graphs

    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    total_loss, total = 0, 0
    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch, data.clinical_x)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)  # class with the highest prob

            all_targets.extend(data.y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += data.num_graphs

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    if len(unique_labels) == 2:
        auc_val = roc_auc_score(all_targets, all_probs[:, 1])
    elif len(set(all_targets)) > 2:  # Multi-Class AUC --> OvR strategy (One-vs-Rest)
        auc_val = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    else:
        auc_val = 0.0

    metrics = {
        'acc': np.mean(all_preds == all_targets),
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'auc': auc_val,
    }

    return total_loss / total, metrics


def train_and_save_model(params):
    train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size'), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.get('batch_size'), shuffle=False)

    model = MultiOmicGAT(
        in_features=num_genes_features,
        hidden_dim=params.get('hidden_channels'),
        out_channels=num_classes,
        heads=4,
        dropout=params.get('dropout'),
        num_clinical_features=num_clinical_features,
    ).to(device)
    logging.info(str(model) + '\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.get('lr'), weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_metrics = {}

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer)
        val_loss, val_metrics = evaluate(model, val_loader)

        #scheduler.step(val_loss)  # update learning rate

        logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}')
        logging.info(f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

        if val_loss < best_val_loss:  # save best model based on Val Loss
            best_val_loss = val_loss
            best_metrics = val_metrics

            logging.info("IMPROVEMENT!\n")

            #torch.save(model.state_dict(), f'best_model_{params}.pth')
            #logging.info("--- Found and saved a better model! ---\n")
            early_stopping_counter = 0

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping, 20 epochs without improvement ---\n")
            break
        else:
            early_stopping_counter += 1

    del model
    torch.cuda.empty_cache()

    return best_metrics


best_overall_auc = 0
best_overall_f1 = 0
best_config = None

for i, params in enumerate(combinations):
    logging.info(f"--- GRIDSEARCH TEST {i + 1}/{len(combinations)} | CONFIG: {params} ---\n")

    # training for current configuration
    current_config_metrics = train_and_save_model(params)

    if current_config_metrics.get['auc'] > best_overall_auc:
        best_overall_acc = current_config_metrics.get['auc']
        best_overall_f1 = current_config_metrics.get['f1']
        best_config = params
        logging.info(f"New best score: AUC = {best_overall_auc:.4f}, F1 = {best_overall_f1}\n")

logging.info(f"Search DONE. Best configuration: {best_config} with AUC = {best_overall_auc}, F1 = {best_overall_f1}\n")
