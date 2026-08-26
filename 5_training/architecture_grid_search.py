import logging
import os

import numpy as np
import pandas as pd
import torch
import itertools

from torch_geometric.loader import DataLoader

import task_target
from PatientTissueDataset import PatientTissueDataset
from MultiOmicGAT import MultiOmicGAT
from MultiOmicGAT_Survival import MultiOmicGAT_Survival, CoxPHLoss
from train_functions import train_epoch, evaluate, train_epoch_survival, evaluate_survival


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

task = task_target.TASK
target_column = ""

if task == 'classification':
    target_column = task_target.TARGET_COL
    logging.info(f"Starting training using target feature: {target_column}\n")
elif task == 'survival':
    target_column = task_target.SURVIVAL_EVENT_COL
    time_column = task_target.SURVIVAL_TIME_COL


""""" set general parameters """""

# params grid for GridSearch
params_grid = {
    'lr': [0.001, 0.0001],
    'hidden_channels': [32, 64, 128],
    'batch_size': [16],  # [4, 8, 16],
    'dropout': [0.3, 0.4],
}

# generates all possible params combinations
keys, values = zip(*params_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

path_to_clinical = os.path.join("..", "2_preprocessing", "preprocessed_clinical_data.tsv")

paths_graphs_splits = {
    "train": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "train"),
    "val": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "val"),
}

max_epochs = 100


""""" loading clinical data """""

df_clinical_preprocessed = pd.read_csv(path_to_clinical, sep="\t")

# reverse preprocessing formatting
df_clinical_preprocessed = df_clinical_preprocessed.replace(-1, np.nan)

if task == 'classification':
    target_column = task_target.TARGET_COL
    df_clinical_preprocessed = df_clinical_preprocessed.dropna(subset=[target_column])
    df_clinical_preprocessed[target_column] = df_clinical_preprocessed[target_column].astype(int)
    logging.info(f"Task: Classification | Target col: {target_column}")
elif task == 'survival':
    target_column = task_target.SURVIVAL_EVENT_COL
    time_column = task_target.SURVIVAL_TIME_COL
    df_clinical_preprocessed = df_clinical_preprocessed.dropna(subset=[target_column, time_column])
    df_clinical_preprocessed[target_column] = df_clinical_preprocessed[target_column].astype(int)
    logging.info(f"Task: Survival | Event col: {target_column} | Time col: {time_column}")


""""" loading datasets """""

logging.info("Train Dataset init...")
train_dataset = PatientTissueDataset(paths_graphs_splits.get("train"), df_clinical_preprocessed, target_col=target_column)

logging.info("Val Dataset init...")
val_dataset = PatientTissueDataset(paths_graphs_splits.get("val"), df_clinical_preprocessed, target_col=target_column)


""""" set model parameters """""

num_genes_features = 10  # multiomics feature vector
num_clinical_features = len(train_dataset.clinical_feature_cols)  # total clinical features without the one to predict
logging.info(f"Num Clinical Features found : {num_clinical_features}")

if task == 'classification':
    y_labels = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
    unique_labels = torch.unique(y_labels)
    logging.info(f"Unique labels found in dataset: {unique_labels}")
    num_classes = len(unique_labels)

    class_counts = torch.bincount(y_labels)
    total_samples = len(y_labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
elif task == 'survival':
    criterion = CoxPHLoss()


def train_and_eval_model(params):
    train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size'), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.get('batch_size'), shuffle=False)

    model = None

    if task == 'classification':
        model = MultiOmicGAT(
            in_features=num_genes_features,
            hidden_dim=params.get('hidden_channels'),
            out_channels=num_classes,
            heads=4,
            dropout=params.get('dropout'),
            num_clinical_features=num_clinical_features,
        ).to(device)

    elif task == 'survival':
        model = MultiOmicGAT_Survival(
            in_features=num_genes_features,
            hidden_dim=params.get('hidden_channels'),
            out_channels=1,
            heads=4,
            dropout=params.get('dropout'),
            num_clinical_features=num_clinical_features,
        ).to(device)

    logging.info(str(model) + '\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.get('lr'), weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # possible metrics to evaluate for improvement
    best_val_loss = float('inf')  # minimizing loss
    best_val_c_index = 0.0  # maximizing C-index
    best_metrics = {}
    early_stopping_counter = 0

    for epoch in range(1, max_epochs + 1):
        if task == 'classification':
            train_loss, train_acc = train_epoch(device, model, train_loader, optimizer)
            val_loss, val_metrics = evaluate(device, model, val_loader)

            #scheduler.step(val_loss)  # update learning rate

            logging.info(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}')
            logging.info(f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

            if val_loss < best_val_loss:  # save best model based on Val Loss
                best_val_loss = val_loss
                best_metrics = val_metrics

                logging.info("IMPROVEMENT!\n")

                #torch.save(model.state_dict(), f'best_model_{params}.pth')
                #logging.info("--- Found and saved a better model! ---\n")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        elif task == 'survival':
            train_loss = train_epoch_survival(device, model, train_loader, optimizer, criterion)
            val_loss, val_metrics = evaluate_survival(device, model, val_loader, criterion)
            val_c_index = val_metrics['c_index']

            logging.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val C-Index: {val_c_index:.4f}"
            )

            if val_c_index > best_val_c_index:
                logging.info(f"--> C-Index IMPROVEMENT!: {best_val_c_index:.4f} -> {val_c_index:.4f}\n")
                best_val_c_index = val_c_index
                best_metrics = val_metrics
                # torch.save(model.state_dict(), f'best_model_{params}.pth')
                # logging.info("--- Found and saved a better model! ---\n")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping, 20 epochs without improvement ---\n")
            break

    del model
    torch.cuda.empty_cache()

    return best_metrics


best_overall_auc = 0
best_overall_f1 = 0
best_overall_metrics = {}
best_config = {}
best_overall_c_index = 0

for i, params in enumerate(combinations):
    logging.info(f"--- GRIDSEARCH TEST {i + 1}/{len(combinations)} | CONFIG: {params} ---\n")

    # training for current configuration
    current_config_metrics = train_and_eval_model(params)

    if task == 'classification':
        score = current_config_metrics.get['auc']
        if  score > best_overall_auc:
            best_overall_auc = score
            #best_overall_f1 = current_config_metrics.get['f1']
            best_metrics = current_config_metrics
            best_config = params
            logging.info(f"New best score: AUC = {best_overall_auc:.4f}, F1 = {best_overall_f1}\n")
    elif task == 'survival':
        score = current_config_metrics.get('c_index', 0.0)
        if score > best_overall_c_index:
            logging.info(f"New best score: C-Index = {score:.4f}\n")
            best_overall_c_index = score
            best_metrics = current_config_metrics
            best_config = params

logging.info(f"Search DONE. Best configuration: {best_config} with metrics:\n {best_overall_metrics}\n")
#logging.info(f"Search DONE. Best configuration: {best_config} with AUC = {best_overall_auc}, F1 = {best_overall_f1}\n")
