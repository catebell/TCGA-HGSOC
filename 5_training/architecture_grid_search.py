import logging
import os

import numpy as np
import pandas as pd
import torch
import itertools

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

import task_target
from train_eval_utils import  nan_imputation_and_features_normalization
from PatientTissueDataset import PatientTissueDataset
from models.MultiOmicGAT import MultiOmicGAT
from models.MultiOmicGATSurvival import MultiOmicGATSurvival, CoxPHLoss
from train_functions import train_epoch, evaluate, train_epoch_survival, evaluate_survival


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
    'lr': [0.0005, 0.0001], # with then reduce on plateau
    'hidden_channels': [32, 64, 128],
    'batch_size': [16], #[4, 8, 16],
    'dropout': [0.3, 0.4], # 0.2 and 0.5 too low and too high
}

# generates all possible params combinations
keys, values = zip(*params_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

path_to_clinical = os.path.join("..", "2_preprocessing", "preprocessed_clinical_data.tsv")

paths_graphs_splits = {
    #"train": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "train"),
    #"val": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "val"),
    "train_val": os.path.join("..", "4_samples_graph_creation", "processed_graphs", "train_val"),
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

logging.info("Train + Val Dataset init...")
train_val_dataset = PatientTissueDataset(paths_graphs_splits.get("train_val"), df_clinical_preprocessed, target_col=target_column)


""""" set model parameters """""

num_genes_features = 10  # multiomics feature vector
num_clinical_features = len(train_val_dataset.clinical_feature_cols)  # total clinical features without the one to predict
logging.info(f"Clinical Features found : {num_clinical_features} \n {train_val_dataset.clinical_feature_cols}")

y_target = None

if task == 'classification':
    y_labels = torch.tensor([train_val_dataset[i].y.item() for i in range(len(train_val_dataset))])
    y_target = y_labels
    unique_labels = torch.unique(y_labels)
    logging.info(f"Unique labels found in dataset: {unique_labels}")
    num_classes = len(unique_labels)

elif task == 'survival':
    y_event = torch.tensor([train_val_dataset[i].y_event.item() for i in range(len(train_val_dataset))])
    y_target = y_event

# stratified dataset indices split once for all GridSearch
indices = np.arange(len(train_val_dataset))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=y_target.numpy(),
    random_state=42
)

if task == 'classification':
    # classes weights computed using only train dataset
    y_train_labels = y_target[train_idx]
    class_counts = torch.bincount(y_train_labels)
    total_samples = len(y_train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

elif task == 'survival':
    criterion = CoxPHLoss()

""""" dataset train/val split and clinical processing accordingly """""
# retrieving patient ids of current train/val splits from graphs sample ids
train_samples_ids = ["-".join(train_val_dataset[i].sample_id.split("-")[:3]) for i in train_idx]
val_samples_ids = ["-".join(train_val_dataset[i].sample_id.split("-")[:3]) for i in val_idx]

df_clinical_train = df_clinical_preprocessed[
    df_clinical_preprocessed['bcr_patient_barcode'].isin(train_samples_ids)]
df_clinical_val = df_clinical_preprocessed[df_clinical_preprocessed['bcr_patient_barcode'].isin(val_samples_ids)]

df_clinical_train_processed, scalers, imputers, categories = nan_imputation_and_features_normalization(
    df=df_clinical_train,
    target_col=target_column,
    is_train=True,
)

df_clinical_val_processed = nan_imputation_and_features_normalization(
    df=df_clinical_val,
    target_col=target_column,
    is_train=False,
    scalers=scalers,
    imputers=imputers,
    categories=categories,
)

df_clinical_processed = pd.DataFrame(pd.concat([df_clinical_train_processed, df_clinical_val_processed]))
train_val_dataset.update_clinical_df(df_clinical_processed)

train_dataset = Subset(train_val_dataset, train_idx)
val_dataset = Subset(train_val_dataset, val_idx)


""""" functions """""

def train_and_eval_model(params):
    train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size'), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.get('batch_size'), shuffle=False)

    model = None

    if task == 'classification':
        model = MultiOmicGAT(
            in_node_features=num_genes_features,
            hidden_dim=params.get('hidden_channels'),
            out_channels=num_classes,
            heads=4,
            dropout=params.get('dropout'),
            num_clinical_features=num_clinical_features,
        ).to(device)

    elif task == 'survival':
        model = MultiOmicGATSurvival(
            in_node_features=num_genes_features,
            hidden_dim=params.get('hidden_channels'),
            out_channels=1,
            heads=4,
            dropout=params.get('dropout'),
            num_clinical_features=num_clinical_features,
        ).to(device)

    logging.info(str(model) + '\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.get('lr'), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    # possible metrics to evaluate for improvement
    best_val_loss = float('inf')  # minimizing loss
    best_val_c_index = 0.0  # maximizing C-index
    best_val_auc = -1
    best_metrics = {}
    early_stopping_counter = 0

    for epoch in range(1, max_epochs + 1):
        if task == 'classification':
            train_loss, train_acc = train_epoch(device, model, train_loader, optimizer)
            val_loss, val_metrics = evaluate(device, model, val_loader)

            scheduler.step(val_loss)  # update learning rate; alternative: mode max and val_metrics['auc']

            logging.info(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}')
            logging.info(f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

            if val_metrics['auc'] > best_val_auc:  # save best model based on Val Loss
                best_val_auc = val_metrics['auc']
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


best_overall_auc = -1
best_overall_f1 = 0
best_overall_metrics = {}
best_config = {}
best_overall_c_index = 0
best_overall_loss = float('inf')  # not used

for i, params in enumerate(combinations):
    logging.info(f"--- GRIDSEARCH TEST {i + 1}/{len(combinations)} | CONFIG: {params} ---\n")

    # training for current configuration
    current_config_metrics = train_and_eval_model(params)
    #current_auc = current_config_metrics.get('auc', 0)
    #current_f1 = current_config_metrics.get('f1', 0)

    if task == 'classification':
        score = current_config_metrics.get('auc')
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
