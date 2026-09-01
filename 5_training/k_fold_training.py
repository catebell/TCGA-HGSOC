import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import task_target
from train_eval_utils import print_details, nan_imputation_and_features_normalization
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


""""" set general parameters """""

use_clinical = task_target.USE_CLINICAL_DATA

task = task_target.TASK

if task == 'classification':
    target_column = task_target.TARGET_COL
    logging.info(f"Starting training using target feature: {target_column}\n")
elif task == 'survival':
    target_column = task_target.SURVIVAL_EVENT_COL  # placeholder, won't actually be used here
    logging.info(f"Starting training on survival prediction\n")

path_to_clinical = os.path.join("..", "2_preprocessing", "preprocessed_clinical_data.tsv")

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

if task == 'classification':
    df_clinical_preprocessed = df_clinical_preprocessed.dropna(subset=[target_column])
    df_clinical_preprocessed[target_column] = df_clinical_preprocessed[target_column].astype(int)


logging.info("Train + Val Dataset init...")
train_val_dataset = PatientTissueDataset(paths_graphs_splits.get("train_val"), df_clinical_preprocessed, task=task, target_col=target_column)

logging.info("Test Dataset init...")
test_dataset = PatientTissueDataset(paths_graphs_splits.get("test"), df_clinical_preprocessed, task=task, target_col=target_column)

print_details(train_val_dataset)


""""" set model parameters """""

num_genes_features = train_val_dataset.num_features  # multiomics feature vector
num_clinical_features = len(train_val_dataset.clinical_feature_cols)  # total clinical features without the one to predict
logging.info(f"Clinical Features found : {num_clinical_features} \n {train_val_dataset.clinical_feature_cols}")

epochs = 100


""""" k-fold train loop """""

criterion = None
y_skf = None

if task == 'classification':
    y_labels = torch.tensor([train_val_dataset[i].y.item() for i in range(len(train_val_dataset))])
    y_skf = y_labels
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

elif task == 'survival':
    # stratification based on event (0 = censured, 1 = event)
    y_events = torch.tensor([train_val_dataset[i].y_event.item() for i in range(len(train_val_dataset))])
    y_skf = y_events

    criterion = CoxPHLoss()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_dataset, y_skf)):
    logging.info(f"\n--- FOLD {fold + 1} ---\n")

    model_name = f"best_{task}_model_fold{fold + 1}.pt"

    """"" clinical fold specific processing """""

    # retrieving patient ids of current train/val splits from graphs sample ids
    train_samples_ids = ["-".join(train_val_dataset[i].sample_id.split("-")[:3]) for i in train_idx]
    val_samples_ids = ["-".join(train_val_dataset[i].sample_id.split("-")[:3]) for i in val_idx]
    test_samples_ids = ["-".join(test_dataset[i].sample_id.split("-")[:3]) for i,_ in enumerate(test_dataset)]

    df_clinical_train = df_clinical_preprocessed[df_clinical_preprocessed['bcr_patient_barcode'].isin(train_samples_ids)]
    df_clinical_val = df_clinical_preprocessed[df_clinical_preprocessed['bcr_patient_barcode'].isin(val_samples_ids)]
    df_clinical_test = df_clinical_preprocessed[df_clinical_preprocessed['bcr_patient_barcode'].isin(test_samples_ids)]

    df_clinical_train_processed, scalers, imputers, categories = nan_imputation_and_features_normalization(
        df=df_clinical_train,
        target_col=target_column,
        is_train=True,
    )

    # saving preprocessors for evaluation/model usage
    preprocessors = {
        'scalers': scalers,
        'imputers': imputers,
        'categories': categories,
        'clinical_feature_cols': train_val_dataset.clinical_feature_cols
    }
    prep_save_path = os.path.join(dir_to_save, f"clinical_preprocessors_{task}_fold{fold + 1}.joblib")
    joblib.dump(preprocessors, prep_save_path)

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
    num_clinical_features = len(train_val_dataset.clinical_feature_cols)

    df_clinical_test_processed = nan_imputation_and_features_normalization(
        df=df_clinical_test,
        target_col=target_column,
        is_train=False,
        scalers=scalers,
        imputers=imputers,
        categories=categories,
    )
    test_dataset.update_clinical_df(df_clinical_test_processed )


    """"" fold specific dataset separation and DataLoader """""

    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    """"" task specific model upload """""

    if task == 'classification':
        model = MultiOmicGAT(
            in_node_features=num_genes_features,
            hidden_dim=64,
            out_channels=num_classes,
            heads=4,
            dropout=0.3,
            num_clinical_features=num_clinical_features if use_clinical else 0,
        ).to(device)

    elif task == 'survival':
        model = MultiOmicGATSurvival(
            in_node_features=num_genes_features,
            hidden_dim=64,
            out_channels=1,
            heads=4,
            dropout_rate=0.3,
            num_clinical_features=num_clinical_features if use_clinical else 0,
        ).to(device)

    logging.info(str(model) + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    if task == 'classification':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    elif task == 'survival':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)


    """"" fold train loop """""

    # possible metrics to evaluate for improvement
    best_val_loss = float('inf')  # minimizing loss
    best_val_c_index = 0.0  # maximizing C-index
    early_stopping_counter = 0

    for epoch in tqdm(range(1, epochs + 1)):
        if task == 'classification':
            train_loss, train_acc = train_epoch(device, model, train_loader, optimizer, criterion)
            val_loss, val_metrics = evaluate(device, model, val_loader, criterion)

            scheduler.step(val_loss)  # update learning rate

            logging.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")
            logging.info(f"Val metrics: Acc = {val_metrics['acc']:.4f} | F1 = {val_metrics['f1']:.4f}, AUC = {val_metrics['auc']:.4f}\n")

            if val_loss < best_val_loss:
                logging.info("Loss IMPROVEMENT!\n")
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(dir_to_save, model_name))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        elif task == 'survival':
            train_loss = train_epoch_survival(device, model, train_loader, optimizer, criterion)
            val_loss, val_metrics = evaluate_survival(device, model, val_loader, criterion)
            val_c_index = val_metrics['c_index']

            scheduler.step(val_c_index)  # update learning rate

            logging.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val C-Index: {val_c_index:.4f}"
            )

            if val_c_index > best_val_c_index:
                logging.info(
                    f"--> C-Index IMPROVEMENT!: {best_val_c_index:.4f} -> {val_c_index:.4f}\n")
                best_val_c_index = val_c_index
                torch.save(model.state_dict(), os.path.join(dir_to_save, model_name))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        if early_stopping_counter > 20:
            logging.info("--- Stopping training due to early stopping, 20 epochs without improvement ---\n")
            break

    best_model_path = os.path.join(dir_to_save, model_name)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    if task == 'classification':
        test_loss, test_metrics = evaluate(device, model, test_loader, criterion)
        logging.info(
            f"Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_metrics['acc']:.4f} | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Test AUC: {test_metrics['auc']:.4f}\n"
        )
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'acc': test_metrics['acc'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc']
        })

    elif task == 'survival':
        test_loss, test_metrics = evaluate_survival(device, model, test_loader, criterion)
        logging.info(
            f"Fold {fold + 1} Test Loss: {test_loss:.4f} | "
            f"Test C-Index: {test_metrics['c_index']:.4f}\n"
        )
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss,
            'c_index': test_metrics['c_index']
        })


""""" folds aggregate evaluation """""

df_fold_results = pd.DataFrame(fold_results)

if task == 'classification':
    avg_acc = df_fold_results['acc'].mean()
    std_acc = df_fold_results['acc'].std()
    avg_f1 = df_fold_results['f1'].mean()
    std_f1 = df_fold_results['f1'].std()
    avg_auc = df_fold_results['auc'].mean()
    std_auc = df_fold_results['auc'].std()

    logging.info(f"Accuracy:  {avg_acc:.4f} ± {std_acc:.4f}")
    logging.info(f"F1 Score:  {avg_f1:.4f} ± {std_f1:.4f}")
    logging.info(f"ROC AUC:   {avg_auc:.4f} ± {std_auc:.4f}")

elif task == 'survival':
    avg_c_index = df_fold_results['c_index'].mean()
    std_c_index = df_fold_results['c_index'].std()

    logging.info(f"C-Index:   {avg_c_index:.4f} ± {std_c_index:.4f}")

df_fold_results.to_csv(os.path.join(dir_to_save, f"report_kfold_test_results_{task}.csv"), index=False)  # final report saved

logging.info("DONE\n")