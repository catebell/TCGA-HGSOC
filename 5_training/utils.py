import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import task_target

task = task_target.TASK
target_col = task_target.TARGET_COL


""""" clinical postprocessing """""

def nan_imputation_and_features_normalization(df, is_train=True, scalers=None, imputers=None, categories=None):
    """
    Clinical nan imputation, reformatting, normalization and target column codification preventing data leakage.
    If is_train=True: compute parameters on Train data, else applies same parameters on Test/Validation data too.
    """
    df_norm = df.copy()

    if is_train:
        scalers = {}
        imputers = {}
    else:
        if scalers is None or imputers is None:
            raise ValueError("Scalers and Imputers are needed when is_train=False!")


    """"" AGE : imputer (median) for missing data + std scaler """""

    col = 'age_at_initial_pathologic_diagnosis'
    if col in df_norm.columns and col != target_col:
        if is_train:
            imputers['age'] = SimpleImputer(strategy='median')
            scalers['age'] = StandardScaler()
            df_norm[col] = imputers['age'].fit_transform(df_norm[[col]])
            df_norm[col] = scalers['age'].fit_transform(df_norm[[col]])
        else:
            df_norm[col] = imputers['age'].transform(df_norm[[col]])
            df_norm[col] = scalers['age'].transform(df_norm[[col]])


    """"" DAYS features : Log1p transform + std scaler """""

    time_cols = ['overall_survival_days', 'total_drug_therapy_duration_days', 'days_to_new_tumor_event_after_initial_treatment']
    for col in time_cols:
        if col in df_norm.columns and col != target_col:
            if is_train:
                imputers[col] = SimpleImputer(strategy='median')
                scalers[col] = StandardScaler()
                vals = imputers[col].fit_transform(df_norm[[col]])
                log_vals = np.log1p(np.maximum(0, vals))
                df_norm[col] = scalers[col].fit_transform(log_vals)
            else:
                vals = imputers[col].transform(df_norm[[col]])
                log_vals = np.log1p(np.maximum(0, vals))
                df_norm[col] = scalers[col].transform(log_vals)


    """"" THERAPY CYCLES : RobustScaler (outliers resistant) """""

    col = 'number_cycles'
    if col in df_norm.columns and col != target_col:
        if is_train:
            imputers[col] = SimpleImputer(strategy='median')
            scalers[col] = RobustScaler()
            vals = imputers[col].fit_transform(df_norm[[col]])
            df_norm[col] = scalers[col].fit_transform(vals)
        else:
            vals = imputers[col].transform(df_norm[[col]])
            df_norm[col] = scalers[col].transform(vals)


    """"" ORDINAL FEATURES MinMaxScaler() """""

    cols_ordinal = ['clinical_stage', 'neoplasm_histologic_grade', 'tumor_residual_disease']
    for col in cols_ordinal:
        if col in df_norm.columns and col != target_col:
            if is_train:
                imputers[col] = SimpleImputer(strategy='median')
                scalers[col] = MinMaxScaler()
                vals = imputers[col].fit_transform(df_norm[[col]])
                df_norm[col] = scalers[col].fit_transform(vals)
            else:
                vals = imputers[col].transform(df_norm[[col]])
                df_norm[col] = scalers[col].transform(vals)


    """"" BOOLEAN FEATURES 'most frequent' strategy """""

    cols_boolean = ['primary_therapy_outcome_success', 'postoperative_rx_tx', 'therapy_ongoing', 'had_progression_therapy']
    for col in cols_boolean:
        if col in df_norm.columns and col != target_col:
            if is_train:
                imputers[col] = SimpleImputer(strategy='most_frequent')
                df_norm[col] = imputers[col].fit_transform(df_norm[[col]]).ravel()
            else:
                df_norm[col] = imputers[col].transform(df_norm[[col]]).ravel()


    """"" target codification into zero-based continuous interval for pytorch functions (es. [6,7,9] -> [0,1,3]) """""

    if task == 'classification' and target_col in df_norm.columns:
        if is_train:
            categories = sorted(df_norm[target_col].unique())
        else:
            if categories is None:
                raise ValueError("categories are needed when task = 'classification' and is_train=False.")

        df_norm[target_col] = pd.Categorical(df_norm[target_col], categories=categories).codes

    if is_train:
        return df_norm, scalers, imputers, categories
    return df_norm


""""" utility """""

def print_details(dataset):
    print()
    print(f'Train Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')

    print(dataset[0])
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {dataset[0].num_nodes}')
    print(f'Number of edges: {dataset[0].num_edges}')
    print(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}')
    print(f'Has isolated nodes: {dataset[0].has_isolated_nodes()}')
    #print(f'Has self-loops: {dataset[0].has_self_loops()}')  # added internally during GNN forward, checked in gatv2_conv module
    print(f'Is undirected: {dataset[0].is_undirected()}')
    print()