import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import task_target

print("Clinical splitting and related processing...")

task = task_target.TASK
target_col = task_target.TARGET_COL

path_to_clinical_file = os.path.join("..", "2_preprocessing", "preprocessed_clinical_data.tsv")

df_train_set = pd.read_csv("singles_train.csv")
df_val_set = pd.read_csv("singles_val.csv")
df_test_set = pd.read_csv("singles_test.csv")

df_clinical = pd.read_csv(path_to_clinical_file, sep="\t")


def nan_imputation_and_features_normalization(df, is_train=True, scalers=None, imputers=None, categories=None):
    """ Clinical nan imputation, reformatting, normalization and target column codification based on train/val/test splitting, preventing data leakage when using statistical methods.
    If is_train=True: compute parameters on Train data, else applies same parameters on Test/Validation data too. """

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

    cols_ordinal = ['clinical_stage', 'neoplasm_histologic_grade', 'tumor_residual_disease',]
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

    if task == 'classification':
        if is_train:
            categories = sorted(df_norm[target_col].unique())
        else:
            if categories is None:
                raise ValueError("categories are needed when task = 'classification' and is_train=False.")

        df_norm[target_col] = pd.Categorical(df_norm[target_col], categories=categories).codes

    if is_train:
        return df_norm, scalers, imputers, categories
    return df_norm


""""" reverse preprocessing formatting """""
df_clinical = df_clinical.replace(-1, np.nan)
df_clinical = df_clinical.dropna(subset=[target_col])
df_clinical[target_col] = df_clinical[target_col].astype(int)

# convert all cols to floats except for the target label
int_cols = df_clinical.select_dtypes(include=['int']).columns
int_cols = [c for c in int_cols if c != target_col]
df_clinical[int_cols] = df_clinical[int_cols].astype(float)

# Merge split df with clinical data using Patient_ID / bcr_patient_barcode
df_train_clinical = df_clinical.merge(
    df_train_set[["Patient_ID"]],
    left_on="bcr_patient_barcode",
    right_on="Patient_ID",
    how="inner"
).drop(columns=["Patient_ID"])

df_val_clinical = df_clinical.merge(
    df_val_set[["Patient_ID"]],
    left_on="bcr_patient_barcode",
    right_on="Patient_ID",
    how="inner"
).drop(columns=["Patient_ID"])

df_test_clinical = df_clinical.merge(
    df_test_set[["Patient_ID"]],
    left_on="bcr_patient_barcode",
    right_on="Patient_ID",
    how="inner"
).drop(columns=["Patient_ID"])

df_processed_train, scalers, imputers, categories = nan_imputation_and_features_normalization(df_train_clinical, is_train=True)
print("NaN left in Train:", df_processed_train.isna().sum().sum())

df_processed_val = nan_imputation_and_features_normalization(df_val_clinical, is_train=False, scalers=scalers, imputers=imputers, categories=categories)
print("NaN left in Val:", df_processed_val.isna().sum().sum())
df_processed = pd.concat([df_processed_train, df_processed_val], ignore_index=True)

df_processed_test = nan_imputation_and_features_normalization(df_test_clinical, is_train=False, scalers=scalers, imputers=imputers, categories=categories)
print("NaN left in Test:", df_processed_test.isna().sum().sum())

df_processed_test.to_csv("processed_clinical.tsv", sep="\t", index=False)  # example

#df_processed = pd.concat([df_processed, df_processed_test], ignore_index=True)
#pd.DataFrame(df_processed).to_csv("processed_clinical.tsv", sep='\t', index=False)


# TODO extra con coppie di tessuti
# df_test_norm = nan_imputation_and_features_normalization(df_test, is_train=False, scalers=scalers, imputers=imputers)
