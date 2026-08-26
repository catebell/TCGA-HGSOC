import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import task_target

'''
Split at patient level: multiple tissues belonging to the same patient must end up in the same split.
'''

path_to_mapping_file = os.path.join('..', '1_dataset', 'gdc_sample_level_mapping.csv')
path_to_clinical_file = os.path.join('..', '2_preprocessing', 'preprocessed_clinical_data.tsv')

TASK = task_target.TASK
TARGET_COL, TIME_COL = "", ""  # task label to stratify on
cols_to_fetch = []

if TASK == 'classification':
    TARGET_COL = task_target.TARGET_COL
    cols_to_fetch = [TARGET_COL]

elif TASK == 'survival':
    TARGET_COL = task_target.SURVIVAL_EVENT_COL
    TIME_COL = task_target.SURVIVAL_TIME_COL
    cols_to_fetch = [TARGET_COL, TIME_COL]


# 70% Train, 15% Val, 15% Test
TRAIN_SIZE = 0.7
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 7


""""" samples selection """""

df_complete = pd.read_csv(path_to_mapping_file)

#'A', 'B', 'C', 'D' = tissue vial / workflow; 'R' = Re-extraction
# if possible, better to use the original, first sampling:
df_complete['Tissue_Group'] = df_complete['Sample_Type'].str[:2]  # tissue code (es. '01')
df_complete['Vial_Letter'] = df_complete['Sample_Type'].str[2:]  # workflow letter

priority_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'R': 5}
df_complete['Priority'] = df_complete['Vial_Letter'].map(priority_dict).fillna(5)
df_complete = df_complete.sort_values(by=['Patient_ID', 'Tissue_Group', 'Priority'])  # first = 'best'

# keep only first row per patient
df_complete = df_complete.drop_duplicates(subset=['Patient_ID', 'Tissue_Group'], keep='first')
df_complete = df_complete.drop(columns=['Tissue_Group', 'Vial_Letter', 'Priority'])


""""" tissues single/pairs separation """""

# only tissues with all molecular data available
df_complete = df_complete.dropna(subset=["RNA_File_UUID", "CNV_Allele_Specific_UUID"], how='any')

# evolutionary test set (pairs primary-recurrence/metastasis)
patients_with_more_tissues = df_complete.groupby('Patient_ID').filter(lambda x: len(x['Sample_Type'].unique()) > 1)['Patient_ID'].unique()
df_test_paired = df_complete[df_complete['Patient_ID'].isin(patients_with_more_tissues)]

df_singles_set = df_complete[~df_complete['Patient_ID'].isin(patients_with_more_tissues)]  # with only one tissue type available

df_singles_set.to_csv(os.path.join('tissues_singles.csv'), index=False)
df_test_paired.to_csv(os.path.join('tissues_pairs.csv'), index=False)

print(f"Single tissues Set: {len(df_singles_set)} single primary samples.")
print(f"Pairs Set: {len(df_test_paired)} paired samples ({len(patients_with_more_tissues)} patients).\n")

# pairs will be used only for testing


""""" single tissues train/test separation """""

df_clinical = pd.read_csv(path_to_clinical_file, sep="\t")

# Merge mapping df with target column from clinical file using Patient_ID / bcr_patient_barcode
df_singles_merged = df_singles_set.merge(
    df_clinical[["bcr_patient_barcode"] + cols_to_fetch],
    left_on="Patient_ID",
    right_on="bcr_patient_barcode",
    how="inner"
)

df_singles_merged = df_singles_merged.drop(columns=["bcr_patient_barcode"])

# drop nan in target col (not useful)
for col in cols_to_fetch:
    df_singles_merged[col] = df_singles_merged[col].replace(-1, np.nan)

df_singles_merged = df_singles_merged.dropna(subset=cols_to_fetch)


if TASK == 'survival':
    df_singles_merged = df_singles_merged[df_singles_merged[TIME_COL] > 0]  # remove non valid times
    df_singles_merged[TARGET_COL] = df_singles_merged[TARGET_COL].astype(int)
    print(f"--- Task: SURVIVAL ---")
    print(f"Event Col: '{TARGET_COL}' | Time Col: '{TIME_COL}'")
    print("Event status distribution:\n", df_singles_merged[TARGET_COL].value_counts(normalize=True))

elif TASK == 'classification':
    # drop cases with less than 4 occurrences (not stratifiable)
    df_singles_merged = df_singles_merged.groupby(TARGET_COL).filter(lambda x: len(x) > 4)
    print(f"--- Task: CLASSIFICATION ---")
    print(f"Target Label: '{TARGET_COL}'")
    print("Classes distribution:\n", df_singles_merged[TARGET_COL].value_counts(normalize=True))

print(f"Initial samples in mapping: {len(df_singles_set)}")
print(f"Valid samples with not-null label: {len(df_singles_merged)}")
print("-" * 40)


# split train / val+test --> 0.15 + 0.15 = 0.30 (30%)
ratio = VAL_SIZE + TEST_SIZE
df_train, df_test = train_test_split(df_singles_merged, test_size=ratio, random_state=RANDOM_SEED, stratify=df_singles_merged[TARGET_COL])

# split val / test
ratio = VAL_SIZE / ratio  # 0.15 / 0.30 = 0.5
df_val, df_test = train_test_split(df_test, test_size=ratio, random_state=RANDOM_SEED, stratify=df_test[TARGET_COL])


df_train_val = pd.DataFrame(pd.concat([df_train,df_val]))
df_train_val.to_csv("singles_train_val.csv", index=False)

df_train.to_csv("singles_train.csv", index=False)
df_val.to_csv("singles_val.csv", index=False)
df_test.to_csv("singles_test.csv", index=False)

print("\nSplit:")
total_valid = len(df_singles_merged)
print(f"Single tissues total: {total_valid}")
print(f"Train + Val: {len(df_train_val)} ({len(df_train_val)/total_valid:.1%})")
print(f"Train: {len(df_train)} ({len(df_train)/total_valid:.1%})")
print(f"Val:   {len(df_val)} ({len(df_val)/total_valid:.1%})")
print(f"Test:  {len(df_test)} ({len(df_test)/total_valid:.1%})")
