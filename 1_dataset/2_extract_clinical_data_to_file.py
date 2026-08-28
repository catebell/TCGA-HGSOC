import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob
from tqdm import tqdm

import task_target

dir_clinical_data = f"GDC-TCGA_{task_target.DATASET}"
dir_to_save = "data_extracted"
os.makedirs(dir_to_save, exist_ok=True)
path_to_save = os.path.join(dir_to_save, 'clinical_data.tsv')


def extract_nodes(element, patient_data_dict):
    """Recursive function to extract leaf-nodes from an XML element, ignoring namespaces in cols names."""
    tag = element.tag.split('}')[-1] # Remove namespace (es. {url}gender -> gender)

    if len(element) == 0 and element.text:
        text = element.text.strip()
        if text:
            # Handle eventual duplicate tags by appending values
            if tag in patient_data_dict:
                patient_data_dict[tag] = f"{patient_data_dict[tag]}, {text}"
            else:
                patient_data_dict[tag] = text

    for son in element:
        extract_nodes(son, patient_data_dict)


# find all the files in the directory (recursively) containing "clinical" with .xml extension
path_to_find = os.path.join(dir_clinical_data, "**", "*clinical*.xml")

file_list = glob.glob(path_to_find, recursive=True)
print("\nFound " + str(len(file_list)) + " clinical xml files. Extracting dataframes...")

patients_data_list = []

for file in tqdm(file_list):  # file = absolute path of each xml found

    # convert each xml as a row in a pandas df
    tree = ET.parse(file)
    root = tree.getroot()
    patient_data = {}

    # Start nodes extraction
    extract_nodes(root, patient_data)
    # Convert dict in a DataFrame (1 row)
    df = pd.DataFrame([patient_data])
    patients_data_list.append(df)

clinical_data_df = pd.concat(patients_data_list)

# 'bcr_patients_uuid' or 'bcr_patient_barcode' are used to identify patients across data types, and should be unique.
# there are a few rows where it is duplicated, but they're also in a wrong format, so we ignore them.
clinical_data_df = clinical_data_df[ ['bcr_patient_barcode'] + [ col for col in clinical_data_df.columns if col != 'bcr_patient_barcode' ] ]

clinical_data_df.to_csv(path_to_save, sep='\t', index=False)
print(f"DONE, file saved in {path_to_save}\n")



with open(os.path.join(dir_to_save, 'clinical_feats_stats.txt'), 'w') as f, open(os.path.join(dir_to_save, 'clinical_feats_ignored.txt'), 'w') as d:
    clinical_data_df = pd.read_csv(path_to_save, sep='\t')
    print("Computing clinical stats...")
    N = len(clinical_data_df)
    i=0
    j=0

    for col in tqdm(clinical_data_df.columns):
        counts = len(clinical_data_df[col].value_counts())
        if counts == 1 or counts == len(clinical_data_df):
            i=i+1
            print("\n\n" + str(i) + ") " + str(col) + ": " + str(round((N - clinical_data_df[col].isna().sum()) * 100 / N, 2)) +
                  "% of patients have this data.", file=d)
            print(clinical_data_df[col].value_counts(), file=d)
        else:
            j=j+1
            print("\n\n" + str(j) + ") " + str(col) + ": " + str(round((N - clinical_data_df[col].isna().sum()) * 100 / N, 2)) +
                  "% of patients have this data. [Different values present: " + str(counts) + "]", file=f)
            if counts < 5:
                print(clinical_data_df[col].value_counts(), file=f)

with open(os.path.join(dir_to_save, 'clinical_feats_stats.txt'), 'w') as f, open(os.path.join(dir_to_save, 'clinical_feats_ignored.txt'), 'w') as d:
    clinical_data_df = pd.read_csv(path_to_save, sep='\t')
    print("Computing clinical stats...")
    N = len(clinical_data_df)
    i=0
    j=0

    for col in tqdm(clinical_data_df.columns):
        counts = len(clinical_data_df[col].value_counts())
        if counts == 1 or counts == len(clinical_data_df):
            i=i+1
            print("\n\n" + str(i) + ") " + str(col) + ": " + str(round((N - clinical_data_df[col].isna().sum()) * 100 / N, 2)) +
                  "% of patients have this data.", file=d)
            print(clinical_data_df[col].value_counts(), file=d)
        else:
            j=j+1
            print("\n\n" + str(j) + ") " + str(col) + ": " + str(round((N - clinical_data_df[col].isna().sum()) * 100 / N, 2)) +
                  "% of patients have this data. [Different values present: " + str(counts) + "]", file=f)
            print(clinical_data_df[col].value_counts(), file=f)
