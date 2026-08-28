import glob
import shutil

import pandas as pd
import os
from tqdm import tqdm

import task_target

dir_CNV_data = f"GDC-TCGA_{task_target.DATASET}"
dir_to_save_allelic = os.path.join("data_extracted", "CNV_extracted", "Allele_Specific")
if os.path.exists(dir_to_save_allelic):
    shutil.rmtree(dir_to_save_allelic)  # remove dir if already exists
os.makedirs(dir_to_save_allelic, exist_ok=True)

# find all the files in the directory (recursively) with .txt extension
path_to_find_allelic = os.path.join(dir_CNV_data, "**", "*allel*.txt")

file_list = glob.glob(path_to_find_allelic, recursive=True)
print("\nFound " + str(len(file_list)) + " allele-specific txt files in the CNV data folder. Processing...")

for file in tqdm(file_list):  # file = absolute path of each txt found
    df = pd.read_csv(file, sep="\t")
    df = df.dropna().reset_index(drop=True)

    # es. dataset_folder\8d7a1721-9fb3-46cc-a95a-5e9a6f207c7e\TCGA-CESC.08c9601e-983c-4fc3-960f-df202b944ae9.ascat2.allelic_specific.seg.txt"
    file = os.path.normpath(file) if file else ""  # uniforms separator
    file_uuid = file.split(os.sep)[2].split(".")[1] if file else ""

    path_to_save = os.path.join(dir_to_save_allelic, file_uuid + ".tsv")
    df.to_csv(path_to_save, sep="\t", index=False)

print("DONE\n")