import glob
import pandas as pd
import os
from tqdm import tqdm

dir_RNA_data = "GDC-TCGA_Ovary"
dir_to_save = os.path.join("data_extracted", "RNA_extracted")
os.makedirs(dir_to_save, exist_ok=True)

# find all the files in the directory (recursively) with .tsv extension
path_to_find = os.path.join(dir_RNA_data, "**", "*rna*.tsv")

file_list = glob.glob(path_to_find, recursive=True)
print("\nFound " + str(len(file_list)) + " gene expression tsv files. Processing...")

for file in tqdm(file_list):  # file = absolute path of each tsv found
    df = pd.read_csv(file, sep="\t", skiprows=1)
    df.drop(df[df['gene_id'].str.startswith("ENSG") == False].index, inplace=True)  # drop metadata
    df = df.dropna().reset_index(drop=True)

    # es. datset_folder\0058f6ab-2114-4ead-af5e-ba002b5f9cc2\8fd3c3ec-c5c5-4ded-bcbb-387ddc1ba5f7.rna_seq.augmented_star_gene_counts.tsv
    file = os.path.normpath(file) if file else ""  # uniforms separator
    file_uuid = file.split(os.sep)[2].split(".")[0] if file else ""

    path_to_save = os.path.join(dir_to_save, file_uuid + ".tsv")
    df.to_csv(path_to_save, sep="\t", index=False)

print("DONE\n")