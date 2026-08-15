import os
import pandas as pd
from tqdm import tqdm


def process_omics_folder(sample_files_mapping_df, sample_data_folder_path, value_column, top_genes = None):
    """ Process a folder containing a single omic data into a matrix using values = value_column.
    Returns matrix Genes x Samples and the dictionary to map genes to unique numerical index.
     If top_genes list is passed, the matrix will include only those genes. """

    # useful dict: {filename: sample_id}
    file_to_case = {}
    if sample_data_folder_path.__contains__('RNA'):
        file_to_case = dict(zip(sample_files_mapping_df['RNA_File_UUID'] + ".tsv", sample_files_mapping_df['Sample_ID']))
    elif sample_data_folder_path.__contains__('CNV'):
        file_to_case = dict(zip("genes." + sample_files_mapping_df['CNV_Allele_Specific_UUID'] + ".tsv", sample_files_mapping_df['Sample_ID']))

    print(f"Processing {sample_data_folder_path}...")

    all_samples = []
    map_all_gene_ids_names = {}  # to save also the name version found

    folder_files = set(os.listdir(sample_data_folder_path))

    for filename in tqdm(file_to_case.keys()):
        if filename not in folder_files:
            print(f"SKIPPING {filename} not found.")
            continue

        sample_id = file_to_case[filename]

        cols = ['gene_id', 'gene_name', value_column]
        df = pd.read_csv(os.path.join(sample_data_folder_path, filename), sep="\t", usecols=cols)

        if top_genes is None: # we don't know them yet
            # update dict {gene_id: gene_name} for names not mapped yet
            new_genes_df = df[~df['gene_id'].isin(map_all_gene_ids_names.keys())]
            if not new_genes_df.empty:
                new_map = new_genes_df[['gene_id', 'gene_name']].drop_duplicates().dropna()
                new_dict = dict(zip(new_map['gene_id'], new_map['gene_name']))
                map_all_gene_ids_names.update(new_dict)
        else:
            df = df[df['gene_id'].isin(top_genes)]  # filter and keep only top_genes

        # build matrix
        df_clean = df[['gene_id', value_column]].copy()
        df_clean = df_clean.rename(columns={value_column: sample_id})
        df_clean = df_clean.set_index('gene_id')

        all_samples.append(df_clean)

    # all patients in a matrix
    matrix = pd.concat(all_samples, axis=1)

    if top_genes is None:
        return matrix, map_all_gene_ids_names
    else:
        return matrix
