import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from data_matrix_creation import process_omics_folder
from ppi_build import build_edge_index

""""" Data convertion in PyTorch Geometric Data objects that will be passed to the GAT.
Each graph (Data) will represent a single tissue (its genes = nodes, ppi = edges, RNA/CNV values = node features) """""

# TODO per ora bisogna eliminare le cartelle train/test/val a mano prima di ricreare i grafi


path_to_RNA_dir = os.path.join("..", "2_preprocessing", "preprocessed_RNA")
path_to_CNV_dir = os.path.join("..", "2_preprocessing", "preprocessed_CNV")

top_genes = pd.read_csv("selected_top_genes.tsv", sep='\t', index_col=0)

path_to_output_dir = os.path.join("processed_graphs")


def process_and_save_subset(mapping_df, edge_index, gene_to_num_idx, dir_to_save):
    """ Reads a file defining a split (train/val/test), builds subset matrices, creates Data objects and finally
    saves them in a folder. Creates a list of PyG Data objects, one for each sample/tissue; feature matrices built:
    - matrix_rna_tpm_unstranded: DataFrame [Genes x Samples] about expression
    - matrix_cnv_*: DataFrame [Genes x Samples] about allele-specific data of same genes
    """

    """"" matrices computation """""
    matrix_rna_is_present = process_omics_folder(mapping_df, path_to_RNA_dir, value_column="rna_is_present", top_genes=top_genes.index)
    print(f"1/10 RNA data presence matrix: {matrix_rna_is_present.shape} (Genes x Samples)\n")

    matrix_rna_tpm_unstranded = process_omics_folder(mapping_df, path_to_RNA_dir, value_column="tpm_unstranded_log", top_genes=top_genes.index)
    print(f"2/10 RNA Gene expressions matrix: {matrix_rna_tpm_unstranded.shape} (Genes x Samples)\n")

    matrix_cnv_is_present = process_omics_folder(mapping_df, path_to_CNV_dir, value_column="cnv_is_present", top_genes=top_genes.index)
    print(f"3/10 CNV data presence matrix: {matrix_cnv_is_present.shape} (Genes x Samples)\n")

    matrix_cnv_copy_number = process_omics_folder(mapping_df, path_to_CNV_dir, value_column="Copy_Number", top_genes=top_genes.index)
    print(f"4/10 CNV Copy Number matrix: {matrix_cnv_copy_number.shape} (Genes x Samples)\n")

    matrix_cnv_allelic_imbalance = process_omics_folder(mapping_df, path_to_CNV_dir, value_column="Allelic_Imbalance", top_genes=top_genes.index)
    print(f"5/10 CNV Allelic Imbalance matrix: {matrix_cnv_allelic_imbalance.shape} (Genes x Samples)\n")

    matrix_cnv_maj_copy_number = process_omics_folder(mapping_df, path_to_CNV_dir, value_column="Major_Copy_Number", top_genes=top_genes.index)
    print(f"6/10 CNV Major Copy Number matrix: {matrix_cnv_maj_copy_number.shape} (Genes x Samples)\n")

    matrix_cnv_min_copy_number = process_omics_folder(mapping_df, path_to_CNV_dir, value_column="Minor_Copy_Number", top_genes=top_genes.index)
    print(f"7/10 CNV Minor Copy Number matrix: {matrix_cnv_min_copy_number.shape} (Genes x Samples)\n")

    matrix_cnv_loh = process_omics_folder(mapping_df, path_to_CNV_dir, value_column="LOH", top_genes=top_genes.index)
    print(f"8/10 CNV Loss of Heterozygosity matrix: {matrix_cnv_loh.shape} (Genes x Samples)\n")

    matrix_cnv_hom_del = process_omics_folder(mapping_df, path_to_CNV_dir, value_column="Hom_Del", top_genes=top_genes.index)
    print(f"9/10 CNV Homozygous Deletion matrix: {matrix_cnv_hom_del.shape} (Genes x Samples)\n")

    matrix_cnv_cn_loh = process_omics_folder(mapping_df, path_to_CNV_dir, value_column="CnLOH", top_genes=top_genes.index)
    print(f"10/10 CNV Copy-Neutral LOH matrix: {matrix_cnv_cn_loh.shape} (Genes x Samples)\n")

    # to assure node index = gene numerical index
    genes_ordered = sorted(gene_to_num_idx, key=gene_to_num_idx.get)  # sort to 0,1,2,...

    # matrices build with zeroes in all requested genes. If a gene misses an omic, it will be already zeroed.
    rna_is_present_df = matrix_rna_is_present.reindex(genes_ordered).fillna(0)
    rna_tpm_u_df = matrix_rna_tpm_unstranded.reindex(genes_ordered).fillna(0)

    cnv_is_present_df = matrix_cnv_is_present.reindex(genes_ordered).fillna(0)
    cnv_copy_num_df = matrix_cnv_copy_number.reindex(genes_ordered).fillna(2)
    cnv_all_imb_df = matrix_cnv_allelic_imbalance.reindex(genes_ordered).fillna(0)
    cnv_maj_cn_df = matrix_cnv_maj_copy_number.reindex(genes_ordered).fillna(0)
    cnv_min_cn_df = matrix_cnv_min_copy_number.reindex(genes_ordered).fillna(0)
    cnv_loh_df = matrix_cnv_loh.reindex(genes_ordered).fillna(0)
    cnv_hom_del_df = matrix_cnv_hom_del.reindex(genes_ordered).fillna(0)
    cnv_cn_loh_df = matrix_cnv_cn_loh.reindex(genes_ordered).fillna(0)

    common_samples = set(rna_tpm_u_df.columns).intersection(set(cnv_copy_num_df.columns))
    print(f"\nReceived {len(common_samples)} samples. Creating graphs...\n")

    for sample_col in tqdm(common_samples):
        # molecular features extraction and concatenation for each node (gene) --> x dim = [Number of Genes, Number of Features]
        x = torch.tensor(np.stack([
            rna_is_present_df[sample_col].values,
            rna_tpm_u_df[sample_col].values,
            cnv_is_present_df[sample_col].values,
            cnv_copy_num_df[sample_col].values,
            cnv_all_imb_df[sample_col].values,
            cnv_maj_cn_df[sample_col].values,
            cnv_min_cn_df[sample_col].values,
            cnv_loh_df[sample_col].values,
            cnv_hom_del_df[sample_col].values,
            cnv_cn_loh_df[sample_col].values
        ], axis=1), dtype=torch.float)

        sample_graph = Data(
            x=x,
            edge_index=edge_index,
            sample_id=sample_col  # Sample_ID saved for later pairing
        )

        save_path = os.path.join(dir_to_save, f"{sample_col}.pt")
        torch.save(sample_graph, save_path)  # saving .pt file


print("Extracting edge index...")
e_index, gene_to_idx = build_edge_index()  # the same for every subset


splits = {
    "train_val": os.path.join("..", "3_train_test_split", "singles_train_val.csv"),
    "test": os.path.join("..", "3_train_test_split", "singles_test.csv")
}

for split_name, path_to_mapping_file in splits.items():
    if not os.path.exists(path_to_mapping_file):
        print(f"SKIPPING {split_name}: File {path_to_mapping_file} not found.")
        continue

    print(f"\n--- Processing split {split_name}... ---\n")
    file_mapping_df = pd.read_csv(path_to_mapping_file)

    path_to_save_dir = os.path.join(path_to_output_dir, split_name)
    os.makedirs(path_to_save_dir, exist_ok=True)

    process_and_save_subset(file_mapping_df, e_index, gene_to_idx, path_to_save_dir)
    print(f"Save completed for {split_name} in {path_to_save_dir}\n")