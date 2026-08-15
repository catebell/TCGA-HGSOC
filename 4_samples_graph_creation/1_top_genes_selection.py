import os
import pandas as pd

from data_matrix_creation import process_omics_folder

path_to_RNA_dir = os.path.join("..", "2_preprocessing", "preprocessed_RNA")
path_to_CNV_dir = os.path.join("..", "2_preprocessing", "preprocessed_CNV")

# top genes computed only on train (+ val) data subset
mapping_file = os.path.join("..", "3_train_test_split", "singles_train_val.csv")
mapping_df = pd.read_csv(mapping_file)


matrix_rna, map_gene_ids_names = process_omics_folder(mapping_df, path_to_RNA_dir, value_column="tpm_unstranded_log")
print(f"\nRNA Gene expressions matrix: {matrix_rna.shape} (Genes x Samples)\n")
# transposed --> (Samples x Genes) so we can select columns, easier computation
matrix_rna = matrix_rna.T


""""" RNA expression filter """""

# Selection of N genes with higher variance (Highly Variable Genes)
var_rna = matrix_rna.var()  # var computed by rows
top_genes_var = var_rna.nlargest(5000)
matrix_rna = matrix_rna[top_genes_var.index]

#matrix_rna = matrix_rna.T # save matrix as (Genes x Samples)
#matrix_rna.to_csv("rna_patients_top_genes_matrix.tsv", sep='\t')

top_genes_df = pd.DataFrame()
top_genes_df['gene_id'] = top_genes_var.index
top_genes_df['variance'] = top_genes_var.values
top_genes_df['gene_name'] = top_genes_var.index.map(map_gene_ids_names)

top_genes_df.to_csv("selected_top_genes.tsv", sep='\t', index=False)


""""" CNV intersection filter """""

# no other filters on cnv: if a gene has cnv features --> features will be added to the node, else nvm

print("DONE\n")