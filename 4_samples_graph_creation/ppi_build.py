import os
import numpy as np
import pandas as pd
import torch

path_ppi_file = os.path.join("..", "ppi_string_files","9606.protein.links.v12.0.txt")
#path_protein_gene_file = os.path.join("..", "ppi_string_files","9606.protein.aliases.gene.tsv")
path_protein_gene_file = os.path.join("..", "ppi_string_files","9606.protein.aliases.v12.0.txt")
path_top_genes_file = os.path.join("selected_top_genes.tsv")

def build_edge_index():
    """ Select existing connections only between top genes found """

    print("Reading STRING files...")

    top_genes = pd.read_csv(path_top_genes_file, sep='\t', usecols=['gene_id', 'gene_name'])

    df_protein_gene = pd.read_csv(path_protein_gene_file, sep='\t')
    df_protein_gene = df_protein_gene.rename(columns={'#string_protein_id': 'protein_id', 'alias': 'gene_name'})
    df_protein_gene = df_protein_gene.drop(columns='source')

    df_ppi = pd.read_csv(path_ppi_file, sep=' ')
    df_ppi['combined_score'] = df_ppi['combined_score'] / 1000  # remap in [0,1] interval
    df_ppi.drop(df_ppi[df_ppi['combined_score'] < 0.9].index, inplace=True)

    print("Adding matches from protein.aliases.gene file to associate all protein isoforms coded per gene...")

    df_filtered_proteins = pd.DataFrame(top_genes)

    # add all protein_ids associated to a gene_id/name as multiple rows
    df_filtered_proteins = pd.merge(df_filtered_proteins, df_protein_gene, how='left', on=['gene_name'])
    df_filtered_proteins = df_filtered_proteins.dropna()
    df_filtered_proteins = df_filtered_proteins.drop_duplicates(subset=['gene_id','protein_id'])
    df_filtered_proteins = df_filtered_proteins.reset_index(drop=True)

    print("Retrieving protein-protein interactions from file protein.links...")

    # only interactions between genes both present in df_filtered_proteins, both ways (p1-->p2 and p2-->p1)
    genes_network_df = df_ppi[(df_ppi['protein1'].isin(df_filtered_proteins['protein_id'])) &
                                  (df_ppi['protein2'].isin(df_filtered_proteins['protein_id']))].copy()
    genes_network_df.reset_index(inplace=True, drop=True)

    # map {protein_id --> gene_id}
    prot_to_gene_map = dict(zip(df_filtered_proteins['protein_id'], df_filtered_proteins['gene_id']))

    genes_network_df['gene1'] = genes_network_df['protein1'].map(prot_to_gene_map)
    genes_network_df['gene2'] = genes_network_df['protein2'].map(prot_to_gene_map)

    # removed self-loops by isoforms from same gene (GATConv handles them with add_self_loops=True)
    genes_network_df = genes_network_df[genes_network_df['gene1'] != genes_network_df['gene2']]
    # removed duplicated edges
    genes_network_df = genes_network_df.drop_duplicates(subset=['gene1', 'gene2'])

    print("Proteins coded by subset of genes: " + str(len(df_filtered_proteins)))
    print("Genes directed interactions found: " + str(len(genes_network_df)))

    # dict to map gene_id to numerical index [0, 1, ..., N-1]
    gene_to_num_idx = {gene: idx for idx, gene in enumerate(top_genes['gene_id'])}

    # gene ids mapped to numerical ids for GNN tensors
    edge_src = genes_network_df['gene1'].map(gene_to_num_idx).values
    edge_dst = genes_network_df['gene2'].map(gene_to_num_idx).values

    # PyTorch Geometric wants edges in format [2, Num_Archi]
    e_index = torch.tensor(np.array([edge_src, edge_dst]), dtype=torch.long)

    print("DONE!\n")

    return e_index, gene_to_num_idx


if __name__ == "__main__":
    edge_index, gene_to_idx = build_edge_index()
    # print to verify that links go both ways (755 can be changed to any gene num_idx):
    # pd.DataFrame(edge_index.detach().numpy()).T[(pd.DataFrame(edge_index.detach().numpy()).T[0] == 755) | (pd.DataFrame(edge_index.detach().numpy()).T[1] == 755)]
    print()
