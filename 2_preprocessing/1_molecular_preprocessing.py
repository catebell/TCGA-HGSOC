import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

RNA_data_dir = os.path.join("..", "1_dataset", "data_extracted","RNA_extracted")
dir_to_save_RNA = os.path.join("preprocessed_RNA")

CNV_data_dir = os.path.join("..", "1_dataset", "data_extracted","CNV_extracted", "Gene_Level_Extracted")
dir_to_save_CNV = os.path.join("preprocessed_CNV")


def clean_gene_identifiers(df_data, gene_col='gene_id'):
    """ Remove Ensembl ID / Gene Name version number (es. ENSG00000141510.12 -> ENSG00000141510) and handle eventual duplicates. """

    df_data[gene_col] = df_data[gene_col].astype(str).str.split('.').str[0]

    return df_data


""""" RNA """""

def rna_preprocessing():
    if os.path.exists(dir_to_save_RNA):
        shutil.rmtree(dir_to_save_RNA)  # remove dir if already exists
    os.makedirs(dir_to_save_RNA, exist_ok=True)

    for file in tqdm(os.listdir(RNA_data_dir)):
        df_rna = pd.read_csv(os.path.join(RNA_data_dir, file), sep="\t")

        # keep only protein_coding
        df_rna = df_rna[df_rna['gene_type'] == 'protein_coding']

        df_rna = clean_gene_identifiers(df_rna, gene_col='gene_id')
        df_rna = clean_gene_identifiers(df_rna, gene_col='gene_name')

        # if there are ambiguities (gene_ids duplicated), keep higher expression
        df_rna_clean = df_rna.groupby('gene_id', as_index=False).agg({
            'gene_name': 'first',
            'unstranded': 'max',
            'tpm_unstranded': 'max',
        })  # other features automatically discarded (not useful for the GNN)

        df_rna_clean['tpm_unstranded_log'] = np.log2(df_rna_clean['tpm_unstranded'] + 1)  # normalization
        #df_rna_clean['tpm_unstranded_is_present'] = float(~df_rna_clean['tpm_unstranded_log'].isna())

        # boolean masks added; nan values removed in extraction, genes later missing will be added with 0
        df_rna_clean['rna_is_present'] = (~df_rna_clean['tpm_unstranded'].isna()).astype(int)
        reordered = df_rna_clean[['rna_is_present'] + [col for col in df_rna_clean.columns if col != 'rna_is_present']]

        reordered.to_csv(os.path.join(dir_to_save_RNA, file), sep="\t", index=False)


""""" CNV """""

def cnv_preprocessing():
    if os.path.exists(dir_to_save_CNV):
        shutil.rmtree(dir_to_save_CNV)  # remove dir if already exists
    os.makedirs(dir_to_save_CNV, exist_ok=True)

    for file in tqdm(os.listdir(CNV_data_dir)):
        df_cnv = pd.read_csv(os.path.join(CNV_data_dir, file), sep="\t")

        df_cnv = clean_gene_identifiers(df_cnv, gene_col='gene_id')
        df_cnv = clean_gene_identifiers(df_cnv, gene_col='gene_name')

        # alternative for Copy_Number, Major, Minor, Allelic_Imbalance: weighted mean by segment length (End - Start)
        cnv_agg_rules = {
            'Chromosome': 'first',  # fixed for every gene
            'gene_name': 'first',

            'Copy_Number': 'median',  # or 'mean'
            'Major_Copy_Number': 'median',  # or 'mean'
            'Minor_Copy_Number': 'median',  # or 'mean'
            'Allelic_Imbalance': 'mean',

            # binary flags --> even if only a single region has '1' it's relevant
            'LOH': 'max',
            'Hom_Del': 'max',
            'CnLOH': 'max'
        }  # other features automatically discarded (not useful for the GNN)

        df_cnv_clean = df_cnv.groupby('gene_id', as_index=False).agg(cnv_agg_rules)

        # masks

        df_cnv_clean['cnv_is_present'] = (~df_cnv_clean['Copy_Number'].isna()).astype(int)
        reordered = df_cnv_clean[['cnv_is_present'] + [col for col in df_cnv_clean.columns if col != 'cnv_is_present']]

        reordered.to_csv(os.path.join(dir_to_save_CNV, file), sep="\t", index=False)


if __name__=="__main__":
    rna_preprocessing()
    cnv_preprocessing()


print("DONE\n")