import os.path
from tqdm import tqdm
import pandas as pd
import pyranges as pr

allele_specific_cnv_dir = os.path.join("data_extracted", "CNV_extracted", "Allele_Specific")
dir_to_save = os.path.join('data_extracted', 'CNV_extracted', 'Gene_Level_Extracted')
os.makedirs(dir_to_save, exist_ok=True)

gtf_dir_path = os.path.join('..', 'gencode.annotation.gtf')
gtf_df_filepath = os.path.join(gtf_dir_path, 'gencode.v36.annotation.tsv')

if not os.path.exists(gtf_df_filepath):  # only first time that it's opened, so that next time faster
    print("GTF file upload...\n")
    gtf = pr.read_gtf(os.path.join(gtf_dir_path, "gencode.v36.annotation.gtf"))
    genes_pr = gtf[gtf.Feature == "gene"]  # exclude exons, transcripts, UTR, ...
    genes_df = genes_pr.as_df()[['Chromosome', 'Start', 'End', 'gene_id', 'gene_name']]
    genes_df.to_csv(gtf_df_filepath, sep='\t')

genes_df = pd.read_csv(gtf_df_filepath, sep='\t', index_col=0)

#genes_df['gene_id_clean'] = genes_df['gene_id'].apply(lambda x: x.split('.')[0])

print(f"Found {len(genes_df)} genes from GTF version.\n")

genes_pr = pr.PyRanges(genes_df)  # PyRanges re-initialization with only useful cols

'''
# Example data
cnv_data = {
    'Chromosome': ['chr1', 'chr1', 'chr1', 'chr1'],
    'Start': [61735, 9302282, 10949486, 22151000],
    'End': [9300628, 10942522, 22149559, 26121326],
    'Copy_Number': [6, 10, 8, 6],
    'Major_Copy_Number': [3, 5, 4, 3],
    'Minor_Copy_Number': [3, 5, 4, 3]
}
cnv_df = pd.DataFrame(cnv_data)
'''

for file in tqdm(os.listdir(allele_specific_cnv_dir)):
    cnv_data = pd.read_csv(os.path.join(allele_specific_cnv_dir, file), sep='\t')
    cnv_pr = pr.PyRanges(cnv_data)

    # Spatial overlap mapping: overlap() finds intersections between genes and chromosomes segments

    # .join() joins genes with copy number segments where there is overlap
    # how='containment' -> gene completely inside the segment; remove to have also those partially touching
    mapped_pr = genes_pr.join(cnv_pr, how='containment')
    mapped_df = mapped_pr.as_df()
    mapped_df = mapped_df.rename(columns={"Start_b": "Start_segm", "End_b": "End_segm"})
    mapped_df.drop(columns=['GDC_Aliquot'], inplace=True)  # same sample -> same aliquot

    """"" Adding Features for GNN """""

    # Loss of Heterozygosity: 1 if minor allele is 0 and at least 1 major copy, else 0
    mapped_df['LOH'] = ((mapped_df['Minor_Copy_Number'] == 0) & (mapped_df['Major_Copy_Number'] > 0)).astype(int)

    # Allelic Imbalance Ratio
    mapped_df['Allelic_Imbalance'] = (mapped_df['Major_Copy_Number'] - mapped_df['Minor_Copy_Number']) / mapped_df['Copy_Number']
    mapped_df['Allelic_Imbalance'] = mapped_df['Allelic_Imbalance'].fillna(0)  # to handle case Copy_Number = 0

    # Homozygous Deletion: 1 if total gene loss, 0 else
    mapped_df['Hom_Del'] = (mapped_df['Copy_Number'] == 0).astype(int)

    # Copy-Neutral LOH: 1 if two total copies but both from only one parent, else 0
    mapped_df['CnLOH'] = ((mapped_df['Copy_Number'] == 2) & (mapped_df['Minor_Copy_Number'] == 0)).astype(int)

    # file saved to be used in PyTorch Geometric
    mapped_df.to_csv(os.path.join(dir_to_save, f"genes.{file}"), sep="\t", index=False)
