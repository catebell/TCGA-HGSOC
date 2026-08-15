import os
from sklearn.preprocessing import MultiLabelBinarizer

from utils_clinical_clean_methods import *

path_to_clinical_file = os.path.join('..', '1_dataset', 'data_extracted', 'clinical_data.tsv')
df_clinical = pd.read_csv(path_to_clinical_file, sep="\t")


# removal of patients that had radiation therapy (heavily impacts CNVs)
df_clinical = df_clinical[~df_clinical['radiation_therapy'].fillna('').astype(str).str.contains('YES', case=False)]

features_to_keep = [
    "bcr_patient_barcode",
    "vital_status",
    "age_at_initial_pathologic_diagnosis",
    #"race",
    #"ethnicity",
    #"batch_number",
    #"tissue_source_site",
    "clinical_stage",
    "neoplasm_histologic_grade",
    "days_to_last_followup",
    "days_to_death",
    "tumor_residual_disease",
    "anatomic_neoplasm_subdivision ",
    "initial_pathologic_diagnosis_method",
    "primary_therapy_outcome_success",
    "drug_name",
    "days_to_drug_therapy_start",
    "days_to_drug_therapy_end",
    "number_cycles",
    "regimen_indication",
    "therapy_type",
    "therapy_ongoing",
    "postoperative_rx_tx",
    "person_neoplasm_cancer_status",
    "days_to_new_tumor_event_after_initial_treatment",
    "new_neoplasm_event_type",
    "progression_determined_by",
    "lymphatic_invasion",
    "venous_invasion",
]

# filter dataset to available desired features
available_features = [col for col in features_to_keep if col in df_clinical.columns]
df_clinical = df_clinical[available_features]

# exclude features with < 50% valid values
threshold = 0.50 * len(df_clinical)
df_filtered = df_clinical.dropna(thresh=threshold, axis=1)

dropped_cols = set(df_clinical.columns) - set(df_filtered.columns)
if dropped_cols:
    print(f"Dropped due to >50% missing values: {list(dropped_cols)}")


# TODO eventually modify for temporal history (es. values up-to-metastasis), now overall aggregation
def clean_and_aggregate_clinical_data(df):
    """ Clean cols data formatted as lists for every patient. Returns the df with cols replaced with single values or lists for later encoding. """

    df_clean = df.copy()

    if 'vital_status' in df_clean.columns:
        df_clean['vital_status'] = df_clean['vital_status'].apply(clean_vital_status)

    if 'days_to_last_followup' in df_clean.columns:
        df_clean['days_to_last_followup'] = df_clean['days_to_last_followup'].apply(clean_days_to_last_followup)

    if 'days_to_death' in df_clean.columns:
        df_clean['days_to_death'] = df_clean['days_to_death'].apply(clean_days_to_death)

    if 'postoperative_rx_tx' in df_clean.columns:
        df_clean['postoperative_rx_tx'] = df_clean['postoperative_rx_tx'].apply(clean_postoperative_rx_tx)

    if 'primary_therapy_outcome_success' in df_clean.columns:
        df_clean['primary_therapy_outcome_success'] = df_clean['primary_therapy_outcome_success'].apply(clean_primary_therapy_outcome_success)

    if 'number_cycles' in df_clean.columns:
        df_clean['number_cycles'] = df_clean['number_cycles'].apply(clean_number_cycles)

    if all(col in df_clean.columns for col in ['days_to_drug_therapy_start', 'days_to_drug_therapy_end']):
        df_clean['total_drug_therapy_duration_days'] = df_clean.apply(compute_therapy_duration, axis=1)
        df_clean = df_clean.drop(columns=['days_to_drug_therapy_start', 'days_to_drug_therapy_end'])

    if 'person_neoplasm_cancer_status' in df_clean.columns:
        df_clean['person_neoplasm_cancer_status'] = df_clean['person_neoplasm_cancer_status'].apply(clean_person_neoplasm_cancer_status)

    if 'days_to_new_tumor_event_after_initial_treatment' in df_clean.columns:
        df_clean['days_to_new_tumor_event_after_initial_treatment'] = df_clean['days_to_new_tumor_event_after_initial_treatment'].apply(clean_days_to_new_tumor_event_after_initial_treatment)

    if 'regimen_indication' in df_clean.columns:
        df_clean['had_progression_therapy'] = df_clean['regimen_indication'].apply(compute_if_therapy_progression)
        df_clean = df_clean.drop(columns=['regimen_indication'])

    if 'therapy_ongoing' in df_clean.columns:
        df_clean['therapy_ongoing'] = df_clean['therapy_ongoing'].apply(clean_therapy_ongoing)


    """"" features to be split and kept al list/set (for later Multi-Label Binarization) """""

    if 'therapy_type' in df_clean.columns:
        df_clean['therapy_type'] = df_clean['therapy_type'].apply(clean_therapy_type)

    if 'new_neoplasm_event_type' in df_clean.columns:
        df_clean['new_neoplasm_event_type'] = df_clean['new_neoplasm_event_type'].apply(parse_list)

    if 'progression_determined_by' in df_clean.columns:
        df_clean['progression_determined_by'] = df_clean['progression_determined_by'].apply(clean_progression_determined_by)

    if 'drug_name' in df_clean.columns:
        df_clean['drug_name'] = df_clean['drug_name'].apply(clean_drug_name)

    return df_clean


def map_to_num_and_fill_static(df_clean):
    """ Returns features mapped to numerical or static labels/fill_nan for later encoding and nan imputation.
    Applies also static rules to try and infer some features from others.
    Unknowns mapping become nan. """

    df_mapped = df_clean.copy()

    if "vital_status" in df_mapped.columns:
        vital_status_mapping = {
            "Alive": 0,
            "Dead": 1,
        }
        df_mapped["vital_status"] = df_mapped["vital_status"].map(vital_status_mapping)
        df_mapped = infer_missing_vital_status(df_mapped)

    if "clinical_stage" in df_mapped.columns:
        stage_mapping = {
            "Stage I": 1, "Stage IA": 1, "Stage IB": 2, "Stage IC": 3,
            "Stage II": 4, "Stage IIA": 4, "Stage IIB": 5, "Stage IIC": 6,
            "Stage III": 7, "Stage IIIA": 7, "Stage IIIB": 8, "Stage IIIC": 9,
            "Stage IV": 10,
        }
        df_mapped["clinical_stage"] = df_mapped["clinical_stage"].map(stage_mapping)

    if "neoplasm_histologic_grade" in df_mapped.columns:
        grade_mapping = {
            "G1": 1, "G2": 2,
            "G3": 3, "G4": 4,
        }
        df_mapped["neoplasm_histologic_grade"] = df_mapped["neoplasm_histologic_grade"].map(grade_mapping)

    if "tumor_residual_disease" in df_mapped.columns:
        tumor_residual_disease_mapping = {
            "No Macroscopic disease": 0,
            "1-10 mm": 1,
            "11-20 mm": 2,
            ">20 mm": 3,
        }
        df_mapped["tumor_residual_disease"] = df_mapped["tumor_residual_disease"].map(
            tumor_residual_disease_mapping)

    if "initial_pathologic_diagnosis_method" in df_mapped.columns:
        initial_pathologic_diagnosis_method_mapping = {
            "Tumor resection": "Resection",
            "Cytology (e.g. Peritoneal or pleural fluid)": "Cytology",
        }
        df_mapped["initial_pathologic_diagnosis_method"] = df_mapped["initial_pathologic_diagnosis_method"].map(
            initial_pathologic_diagnosis_method_mapping)
        df_mapped["initial_pathologic_diagnosis_method"] = df_mapped["initial_pathologic_diagnosis_method"].fillna(
            "Biopsy/Other")

    if 'person_neoplasm_cancer_status' in df_mapped.columns:
        df_mapped['person_neoplasm_cancer_status'] = df_mapped['person_neoplasm_cancer_status'].fillna('UNKNOWN')

    if 'primary_therapy_outcome_success' in df_mapped.columns:
        df_mapped['primary_therapy_outcome_success'] = df_mapped.apply(infer_missing_primary_therapy_outcome_success, axis=1)
        df_mapped = df_mapped.rename(columns={'primary_therapy_outcome_success': 'primary_therapy_resistance'})

    if 'had_progression_therapy' in df_mapped.columns:
        df_mapped = infer_had_progression_therapy(df_mapped)


    """"" days_to_death and days_to_last followup unified in --> Overall Survival """""

    if all(col in df_mapped.columns for col in ['vital_status', 'days_to_death', 'days_to_last_followup']):
        df_mapped['overall_survival_days'] = np.where(
            df_mapped['vital_status'] == 1,
            df_mapped['days_to_death'],
            df_mapped['days_to_last_followup']
        ).astype(float)
        df_mapped = df_mapped.drop(columns=['days_to_death', 'days_to_last_followup'])

    return df_mapped


def features_encoding(df_mapped):
    """ Returns categorical features encoded. """
    df_encoded = df_mapped.copy()

    """"" MultiLabelBinarizer for lists of values """""

    cols_for_mlb = ['therapy_type', 'new_neoplasm_event_type', 'progression_determined_by', 'drug_name', ]

    for col in cols_for_mlb:
        if col in df_encoded.columns:

            mlb = MultiLabelBinarizer()
            multi_hot_matrix = mlb.fit_transform(df_encoded[col])

            # new df with cols "col_Value"
            new_df = pd.DataFrame(
                multi_hot_matrix,
                columns=[f"{col}_{cls.replace(' ', '_')}" for cls in mlb.classes_],
                index=df_encoded.index
            )
            df_encoded = (pd.concat([df_encoded, new_df], axis=1))
            df_encoded = df_encoded.drop(columns=col)


    """"" One-Hot Encoding for mutually exclusive features """""

    cols_for_one_hot = ['initial_pathologic_diagnosis_method', 'person_neoplasm_cancer_status',]

    for col in cols_for_one_hot:
        if col in df_encoded.columns:
            df_encoded = pd.get_dummies(
                df_encoded,
                columns=cols_for_one_hot,
                drop_first=False,  # set True if preparing for unregularized regression models
                dtype=int,
            )

    return df_encoded


df_cleaned = clean_and_aggregate_clinical_data(df_filtered)
df_remapped = map_to_num_and_fill_static(df_cleaned)
df_encoded = features_encoding(df_remapped)

# convert float cols in integers (for label usage; feats will later be converted to floats and imputed for tensor)
df_processed = df_encoded.fillna(-1)
float_cols = df_processed.select_dtypes(include=['float']).columns
df_processed[float_cols] = df_processed[float_cols].astype(int)

print("\nFinal Processed Data Shape:", df_processed.shape)
print("\nColumns in final DataFrame:")
print(df_processed.columns.tolist())

# Save preprocessed clinical data
df_processed.to_csv("preprocessed_clinical_data.tsv", sep='\t', index=False)






'''
# separazione temporale per tessuto
def extract_tissue_clinical_features(patient_row, sample_type="01A"):
    """
    Estrae le feature cliniche corrette in base al tipo di tessuto (01A o 02A)
    evitando data leakage temporale.
    """
    # Base feature diagnostiche valide per entrambi
    clinical_dict = {
        'age': float(patient_row['age_at_initial_pathologic_diagnosis']),
        'stage': str(patient_row['clinical_stage']),
        'grade': str(patient_row['neoplasm_histologic_grade']),
        'residual_disease': str(patient_row['tumor_residual_disease'])
    }
    
    # Se il tessuto è il PRIMARIO (01A), è Naive ai trattamenti
    if sample_type == "01A":
        clinical_dict['has_prior_chemo'] = 0
        clinical_dict['total_prior_cycles'] = 0
        clinical_dict['drugs_list'] = []
        return clinical_dict

    # Se il tessuto è METASTASI/RECIDIVA (02A), calcoliamo i trattamenti SUBITI PRIMA
    elif sample_type == "02A":
        # Data di comparsa del nuovo tumore (es. 927)
        event_days_str = str(patient_row['days_to_new_tumor_event_after_initial_treatment'])
        t_event = float(event_days_str.split(',')[0]) if event_days_str != 'nan' else 0
        
        # Unpack delle liste di farmaci
        drugs = [d.strip() for d in str(patient_row['drug_name']).split(',')]
        starts = [float(s.strip()) if s.strip()!='nan' else 0 for s.strip() in str(patient_row['days_to_drug_therapy_start']).split(',')]
        cycles = [float(c.strip()) if c.strip()!='nan' else 0 for c.strip() in str(patient_row['number_cycles']).split(',')]
        
        prior_drugs = []
        prior_cycles = 0
        
        # Filtriamo solo le terapie iniziate PRIMA della recidiva
        for drug, start_day, cycle in zip(drugs, starts, cycles):
            if start_day <= t_event:
                prior_drugs.append(drug)
                prior_cycles += cycle
                
        clinical_dict['has_prior_chemo'] = 1 if len(prior_drugs) > 0 else 0
        clinical_dict['total_prior_cycles'] = prior_cycles
        clinical_dict['drugs_list'] = list(set(prior_drugs)) # Rimuove duplicati
        
        return clinical_dict
'''