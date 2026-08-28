import json
import os.path

import pandas as pd

import task_target

path_to_metadata = ""
path_to_save = ""

with open(os.path.join(path_to_metadata, f'metadata.repository.{task_target.DATASET}.json'), 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# key = "PatientID_SampleType" (es. "TCGA-29-1692_01A")
samples_map = {}

for item in metadata:
    file_name = item.get("file_name", "")

    if file_name:
        splits = file_name.split(".") if file_name else ""
        file_uuid = ""
        for i in splits:
            if len(i) > 20:
                file_uuid = i
                break
    else:
        continue

    data_category = item.get("data_category", "")
    data_type = item.get("data_type", "")

    # Temporary list for (patient, tissue) associated to this file
    current_file_targets = []

    associated_entities = item.get("associated_entities", [])

    if associated_entities:
        for entity in associated_entities:  # a single file can be associated with more samples
            entity_id = entity.get("entity_submitter_id", "")
            parts = entity_id.split("-")

            if len(parts) >= 3 and parts[0] == "TCGA":
                p_id = f"{parts[0]}-{parts[1]}-{parts[2]}"
                s_type = parts[3] if len(parts) >= 4 else "Unknown"  # some entities only have the first part (patient_id)
                current_file_targets.append((p_id, s_type))

    # if it's a clinical file with no associated_entities, we extract patient_id from the XML title
    elif data_category == "Clinical" and "TCGA-" in file_name:
        for part in file_name.split("."):
            if part.startswith("TCGA-"):
                current_file_targets.append((part, "Clinical_Global"))
                break

    # if no target identified, skip
    if not current_file_targets:
        continue

    # file association with every target found:
    for patient_id, sample_type in current_file_targets:

        # unique key for row = patient_id+sample_id (es. "TCGA-29-1692_01A"), only patient_id if file is clinical
        if sample_type == "Clinical_Global":
            sample_key = f"{patient_id}_Clinical"
        else:
            sample_key = f"{patient_id}_{sample_type}"

        # Initialize tissue record in dict if not exists
        if sample_key not in samples_map:
            sample_id = patient_id + "-" + sample_type if sample_type != "Clinical_Global" else None
            samples_map[sample_key] = {
                "Patient_ID": patient_id,
                "Sample_Type": sample_type if sample_type != "Clinical_Global" else None,
                "Sample_ID": sample_id,
                "Clinical_Data": None,
                "RNA_File_UUID": None,
                "CNV_Allele_Specific_UUID": None,
            }

        # filling fields
        if data_category == "Clinical":
            samples_map[sample_key]["Clinical_Data"] = "present"

        elif data_category == "Transcriptome Profiling":
            #if "Gene Expression" in data_type:
            if "rna_seq" in file_name:
                samples_map[sample_key]["RNA_File_UUID"] = file_uuid

        elif data_category == "Copy Number Variation":
            #if "Allele-specific" in data_type:
            if "allelic" in file_name:
                samples_map[sample_key]["CNV_Allele_Specific_UUID"] = file_uuid

df_final_mapping = pd.DataFrame(list(samples_map.values()))


# clinical files don't have a tissue-type. We propagate "Clinical_Data = present" to all tissues of related patient.
clinical_patients = df_final_mapping[(df_final_mapping['Sample_Type']=="Unknown") & (df_final_mapping['Clinical_Data'] == 'present')][
    'Patient_ID'].unique()
df_final_mapping.loc[df_final_mapping['Patient_ID'].isin(clinical_patients), 'Clinical_Data'] = 'present'

# purely clinical rows deleted
df_final_mapping = df_final_mapping.replace("Unknown", pd.NA).dropna(subset=['Sample_Type'])

# rows about normal/control tissues deleted (not useful for the task, already internally used for their values)
normal_tissues_subset = df_final_mapping[(df_final_mapping["Sample_Type"].str.contains("10")) |
                                         (df_final_mapping["Sample_Type"].str.contains("11"))]
df_final_mapping = df_final_mapping.drop(normal_tissues_subset.index).reset_index()

columns_order = [
    "Patient_ID", "Sample_Type", "Sample_ID", "Clinical_Data",
    "RNA_File_UUID", "CNV_Allele_Specific_UUID"
]
df_final_mapping = df_final_mapping[columns_order]

# remove patients with no Gene Expression or CNV file associated
df_final_mapping = df_final_mapping.dropna(subset=["RNA_File_UUID", "CNV_Allele_Specific_UUID"], how='all')

df_final_mapping.to_csv(path_to_save + "gdc_sample_level_mapping.csv", index=False)

print(f"\nDONE! Found {len(df_final_mapping)} unique mapped tissues and {len(df_final_mapping['Patient_ID'].unique())} unique patients with molecular data.\n")
print(df_final_mapping.head(10))