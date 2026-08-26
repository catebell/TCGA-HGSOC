import os
import torch
from torch_geometric.data import Dataset

# columns about future to be excluded because they give away the prediction (when target class is one of them)
leaking_cols = [
    'primary_therapy_resistance',
    'had_progression_therapy',
    'days_to_new_tumor_event_after_initial_treatment',
    'overall_survival_days',
    'deceased',
    'total_drug_therapy_duration_days',
    'therapy_ongoing',
]


class PatientTissueDataset(Dataset):
    def __init__(self, graphs_dir, clinical_df, task, target_col=None, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.graphs_dir = graphs_dir
        self.graph_filenames = [f for f in os.listdir(graphs_dir) if f.endswith('.pt')]

        # preventing duplicates (should not be any)
        self.clinical_df = clinical_df.drop_duplicates(subset=['bcr_patient_barcode']).set_index('bcr_patient_barcode')
        self.target_col = target_col
        self.task = task

        if task == 'classification':
            self.clinical_feature_cols = [c for c in self.clinical_df.columns if c not in [target_col]
                                          and c not in leaking_cols
                                          and not c.startswith('new_neoplasm_event_type_')
                                          and not c.startswith('progression_determined_by_')
                                          and not c.startswith('person_neoplasm_cancer_status_')]
        elif task == 'survival':
            self.clinical_feature_cols = [c for c in self.clinical_df.columns if c not in ['overall_survival_days', 'deceased']
                                          and c not in leaking_cols
                                          and not c.startswith('new_neoplasm_event_type_')
                                          and not c.startswith('progression_determined_by_')
                                          and not c.startswith('person_neoplasm_cancer_status_')]


    def len(self):
        return len(self.graph_filenames)

    def get(self, idx):
        file_path = os.path.join(self.graphs_dir, self.graph_filenames[idx])
        data = torch.load(file_path, weights_only=False)

        # 'TCGA-24-1469-01A' -> 'TCGA-24-1469'
        patient_id = "-".join(data.sample_id.split("-")[:3])

        # label (y) association with patient
        if patient_id in self.clinical_df.index:
            if self.task == 'classification':
                y_val = self.clinical_df.loc[patient_id, self.target_col]
                data.y = torch.tensor([y_val], dtype=torch.long)  # torch.float if regression
            elif self.task == 'survival':
                target_time = self.clinical_df.loc[patient_id, 'overall_survival_days']
                target_event = self.clinical_df.loc[patient_id, 'deceased']

                data.y_time = torch.tensor([target_time], dtype=torch.float)
                data.y_event = torch.tensor([target_event], dtype=torch.float)

            # clinical features association
            clin_feats = self.clinical_df.loc[patient_id, self.clinical_feature_cols].values
            data.clinical_x = torch.tensor(clin_feats, dtype=torch.float).unsqueeze(0)  # Shape [1, Num_Clin_Feats]

        else:  # fallback
            data.clinical_x = torch.zeros((1, len(self.clinical_feature_cols)), dtype=torch.float)
            if self.task == 'classification':
                data.y = torch.tensor([0], dtype=torch.long)
            elif self.task == 'survival':
                data.y_time = torch.tensor([0.0], dtype=torch.float)
                data.y_event = torch.tensor([0.0], dtype=torch.float)

        # TODO magari cambiare fallback in filtraggio a monte per evitare problemi con COX, ex:
        """
        import os
        import torch
        from torch_geometric.data import Dataset
        
        leaking_cols = [
            'primary_therapy_resistance',
            'had_progression_therapy',
            'days_to_new_tumor_event_after_initial_treatment',
            'overall_survival_days',
            'deceased',
            'total_drug_therapy_duration_days',
            'therapy_ongoing',
        ]
        
        
        class PatientTissueDataset(Dataset):
            def __init__(self, graphs_dir, clinical_df, task='survival', target_col=None, transform=None, pre_transform=None):
                super().__init__(None, transform, pre_transform)
                self.graphs_dir = graphs_dir
                self.clinical_df = clinical_df.drop_duplicates(subset=['bcr_patient_barcode']).set_index('bcr_patient_barcode')
                self.target_col = target_col
                self.task = task
        
                # 1. Filtra a monte i grafi per tenere solo i pazienti presenti nel df clinico
                all_files = [f for f in os.listdir(graphs_dir) if f.endswith('.pt')]
                self.graph_filenames = []
                for f in all_files:
                    # Estrazione rapida del patient_id (evita di caricare tutto il grafo se non necessario)
                    # Presume il naming standard dei file o controlla il mapping
                    pid = "-".join(f.split("-")[:3])
                    if pid in self.clinical_df.index:
                        self.graph_filenames.append(f)
        
                # 2. Selezione feature cliniche
                cols_to_exclude = set(leaking_cols)
                if target_col:
                    cols_to_exclude.add(target_col)
                if task == 'survival':
                    cols_to_exclude.update(['overall_survival_days', 'deceased'])
        
                self.clinical_feature_cols = [
                    c for c in self.clinical_df.columns 
                    if c not in cols_to_exclude
                    and not c.startswith('new_neoplasm_event_type_')
                    and not c.startswith('progression_determined_by_')
                    and not c.startswith('person_neoplasm_cancer_status_')
                ]
        
            def len(self):
                return len(self.graph_filenames)
        
            def get(self, idx):
                file_path = os.path.join(self.graphs_dir, self.graph_filenames[idx])
                data = torch.load(file_path, weights_only=False)
        
                # 'TCGA-24-1469-01A' -> 'TCGA-24-1469'
                patient_id = "-".join(data.sample_id.split("-")[:3])
        
                # Feature cliniche (assicura shape 2D: [1, Num_Clin_Feats])
                clin_feats = self.clinical_df.loc[patient_id, self.clinical_feature_cols].values
                data.clinical_x = torch.tensor(clin_feats, dtype=torch.float).unsqueeze(0)
        
                # Assegnazione Target (scalari 1D che PyG concatenera' in vettori [Batch_Size])
                if self.task == 'classification':
                    y_val = self.clinical_df.loc[patient_id, self.target_col]
                    data.y = torch.tensor(y_val, dtype=torch.long)
                    
                elif self.task == 'survival':
                    target_time = self.clinical_df.loc[patient_id, 'overall_survival_days']
                    target_event = self.clinical_df.loc[patient_id, 'deceased']
        
                    data.y_time = torch.tensor(target_time, dtype=torch.float)
                    data.y_event = torch.tensor(target_event, dtype=torch.float)
        
                return data
        
            def update_clinical_df(self, new_df_clinical):
                self.clinical_df = new_df_clinical.drop_duplicates(subset=['bcr_patient_barcode']).set_index('bcr_patient_barcode')
        """

        return data

    def update_clinical_df(self, new_df_clinical):
        self.clinical_df = new_df_clinical.drop_duplicates(subset=['bcr_patient_barcode']).set_index(
            'bcr_patient_barcode')  # preventing duplicates
