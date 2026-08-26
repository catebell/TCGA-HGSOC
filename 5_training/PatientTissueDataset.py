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


'''
Con colonne leaky :(

2026-08-11 12:17:17,899 - INFO - Device: cuda
2026-08-11 12:17:17,899 - INFO - Starting training using target feature: primary_therapy_resistance

2026-08-11 12:17:17,900 - INFO - Train Dataset init...
2026-08-11 12:17:17,978 - INFO - Val Dataset init...
2026-08-11 12:17:18,075 - INFO - Test Dataset init...
2026-08-11 12:17:18,108 - INFO - Num Clinical Features found : 44
2026-08-11 12:17:46,497 - INFO - Unique labels found in dataset: tensor([0, 1])
2026-08-11 12:17:46,672 - INFO - classes_counts:
2026-08-11 12:17:46,672 - INFO - tensor([370,  79])
2026-08-11 12:17:46,673 - INFO - weights:
2026-08-11 12:17:46,673 - INFO - tensor([0.6068, 2.8418], device='cuda:0')
2026-08-11 12:17:46,809 - INFO - --- FOLD 1 ---
2026-08-11 12:17:46,859 - INFO - MultiOmicGAT(
  (gat1): GATv2Conv(10, 32, heads=4)
  (residual_proj): Linear(in_features=10, out_features=128, bias=True)
  (gat2): GATv2Conv(128, 32, heads=1)
  (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
  (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  (clinical_encoder): Sequential(
    (0): Linear(in_features=44, out_features=32, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=64, out_features=16, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=16, out_features=2, bias=True)
  )
)
  0%|          | 0/50 [00:00<?, ?it/s]2026-08-11 12:17:52,775 - INFO - Epoch 01 | Train Loss: 0.7031, Train Acc: 0.5794 | Val Loss: 0.6789, Val Acc: 0.7556
2026-08-11 12:17:52,775 - INFO - IMPROVEMENT!

  2%|▏         | 1/50 [00:06<05:09,  6.31s/it]2026-08-11 12:17:58,644 - INFO - Epoch 02 | Train Loss: 0.6784, Train Acc: 0.6852 | Val Loss: 0.6595, Val Acc: 0.8222
2026-08-11 12:17:58,644 - INFO - IMPROVEMENT!

  4%|▍         | 2/50 [00:11<04:39,  5.83s/it]2026-08-11 12:18:04,080 - INFO - Epoch 03 | Train Loss: 0.6486, Train Acc: 0.6852 | Val Loss: 0.6200, Val Acc: 0.8111
2026-08-11 12:18:04,080 - INFO - IMPROVEMENT!

  6%|▌         | 3/50 [00:17<04:25,  5.64s/it]2026-08-11 12:18:09,697 - INFO - Epoch 04 | Train Loss: 0.6052, Train Acc: 0.6908 | Val Loss: 0.5580, Val Acc: 0.8000
2026-08-11 12:18:09,698 - INFO - IMPROVEMENT!

  8%|▊         | 4/50 [00:22<04:20,  5.65s/it]2026-08-11 12:18:15,180 - INFO - Epoch 05 | Train Loss: 0.5500, Train Acc: 0.7939 | Val Loss: 0.4844, Val Acc: 0.7667
2026-08-11 12:18:15,180 - INFO - IMPROVEMENT!

 10%|█         | 5/50 [00:28<04:10,  5.57s/it]2026-08-11 12:18:21,075 - INFO - Epoch 06 | Train Loss: 0.4966, Train Acc: 0.7855 | Val Loss: 0.4208, Val Acc: 0.8444
2026-08-11 12:18:21,075 - INFO - IMPROVEMENT!

 12%|█▏        | 6/50 [00:34<04:10,  5.68s/it]2026-08-11 12:18:26,733 - INFO - Epoch 07 | Train Loss: 0.4406, Train Acc: 0.8106 | Val Loss: 0.3708, Val Acc: 0.8444
2026-08-11 12:18:26,733 - INFO - IMPROVEMENT!

 14%|█▍        | 7/50 [00:39<04:04,  5.68s/it]2026-08-11 12:18:32,024 - INFO - Epoch 08 | Train Loss: 0.4191, Train Acc: 0.8134 | Val Loss: 0.3378, Val Acc: 0.8111
2026-08-11 12:18:32,025 - INFO - IMPROVEMENT!

 16%|█▌        | 8/50 [00:45<03:53,  5.55s/it]2026-08-11 12:18:37,204 - INFO - Epoch 09 | Train Loss: 0.3672, Train Acc: 0.8496 | Val Loss: 0.3115, Val Acc: 0.8556
2026-08-11 12:18:37,204 - INFO - IMPROVEMENT!

 18%|█▊        | 9/50 [00:50<03:42,  5.44s/it]2026-08-11 12:18:42,320 - INFO - Epoch 10 | Train Loss: 0.3342, Train Acc: 0.8468 | Val Loss: 0.2908, Val Acc: 0.9111
2026-08-11 12:18:42,320 - INFO - IMPROVEMENT!

 20%|██        | 10/50 [00:55<03:33,  5.34s/it]2026-08-11 12:18:47,246 - INFO - Epoch 11 | Train Loss: 0.3492, Train Acc: 0.8524 | Val Loss: 0.2808, Val Acc: 0.8889
2026-08-11 12:18:47,246 - INFO - IMPROVEMENT!

 22%|██▏       | 11/50 [01:00<03:23,  5.21s/it]2026-08-11 12:18:52,390 - INFO - Epoch 12 | Train Loss: 0.3478, Train Acc: 0.8552 | Val Loss: 0.2864, Val Acc: 0.8222
 24%|██▍       | 12/50 [01:05<03:17,  5.19s/it]2026-08-11 12:18:57,561 - INFO - Epoch 13 | Train Loss: 0.3139, Train Acc: 0.8663 | Val Loss: 0.2681, Val Acc: 0.8556
2026-08-11 12:18:57,562 - INFO - IMPROVEMENT!

 26%|██▌       | 13/50 [01:10<03:11,  5.19s/it]2026-08-11 12:19:02,490 - INFO - Epoch 14 | Train Loss: 0.3192, Train Acc: 0.8496 | Val Loss: 0.2593, Val Acc: 0.8889
2026-08-11 12:19:02,490 - INFO - IMPROVEMENT!

 28%|██▊       | 14/50 [01:15<03:03,  5.11s/it]2026-08-11 12:19:07,909 - INFO - Epoch 15 | Train Loss: 0.2835, Train Acc: 0.8719 | Val Loss: 0.2541, Val Acc: 0.8444
2026-08-11 12:19:07,909 - INFO - IMPROVEMENT!

 30%|███       | 15/50 [01:21<03:02,  5.20s/it]2026-08-11 12:19:13,394 - INFO - Epoch 16 | Train Loss: 0.3193, Train Acc: 0.8579 | Val Loss: 0.2528, Val Acc: 0.9111
2026-08-11 12:19:13,395 - INFO - IMPROVEMENT!

 32%|███▏      | 16/50 [01:26<02:59,  5.29s/it]2026-08-11 12:19:19,193 - INFO - Epoch 17 | Train Loss: 0.3243, Train Acc: 0.8552 | Val Loss: 0.2501, Val Acc: 0.8333
2026-08-11 12:19:19,193 - INFO - IMPROVEMENT!

 34%|███▍      | 17/50 [01:32<02:59,  5.44s/it]2026-08-11 12:19:24,853 - INFO - Epoch 18 | Train Loss: 0.2961, Train Acc: 0.8774 | Val Loss: 0.2394, Val Acc: 0.8889
2026-08-11 12:19:24,854 - INFO - IMPROVEMENT!

 36%|███▌      | 18/50 [01:37<02:56,  5.51s/it]2026-08-11 12:19:30,567 - INFO - Epoch 19 | Train Loss: 0.3133, Train Acc: 0.8607 | Val Loss: 0.2435, Val Acc: 0.8778
 38%|███▊      | 19/50 [01:43<02:52,  5.57s/it]2026-08-11 12:19:36,596 - INFO - Epoch 20 | Train Loss: 0.2829, Train Acc: 0.8774 | Val Loss: 0.2381, Val Acc: 0.8778
2026-08-11 12:19:36,596 - INFO - IMPROVEMENT!

 40%|████      | 20/50 [01:49<02:51,  5.71s/it]2026-08-11 12:19:42,322 - INFO - Epoch 21 | Train Loss: 0.2758, Train Acc: 0.8747 | Val Loss: 0.2444, Val Acc: 0.8444
 42%|████▏     | 21/50 [01:55<02:45,  5.71s/it]2026-08-11 12:19:47,993 - INFO - Epoch 22 | Train Loss: 0.2749, Train Acc: 0.9025 | Val Loss: 0.2330, Val Acc: 0.8667
2026-08-11 12:19:47,993 - INFO - IMPROVEMENT!

 44%|████▍     | 22/50 [02:01<02:39,  5.70s/it]2026-08-11 12:19:53,771 - INFO - Epoch 23 | Train Loss: 0.2511, Train Acc: 0.8774 | Val Loss: 0.2191, Val Acc: 0.8889
2026-08-11 12:19:53,771 - INFO - IMPROVEMENT!

 46%|████▌     | 23/50 [02:06<02:34,  5.73s/it]2026-08-11 12:19:59,581 - INFO - Epoch 24 | Train Loss: 0.2847, Train Acc: 0.8997 | Val Loss: 0.2227, Val Acc: 0.8667
 48%|████▊     | 24/50 [02:12<02:29,  5.75s/it]2026-08-11 12:20:05,489 - INFO - Epoch 25 | Train Loss: 0.2387, Train Acc: 0.9053 | Val Loss: 0.2187, Val Acc: 0.8889
2026-08-11 12:20:05,489 - INFO - IMPROVEMENT!

 50%|█████     | 25/50 [02:18<02:24,  5.80s/it]2026-08-11 12:20:11,807 - INFO - Epoch 26 | Train Loss: 0.2391, Train Acc: 0.8969 | Val Loss: 0.2215, Val Acc: 0.8667
 52%|█████▏    | 26/50 [02:24<02:22,  5.95s/it]2026-08-11 12:20:18,461 - INFO - Epoch 27 | Train Loss: 0.2235, Train Acc: 0.8969 | Val Loss: 0.2104, Val Acc: 0.9000
2026-08-11 12:20:18,462 - INFO - IMPROVEMENT!

 54%|█████▍    | 27/50 [02:31<02:21,  6.17s/it]2026-08-11 12:20:24,684 - INFO - Epoch 28 | Train Loss: 0.2661, Train Acc: 0.9025 | Val Loss: 0.2266, Val Acc: 0.8556
 56%|█████▌    | 28/50 [02:37<02:15,  6.18s/it]2026-08-11 12:20:31,781 - INFO - Epoch 29 | Train Loss: 0.2323, Train Acc: 0.9025 | Val Loss: 0.2102, Val Acc: 0.9000
2026-08-11 12:20:31,781 - INFO - IMPROVEMENT!

 58%|█████▊    | 29/50 [02:44<02:15,  6.46s/it]2026-08-11 12:20:38,232 - INFO - Epoch 30 | Train Loss: 0.2156, Train Acc: 0.9109 | Val Loss: 0.2052, Val Acc: 0.9111
2026-08-11 12:20:38,232 - INFO - IMPROVEMENT!

 60%|██████    | 30/50 [02:51<02:09,  6.45s/it]2026-08-11 12:20:44,298 - INFO - Epoch 31 | Train Loss: 0.2027, Train Acc: 0.9276 | Val Loss: 0.2097, Val Acc: 0.8889
 62%|██████▏   | 31/50 [02:57<02:00,  6.34s/it]2026-08-11 12:20:49,875 - INFO - Epoch 32 | Train Loss: 0.1947, Train Acc: 0.9053 | Val Loss: 0.2178, Val Acc: 0.8889
 64%|██████▍   | 32/50 [03:03<01:49,  6.11s/it]2026-08-11 12:20:56,393 - INFO - Epoch 33 | Train Loss: 0.2181, Train Acc: 0.9081 | Val Loss: 0.2156, Val Acc: 0.9000
 66%|██████▌   | 33/50 [03:09<01:45,  6.23s/it]2026-08-11 12:21:02,017 - INFO - Epoch 34 | Train Loss: 0.2210, Train Acc: 0.9192 | Val Loss: 0.2168, Val Acc: 0.8889
 68%|██████▊   | 34/50 [03:15<01:36,  6.05s/it]2026-08-11 12:21:07,323 - INFO - Epoch 35 | Train Loss: 0.1934, Train Acc: 0.9276 | Val Loss: 0.2182, Val Acc: 0.9111
 70%|███████   | 35/50 [03:20<01:27,  5.83s/it]2026-08-11 12:21:13,316 - INFO - Epoch 36 | Train Loss: 0.1794, Train Acc: 0.9331 | Val Loss: 0.2129, Val Acc: 0.8889
 72%|███████▏  | 36/50 [03:26<01:22,  5.88s/it]2026-08-11 12:21:20,857 - INFO - Epoch 37 | Train Loss: 0.2075, Train Acc: 0.9053 | Val Loss: 0.2138, Val Acc: 0.8889
 74%|███████▍  | 37/50 [03:33<01:22,  6.38s/it]2026-08-11 12:21:26,897 - INFO - Epoch 38 | Train Loss: 0.1986, Train Acc: 0.9304 | Val Loss: 0.2277, Val Acc: 0.9000
 76%|███████▌  | 38/50 [03:40<01:15,  6.27s/it]2026-08-11 12:21:33,690 - INFO - Epoch 39 | Train Loss: 0.1704, Train Acc: 0.9415 | Val Loss: 0.2157, Val Acc: 0.8889
 78%|███████▊  | 39/50 [03:46<01:10,  6.43s/it]2026-08-11 12:21:39,627 - INFO - Epoch 40 | Train Loss: 0.1800, Train Acc: 0.9164 | Val Loss: 0.2470, Val Acc: 0.8889
 80%|████████  | 40/50 [03:52<01:02,  6.28s/it]2026-08-11 12:21:45,925 - INFO - Epoch 41 | Train Loss: 0.1844, Train Acc: 0.9359 | Val Loss: 0.2128, Val Acc: 0.8889
 82%|████████▏ | 41/50 [03:59<00:56,  6.29s/it]2026-08-11 12:21:51,599 - INFO - Epoch 42 | Train Loss: 0.1902, Train Acc: 0.9109 | Val Loss: 0.2231, Val Acc: 0.9000
 84%|████████▍ | 42/50 [04:04<00:48,  6.10s/it]2026-08-11 12:21:57,754 - INFO - Epoch 43 | Train Loss: 0.1900, Train Acc: 0.9359 | Val Loss: 0.2124, Val Acc: 0.8889
 86%|████████▌ | 43/50 [04:10<00:42,  6.12s/it]2026-08-11 12:22:03,484 - INFO - Epoch 44 | Train Loss: 0.1777, Train Acc: 0.9053 | Val Loss: 0.2186, Val Acc: 0.9000
 88%|████████▊ | 44/50 [04:16<00:36,  6.00s/it]2026-08-11 12:22:09,296 - INFO - Epoch 45 | Train Loss: 0.1631, Train Acc: 0.9471 | Val Loss: 0.2166, Val Acc: 0.9111
 90%|█████████ | 45/50 [04:22<00:29,  5.95s/it]2026-08-11 12:22:14,896 - INFO - Epoch 46 | Train Loss: 0.1748, Train Acc: 0.9331 | Val Loss: 0.2153, Val Acc: 0.9000
 92%|█████████▏| 46/50 [04:28<00:23,  5.84s/it]2026-08-11 12:22:21,183 - INFO - Epoch 47 | Train Loss: 0.1918, Train Acc: 0.9136 | Val Loss: 0.2003, Val Acc: 0.9111
2026-08-11 12:22:21,183 - INFO - IMPROVEMENT!

 94%|█████████▍| 47/50 [04:34<00:17,  5.98s/it]2026-08-11 12:22:27,591 - INFO - Epoch 48 | Train Loss: 0.1869, Train Acc: 0.9276 | Val Loss: 0.2017, Val Acc: 0.8889
 96%|█████████▌| 48/50 [04:40<00:12,  6.10s/it]2026-08-11 12:22:34,871 - INFO - Epoch 49 | Train Loss: 0.1883, Train Acc: 0.9359 | Val Loss: 0.2030, Val Acc: 0.9000
 98%|█████████▊| 49/50 [04:48<00:06,  6.46s/it]2026-08-11 12:22:40,946 - INFO - Epoch 50 | Train Loss: 0.1612, Train Acc: 0.9331 | Val Loss: 0.2177, Val Acc: 0.9000
100%|██████████| 50/50 [04:54<00:00,  5.88s/it]
'''