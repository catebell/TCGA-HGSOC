import re

import numpy as np
import pandas as pd


def parse_list(val):
    """ List parsing (es. value = "Alive, Dead"  --> ["Alive", "Dead"]) """
    if pd.isna(val) or str(val).strip() == '':
        return []
    return [x.strip() for x in str(val).split(',') if x.strip() != '']


def parse_float_list(val):
    """ List parsing (es. value = "600, 900"  --> ["600", "900"]) """
    items = parse_list(val)
    res = []
    for x in items:
        try:
            res.append(float(x))
        except ValueError:
            pass
    return res


def standardize_drug_type(drug_str):
    """ Pattern matching / RegEx rules for mapping drugs to classes or active principles. """
    if not isinstance(drug_str, str) or pd.isna(drug_str):
        return []

    text = drug_str.lower().strip()
    found_drugs = set()
    matched = False  # if at least one match found in list

    if re.search(r'carbo|cbp|ccdp|cisplat|ciplas', text):
        matched = True
        if 'oxali' not in text:
            found_drugs.add('Platinum (Carboplatin/Cisplatin)')
        else:
            found_drugs.add('Platinum (Oxaliplatin)')

    if re.search(r'pacli|taxol|docet|doxet|taxat|taxan|abraxane|xyotax|ct2103', text):
        matched = True
        found_drugs.add('Taxanes (Paclitaxel/Docetaxel)')

    if re.search(r'dox|adriamycin|docorubicin', text):
        matched = True
        found_drugs.add('Anthracyclines (Doxorubicin/Doxil)')

    if re.search(r'gem|gamzar|gezmar|alimta|altima|xeloda', text):
        matched = True
        found_drugs.add('Antimetabolites (Gemcitabine)')

    if re.search(r'beva|avastin|cediranib|amg\s*706|amgen|ba4|cep\s*11981', text):
        matched = True
        found_drugs.add('Targeted (Anti-VEGF/Bevacizumab)')

    if re.search(r'topo|topecan|toptecan|cpt\s*11|irino|vp\s*16|etop|gaminocampto', text):
        matched = True
        found_drugs.add('Topoisomerase Inhibitors (Topotecan/Etoposide)')

    if re.search(r'cytox|cyclophosphamide|ifosf|melphal|hexalen|hexalin|hexamethyl', text):
        matched = True
        found_drugs.add('Alkylating Agents (Cytoxan/Ifosfamide)')

    if re.search(r'azd\s*2281|olaparib', text):
        matched = True
        found_drugs.add('PARP Inhibitors')

    if re.search(
            r'arimidex|armidex|aromasin|femara|letrozole|faslodex|fuluestrant|lupron|megace|megestrol|provera|tamoxifen',
            text):
        matched = True
        found_drugs.add('Hormone Therapy')

    if re.search(
            r'abagovom|ovarex|oregovomab|catumaxumab|herceptin|cetuximab|il\s*12|il\s*2|interferon|sargramostin',
            text):
        matched = True
        found_drugs.add('Immunotherapy/Antibody')

    if re.search(r'navelbine|vinorelbine|vincristine', text):
        matched = True
        found_drugs.add('Vinca Alkaloid')

    if not matched:
        found_drugs.add('Others')

    return list(found_drugs)


def infer_missing_vital_status(df):
    """
    if vital_status == nan but days_to_death has a value --> dead;
    else, if days_to_last_followup has a value --> alive;
    else, unknown --> remove patient
    """

    cond = (df['vital_status'].isna()) & (df['days_to_death'].notna())
    df.loc[cond, 'vital_status'] = 1.0

    cond = df['vital_status'].isna() & df['days_to_death'].isna() & df['days_to_last_followup'].notna()
    df.loc[cond, 'vital_status'] = 0.0

    return df.dropna(subset=['vital_status'])


def infer_missing_primary_therapy_outcome_success(row):
    """
    if patient latest status is TUMOR FREE and no progression_therapy -> Complete Remission, responsive (0)
    if progression confirmed or new event -> Progressive Disease, resistant (1)
    else, unknown --> nan
    """

    if row['primary_therapy_outcome_success'] != np.nan:
        return row['primary_therapy_outcome_success']  # known value if not nan

    if row['person_neoplasm_cancer_status'] == 'TUMOR FREE' and row['had_progression_therapy'] == 0.0:
        # return 1.0
        return 0  # responder/sensible to therapy

    if row['had_progression_therapy'] == 1.0 or row.get('new_event_Progression of Disease', 0) == 1:
        # return 4.0
        return 1  # resistant

    return np.nan  # leave nan


def infer_had_progression_therapy(df):
    """
    if recurrence col is present, patients that didn't have a recurrence obv didn't have progr. therapy either
    """
    if 'had_progression_therapy' in df.columns:
        if 'has_recurrence_event' in df.columns:
            cond = (df['has_recurrence_event'] == 0) & (df['had_progression_therapy'].isna())
            df.loc[cond, 'had_progression_therapy'] = 0.0

    return df



""""" feature specific processing functions """""


def clean_vital_status(val):
    """ Return last/more recent status recorded. """
    statuses = parse_list(val)
    return statuses[-1].strip() if statuses else None


def clean_days_to_last_followup(val):
    """ Return days to most recent followup. """
    days = parse_list(val)
    return days[-1].strip() if days else None


def clean_days_to_death(val):
    """ Return days to death. """
    days = parse_list(val)
    return days[-1].strip() if days else None


def clean_drug_name(val):
    """ Return list of standardized drug categories. """
    raw_drugs = parse_list(val)
    standardized_drugs = set()

    for d in raw_drugs:
        cleaned = standardize_drug_type(d)
        if cleaned:
            standardized_drugs.update(cleaned)
    return sorted(standardized_drugs) if standardized_drugs else []


def clean_number_cycles(val):
    """ Return tot sum of therapy cycles. """
    nums = parse_float_list(val)
    return sum(nums) if nums else None


def compute_therapy_duration(row):
    """ Computes days_to_drug_therapy_end - days_to_drug_therapy_start."""
    starts = parse_float_list(row['days_to_drug_therapy_start'])
    ends = parse_float_list(row['days_to_drug_therapy_end'])
    if starts and ends:
        return max(ends) - min(starts)
    return None


def clean_person_neoplasm_cancer_status(val):
    """ Return last cancer status registered. """
    statuses = parse_list(val)
    return statuses[-1] if statuses else None


def clean_days_to_new_tumor_event_after_initial_treatment(val):
    """ Return minimum number of days to first event. """
    days = parse_float_list(val)
    return min(days) if days else None


def compute_if_therapy_progression(val):
    """ Return 1 if therapy had been administered because of disease progression, 0 else. """
    regs = parse_list(val)
    return float(('PROGRESSION' or 'RECURRENCE') in [r.upper() for r in regs])


def clean_therapy_type(val):
    """ Return list of therapy types without repetitions. """
    raw_types = parse_list(val)
    types = set(raw_types)
    return types if types else []


def clean_therapy_ongoing(val):
    """ Return 1 if at least a YES is listed, else 0. """
    items = [x.upper() for x in parse_list(val)]
    return float('YES' in items)


def clean_progression_determined_by(val):
    """ Return list of methods by which progression was determined. """
    items = parse_list(val)
    unique_methods = sorted(list(set(items)))
    return unique_methods if unique_methods else []


def clean_postoperative_rx_tx(val):
    """ Return 1 if at least a YES is listed (patient received therapy after primary tumor resection), else 0. """
    items = parse_list(val)
    return 1 if 'YES' in items else (0 if 'NO' in items else None)


def clean_primary_therapy_outcome_success(val):
    """ Return the worst clinical outcome in record:
    'Complete Remission/Response': 1,
    'Partial Remission/Response': 2,
    'Stable Disease': 3,
    'Progressive Disease': 4 """

    outcomes = parse_list(val)
    worst = None
    '''
    if 'Complete Remission/Response' in outcomes:
        worst = 1
    if 'Partial Remission/Response' in outcomes:
        worst = 2
    if 'Stable Disease' in outcomes:
        worst = 3
    if 'Progressive Disease' in outcomes:
        worst = 4
    '''
    if 'Complete Remission/Response' in outcomes:
        worst = 0  # responder / sensible to therapy
    if 'Partial Remission/Response' in  outcomes or 'Stable Disease' in  outcomes or 'Progressive Disease' in  outcomes:
        worst = 1  # resistant
    return worst
