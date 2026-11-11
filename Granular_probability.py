
import math
import numpy as np
import pandas as pd


# =========================================================================================
# STEP 0 - SETTINGS (change only this part)
# =========================================================================================
INPUT_FILE = "oral_cancer_expression.csv"   # your data file (.csv or .xlsx)
GENE_COLUMN = "Genes"                  # name of the column that holds the gene names
NORMAL_WORD = "normal"             # normal patient columns contain this word
CANCER_WORD = "cancer"           # cancer patient columns contain this word
GENE_TO_SHOW = None                   # name of one gene to print, e.g. "ATP2A1".
                                       # None means: use the first gene in the file


# =========================================================================================
# STEP 1 - READ THE DATA
# =========================================================================================
def read_data(file_name, gene_column, normal_word, cancer_word):
    if file_name.lower().endswith((".xlsx", ".xls")):
        table = pd.read_excel(file_name)
    else:
        table = pd.read_csv(file_name, delimiter=",")

    # repeated column names become "normal.1", "normal.2" ... so remove the part after the dot
    table.columns = table.columns.astype(str).str.split(".").str[0]

    gene_names = table[gene_column].values.tolist()
    normal = table.filter(like=normal_word).to_numpy(dtype=float)
    cancer = table.filter(like=cancer_word).to_numpy(dtype=float)
    return gene_names, np.ascontiguousarray(normal), np.ascontiguousarray(cancer)


# =========================================================================================
# STEP 2 - WHERE DO THE BINS START AND HOW WIDE ARE THEY?
# =========================================================================================
def find_start_and_width(own_values, other_values):
    start = np.mean(own_values)

    own_spread = math.sqrt(np.mean((start - own_values) ** 2))
    other_distance = math.sqrt(np.mean((start - other_values) ** 2))
    if own_spread > 0:
        lam = other_distance / own_spread
    else:
        lam = 1.0                              # all own values are equal, lambda is not needed

    below = own_values[own_values < start]     # own patients on the left side
    above = own_values[own_values > start]     # own patients on the right side

    if len(below) > 0:
        width_left = lam * math.sqrt(np.mean((start - below) ** 2))
    else:
        width_left = 0.0                       # nobody on the left, no left bins needed

    if len(above) > 0:
        width_right = lam * math.sqrt(np.mean((start - above) ** 2))
    else:
        width_right = 0.0

    return start, lam, width_left, width_right


# =========================================================================================
# STEP 3 - PUT EACH PATIENT INTO A BIN
# =========================================================================================
def put_in_bins(own_values, start, width_left, width_right):
    bin_numbers = []
    for value in own_values:
        if value < start:
            steps = math.ceil((start - value) / width_left)     # how many bin widths to the left
            bin_numbers.append(-steps)                          # -1, -2, -3 ...
        else:
            if width_right > 0:
                steps = math.floor((value - start) / width_right)
            else:
                steps = 0
            bin_numbers.append(steps)                           # 0, 1, 2 ...
    return bin_numbers


# =========================================================================================
# STEP 4 - GRANULAR PROBABILITY OF ONE GROUP
# =========================================================================================
def granular_probability_of_group(own_values, other_values):
    start, lam, width_left, width_right = find_start_and_width(own_values, other_values)
    bin_numbers = put_in_bins(own_values, start, width_left, width_right)

    group_size = len(own_values)
    probabilities = []
    for b in bin_numbers:
        patients_in_bin = bin_numbers.count(b)
        probabilities.append(patients_in_bin / group_size)
    return probabilities, bin_numbers


# =========================================================================================
# STEP 5 - GRANULAR PROBABILITY OF ONE gene (both groups)
# =========================================================================================
def granular_probability_of_gene(normal_values, cancer_values):
    prob_s, bins_s = granular_probability_of_group(normal_values, cancer_values)
    prob_r, bins_r = granular_probability_of_group(cancer_values, normal_values)
    return prob_s, bins_s, prob_r, bins_r


# =========================================================================================
# MAIN PROGRAM
# =========================================================================================
if __name__ == "__main__":

    gene_names, normal, cancer = read_data(INPUT_FILE, GENE_COLUMN, NORMAL_WORD, CANCER_WORD)
    print("genes:", len(gene_names), "| normal patients:", normal.shape[1],
          "| cancer patients:", cancer.shape[1])

    # ---- show the result for ONE gene --------------------------------------------------
    if GENE_TO_SHOW is None:
        row = 0
    else:
        row = gene_names.index(GENE_TO_SHOW)

    prob_s, bins_s, prob_r, bins_r = granular_probability_of_gene(normal[row], cancer[row])

    print("\ngene:", gene_names[row])
   
    all_probabilities = []
    for i in range(len(gene_names)):
        p_s, b_s, p_r, b_r = granular_probability_of_gene(normal[i], cancer[i])
        all_probabilities.append(p_s + p_r)          # normal patients first, then cancer

    patient_names = ([f"{NORMAL_WORD}_{k + 1}" for k in range(normal.shape[1])] +
                     [f"{CANCER_WORD}_{k + 1}" for k in range(cancer.shape[1])])
    result = pd.DataFrame(all_probabilities, columns=patient_names)
    result.insert(0, "Genes", gene_names)
    result.to_csv("Granular_probability.csv", index=False) #saved granular probability 
  