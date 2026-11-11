
import math
import numpy as np
import pandas as pd
from granular_probability_simple import read_data, granular_probability_of_gene


# =========================================================================================
# STEP 0 - SETTINGS (change only this part)
# =========================================================================================
INPUT_FILE = "oral_cancer_expression.csv"  
GENE_COLUMN = "Genes"                 
NORMAL_WORD = "normal"            
CANCER_WORD = "cancer"           
TOP_PERCENT = 25                   #percentage of genes to be selected


# =========================================================================================
# STEP 1 - Computation of fuzzy membership value of an expression value
# =========================================================================================
def fuzzy_membership(value, mean_normal, mean_cancer):
    distance_s = (value - mean_normal) ** 2
    distance_r = (value - mean_cancer) ** 2

    if distance_s == 0 and distance_r == 0:      # both averages are equal to the value
        return 0.5, 0.5
    if distance_s == 0:                         
        return 1.0, 0.0
    if distance_r == 0:                          
        return 0.0, 1.0

    member_s = (1 / distance_s) / (1 / distance_s + 1 / distance_r)
    member_r = 1 - member_s
    return member_s, member_r


# =========================================================================================
# STEP 2 - Calculation of RELATIVE FREQUENCY in CONFIRMED ZONE AND OVERLAPPING ZONE
# =========================================================================================
def relative_frequency(normal_values, cancer_values):
    mean_s = np.mean(normal_values)
    mean_r = np.mean(cancer_values)

    member_of_normal_patients = []     # membership in normal group, normal patients
    for value in normal_values:
        member_s, member_r = fuzzy_membership(value, mean_s, mean_r)
        member_of_normal_patients.append(member_s)

    member_of_cancer_patients = []     # membership in cancer group, cancer patients
    for value in cancer_values:
        member_s, member_r = fuzzy_membership(value, mean_s, mean_r)
        member_of_cancer_patients.append(member_r)

    freq_s = np.mean(member_of_normal_patients)
    freq_r = np.mean(member_of_cancer_patients)

    confirmed = (freq_s + freq_r) / 2
    overlapping = 1 - confirmed
    return freq_s, freq_r, confirmed, overlapping, member_of_normal_patients, member_of_cancer_patients


# =========================================================================================
# STEP 3 - ENTROPY value for gene ranking
# =========================================================================================
def entropy_of(confirmed, overlapping):
    total = 0.0
    for part in (confirmed, overlapping):
        if part > 0:
            total = total - part * math.log2(part)
    return total


# =========================================================================================
# STEP 4 - Confidence of the selected gene
# =========================================================================================
def confidence_of_group(members, bins, probabilities, group_average):
    total = 0.0
    for b in set(bins):                                              # every bin once
        members_in_bin = [members[k] for k in range(len(bins)) if bins[k] == b]
        strongest = max(members_in_bin)
        if strongest >= group_average - 1e-12:                       # a good bin
            first_patient = bins.index(b)
            total = total + probabilities[first_patient]             # probability of this bin
    return total


# =========================================================================================
# STEP 5 - Ranking of gene
# =========================================================================================
def spem_of_one_gene(normal_values, cancer_values):
    (freq_s, freq_r, confirmed, overlapping,
     members_s, members_r) = relative_frequency(normal_values, cancer_values)

    entropy = entropy_of(confirmed, overlapping)

    prob_s, bins_s, prob_r, bins_r = granular_probability_of_gene(normal_values, cancer_values)
    confidence_s = confidence_of_group(members_s, bins_s, prob_s, freq_s)
    confidence_r = confidence_of_group(members_r, bins_r, prob_r, freq_r)
    confidence = (confidence_s + confidence_r) / 2

    return entropy, confirmed, overlapping, confidence


if __name__ == "__main__":

    # ---- read the data -------------------------------------------------------------------
    gene_names, normal, cancer = read_data(INPUT_FILE, GENE_COLUMN, NORMAL_WORD, CANCER_WORD)
    print("genes:", len(gene_names), "| normal patients:", normal.shape[1],
          "| cancer patients:", cancer.shape[1])

    # ---- run SPEM on every gene ---------------------------------------------------------
    rows = []
    for i in range(len(gene_names)):
        if np.isnan(normal[i]).any() or np.isnan(cancer[i]).any():
            continue                                   # skip genes with missing values
        entropy, confirmed, overlapping, confidence = spem_of_one_gene(normal[i], cancer[i])
        rows.append([gene_names[i], entropy, confidence])
        
    
    result = pd.DataFrame(rows, columns=["Genes", "entropy", "confidence"])


    # ---- rank: smallest entropy first (if equal, the higher confidence first) ------------
    result = result.sort_values(["entropy", "confidence"], ascending=[True, False])
    result.insert(0, "rank", range(1, len(result) + 1))
    result = result.reset_index(drop=True)

    # ---- keep the top portion (25 percent of 20 genes = 5 genes) -----------------------
    selected_genes = max(1, math.floor(len(result) * TOP_PERCENT / 100 + 0.5))
    selected = result.head(selected_genes)

    result.to_csv("SPEM_ranking_all_genes.csv", index=False)
    selected.to_csv("SPEM_selected_genes.csv", index=False)

    print("\nTop", selected_genes, "genes:")
    print(selected.to_string(index=False))