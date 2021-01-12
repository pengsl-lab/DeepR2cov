# -*- coding: utf-8 -*-

import numpy as np


def extract_drug():
    druginfile = open('path/drug_dict_map.txt')
    drug_dict_map = druginfile.readline().replace("\n", "").lower()
    protein_list = []
    while(drug_dict_map):
        drug_dict_map = druginfile.readline().replace("\n", "").lower()
		drug_dict[i].append(drug_dict_map[:7])
        drug_dict[i].append(drug_dict_map[8:])
    return drug_dict


def sort_index(tar):
    test_result = np.loadtxt('path/PDI_result.txt')
    Cor_mean_result = np.array(test_result).mean(axis=0)
    return Cor_mean_result

def top_rank(tar):
    Cor_top_rank = open("path/top_rank_" + tar + ".txt", "w")
    Cor_top_rank.write("No." + "\t" + "drugbank ID: drug name" + "\t\t" + "Confidence" + "\n")

    drug = extract_drug()
    drug_num = len(drug)
    Cor_result = sort_index(tar)

    if tar == 'TNF':
        Cor_result = Cor_result[:drug_num]
        Cor_result = Cor_result + np.fabs(Cor_result.min())
    elif tar == 'IL6':
        Cor_result = Cor_result[drug_num:]
        Cor_result = Cor_result + np.fabs(Cor_result.min())

    Cor_sort_index = np.argsort(Cor_result)
    num = 0
 
    for i in range(drug_num-1, -1, -1):
        num += 1
        Cor_top_rank.write(str(num).ljust(4, " ") + "\t" + drug[Cor_sort_index[i]].ljust(25, " ") +
            "\t" + str(Cor_result[Cor_sort_index[i]]) + "\n")
    Cor_top_rank.close()


if __name__ == '__main__':
    target = ['TNF', 'IL6']
    for tar in target:
       top_rank(tar)