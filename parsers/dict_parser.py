import numpy as np
from typing import List, Dict, Tuple

def parser_fasta_tax_dicts(fasta_dict: Dict[str, str], tax_dict: Dict[str, Dict[str, str]], max_level):

    sequences = []
    labels = []
    ranks_to_label = {}
    label_to_ranks = {}

    for id, sequence in fasta_dict.items():

        label_vector = np.zeros(max_level) - 1

        for idx, (rank, rank_value) in enumerate(tax_dict[id].items()):
            
            if idx == max_level:
                break

            if rank not in ranks_to_label:
                ranks_to_label[rank] = {}
                label_to_ranks[rank] = {}

            rank_level_to_label = ranks_to_label[rank]
            label_level_to_rank = label_to_ranks[rank]

            if rank_value not in rank_level_to_label:
                
                label = len(rank_level_to_label)
                
                rank_level_to_label[rank_value] = label
                label_level_to_rank[label] = rank_value

            label = rank_level_to_label[rank_value]
            label_vector[idx] = label
        
        sequences.append(sequence)
        labels.append(label_vector)
    
    return sequences, labels, ranks_to_label, label_to_ranks