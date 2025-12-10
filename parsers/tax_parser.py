
from typing import List, Dict, Tuple

# Taxonomic ranks to extract from taxonomy string
RANK_PREFIXES = {
    "p__": "Phylum",
    "c__": "Class",
    "o__": "Order",
    "f__": "Family",
    "g__": "Genus",
    "s__": "Species",
}

def parse_taxonomy(tax_string: str) -> Dict[str, str]: 
    """
    tax_string example:
    k__Eukaryota_2759;p__Discosea_555280;c__Flabellinia_1485085;...
    returns:
    { "Phylum": "Discosea", "Class": "Flabellinia", ... }
    """

    results = {}
    parts = tax_string.split(";")

    for part in parts:
        for prefix, rank_name in RANK_PREFIXES.items():
            if part.startswith(prefix):
                # remove prefix, keep only taxon name without numeric suffix
                taxon_full = part.split("__")[1]
                taxon_name = taxon_full.split("_")[0]  # remove NCBI taxid
                results[rank_name] = taxon_name

    return results


def get_tax_dict_from_file(tax_path: str) -> Dict[str, Dict[str, str]]:
    
    taxonomy = {}
    with open(tax_path) as tf:
        for line in tf:
            line = line.strip()
            if not line:
                continue
            seq_id, taxa = line.split("\t", 1)
            taxonomy[seq_id] = parse_taxonomy(taxa)
    
    return taxonomy