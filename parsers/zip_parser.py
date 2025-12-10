import os
import zipfile
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path
from .fasta_parser import parse_fasta_from_string

def add_fasta_sequences_to_dataset(zip_file, fasta_file, sequences, labels, rank_to_label, label_to_rank):
   # Get phylum name from filename
    rank_name = Path(fasta_file).stem
        
    # Assign label to phylum if not already assigned
    if rank_name not in rank_to_label:
      label = len(rank_to_label)
      rank_to_label[rank_name] = label
      label_to_rank[label] = rank_name
        
    label = rank_to_label[rank_name]
        
    # Read file content directly from zip
    with zip_file.open(fasta_file) as fasta_file:

      content = fasta_file.read().decode('utf-8')
      phylum_sequences = parse_fasta_from_string(content)  
      
      sequences.extend(phylum_sequences)
      labels.extend([label] * len(phylum_sequences))

def load_data_from_zip(zip_path: str):

  sequences = []
  labels = []
  phylum_to_label = {}
  label_to_phylum = {}

  # Leemos el zip file con los .fasta
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    
    # Get list of all files in zip
    fasta_files = zip_ref.namelist()
    
    print(f"Found {len(fasta_files)} FASTA files")
    
    # Process each FASTA file directly from zip
    for fasta_filename in tqdm(fasta_files, desc="Loading FASTA files"):
        add_fasta_sequences_to_dataset(zip_ref,
                                       fasta_filename,
                                       sequences,
                                       labels,
                                       phylum_to_label,
                                       label_to_phylum
                                      )
  
  return sequences, labels, phylum_to_label, label_to_phylum