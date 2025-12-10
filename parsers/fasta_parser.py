from typing import List, Dict, Tuple


def parse_fasta_from_string(fasta_string) -> List[str]:
    """Parse a FASTA file and return list of sequences"""
    sequences = []
    current_seq = []
    
    for line in fasta_string.split('\n'):
      line = line.strip()
      if line.startswith('>'):
          # New sequence header
          if current_seq:
              sequences.append(''.join(current_seq))
              current_seq = []
      else:
          # Sequence data
          current_seq.append(line.upper())
  
    # Don't forget the last sequence
    if current_seq:
      sequences.append(''.join(current_seq))

    return sequences

def get_fasta_dict_from_file(fasta_path: str) -> Dict[str,str]:
    """Parse a FASTA file and return list of sequences."""
    
    sequences = {}
    
    current_id = ""
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                
                if current_seq:
                    sequences[current_id] = (''.join(current_seq))

                current_id = line[1:]  # remove ">"
                current_id = current_id.split(';')[0]
                current_seq = []
                
            else:

                current_seq.append(line.upper())

    # Append the last sequence
    if current_seq:
        sequences[current_id] = (''.join(current_seq))

    return sequences