
from parsers.zip_parser import load_data_from_zip
from parsers.fasta_parser import get_fasta_dict_from_file
from parsers.tax_parser import get_tax_dict_from_file
from parsers.dict_parser import parser_fasta_tax_dicts
from parsers.dict_parser import parser_fasta_tax_dicts
from datasets.fasta_dataset import separate_train_val_test, FastaDataset, augment_sequences_with_ambiguity
import numpy as np
import pickle

def main():
    
    fasta_file = "data/representantes_80.fasta"
    tax_file = "data/MIDORI2_UNIQ_NUC_GB268_CO1.taxon"

    print("Procesando Archivo Fasta...")
    fasta_dict = get_fasta_dict_from_file(fasta_file)
    
    print("Procesando Archivo Tax...")
    tax_dict = get_tax_dict_from_file(tax_file)

    print("Procesando Diccionarios creados...")
    sequences, labels, ranks_to_label, label_to_ranks = parser_fasta_tax_dicts(fasta_dict, tax_dict, 4)
    
    print("Agregando ambiguiedades a las secuencias..")
    sequences, labels = augment_sequences_with_ambiguity(sequences, labels)

    print("Separando conjuntos")
    X_train, X_val, X_test, y_train, y_val, y_test = separate_train_val_test(
                                                                            sequences,
                                                                            labels,
                                                                            test_size=0.2,
                                                                            val_size=0.1
                                                                            )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "ranks_to_label": ranks_to_label, 
        "label_to_ranks": label_to_ranks
    }

    with open("splits_80.pkl", "wb") as f:
        pickle.dump(data, f)

    
if __name__ == "__main__":
    main()
