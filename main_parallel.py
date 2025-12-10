from datasets.fasta_dataset import separate_train_val_test, FastaDataset, pre_process_batch
from torch.utils.data import DataLoader, DistributedSampler
from classifiers.dna_classifier_basic import DNAClassifier
from classifiers.dna_classifier_back_bone import CNNBackbone, ResBlock
from classifiers.rank_classifier import RankClassifer, RankClassiferEnd, RankClassiferCosine
from dnabert_embedder import DNABERTEmbedder
from train_classifier import train_basic_classifier
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import math
import torch.nn as nn
import pickle
import numpy as np
from datetime import datetime

def load_splits(filename="splits.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["X_train"], data["X_val"], data["X_test"], data["y_train"], data["y_val"], data["y_test"], data["ranks_to_label"], data["label_to_ranks"]

def get_number_classes_by_rank(ranks_to_label):
    number_of_classes = []
    
    for rank, labels in ranks_to_label.items():
        number_of_classes.append(len(labels))
    
    return number_of_classes

#####################################################################

def build_classifiers(config_classifiers):
    
    classifiers = []
    
    for config_rank_classifier in config_classifiers:

        classification_end = config_rank_classifier["classification_end"]
        num_classes = config_rank_classifier["num_classes"]
        loss_weight = math.log(num_classes) * 2
        classification_in_features = config_rank_classifier["classification_in_features"]

        classifiers.append(RankClassifer(classification_in_features,
                                              classification_end,
                                              num_classes,
                                              loss_weight))

    return classifiers

def build_model(config):

    config_embedder = config["config_embedder"]
    config_backbone = config["config_backbone"]
    config_classifiers = config["config_classifiers"]
    config_dnaClassifier = config["dnaClassifier_config"]

    embedder = DNABERTEmbedder(config_embedder["path"], max_length=config_embedder["max_length"])    
    cnnBackbone = CNNBackbone(config_backbone)
    classifiers = build_classifiers(config_classifiers)

    return DNAClassifier(embedder, cnnBackbone, classifiers, config_dnaClassifier)

#################################################################


def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
   
    X_train, X_val, X_test, y_train, y_val, y_test, ranks_to_label, label_to_ranks = load_splits("data/splits_80.pkl")
    number_of_classes = get_number_classes_by_rank(ranks_to_label)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    print(f"  Classes: {number_of_classes}")

    backbone_config = {
        "first_channel_size" : 4,
        "deepness" : 3,
        "tmp_channels": 512,
    }

    embedder_config = {
        "path" : "DNABERT-2-117M",
        "max_length" : 900
    }

    config_classifiers = []
    in_features = 768
    
    for num_class in number_of_classes:
        
        if num_class > 1000: 
            classifier_end = RankClassiferCosine(in_features, num_class)
        else:
            classifier_end = RankClassiferEnd(in_features, num_class)

        rank_classifier = {
                            "classification_end" : classifier_end,
                            "num_classes" :  num_class,
                            "classification_in_features" : in_features
                          }
        
        config_classifiers.append(rank_classifier)

    dnaClassifier_config = {
        "tmp_channel" : 768,
        "deepness" : 3
    }

    model_configuration = {
        "config_backbone" : backbone_config,
        "config_embedder" : embedder_config,
        "config_classifiers" : config_classifiers,
        "dnaClassifier_config" : dnaClassifier_config
    }

    training_config = {
        "batch_size" : 64,
        "num_epochs" : 15,
        "lr" : 5e-4,
        "weight_decay": 1e-4,  # Add regularization
        "patience": 5,  # More patience for large class counts
        "label_smoothing": 0.08,  # Add for better generalization
        "gradient_clip": 1.0,  # Prevent gradient explosion
        "warmup_epochs": 2,  # Gradual LR warmup
        "lr_schedule": "cosine",  # Cosine annealing
        "accumulation_steps": 2,  # Gradient accumulation for effective batch_size=128
    }

    # Optimizer config
    optimizer_config = {
        "type": "AdamW",
        "lr": 5e-4,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "eps": 1e-8
    }   

    # Create datasets
    train_dataset = FastaDataset(X_train, y_train, max_length=embedder_config["max_length"])
    val_dataset = FastaDataset(X_val, y_val, max_length=embedder_config["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True, collate_fn=pre_process_batch)
    val_loader = DataLoader(val_dataset, batch_size=training_config["batch_size"], shuffle=False, collate_fn=pre_process_batch)

    classifier = build_model(model_configuration)
    classifier = classifier.to("cuda")

    best_val_acc, best_model_state, history, last_improved = train_basic_classifier(classifier, train_loader, val_loader, training_config, optimizer_config)
    
    date = datetime.now().strftime("%Y%m%d")
    torch.save(best_model_state, f"models/best_model_ddp_{date}.pt")

    best_model_train_metrics = history["train"][last_improved]
    best_model_val_metrics = history["val"][last_improved]

    print("\nBEST MODEL")
    print(f"  Train - Loss: {best_model_train_metrics['loss_avg']:.4f}, Rank Loss: {best_model_train_metrics['loss_rank_avg']}")
    print(f"          Top-1: {best_model_train_metrics['top1_acc']:.4f}, Top-5: {best_model_train_metrics['top5_acc']:.4f}")
    print(f"          Rank Top-1: {best_model_train_metrics['top1_rank_acc']}, Rank Top-5: {best_model_train_metrics['top5_rank_acc']}")
    print(f"  Val   - Loss: {best_model_val_metrics['loss_avg']:.4f}, Rank Loss: {best_model_val_metrics['loss_rank_avg']}")
    print(f"          Top-1: {best_model_val_metrics['top1_acc']:.4f}, Top-5: {best_model_val_metrics['top5_acc']:.4f}")
    print(f"          Rank Top-1: {best_model_val_metrics['top1_rank_acc']}, Rank Top-5: {best_model_val_metrics['top5_rank_acc']}")

if __name__ == "__main__":
    main()