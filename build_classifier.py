from classifiers.dna_classifier_basic import DNAClassifier
from classifiers.dna_classifier_back_bone import CNNBackbone
from classifiers.rank_classifier import RankClassifer, RankClassiferEnd, RankClassiferCosine
from dnabert_embedder import DNABERTEmbedder
import math

def get_model_config(number_of_classes):
    """Get model configuration (same as main_parallel.py)"""
    
    backbone_config = {
        "first_channel_size": 4,
        "deepness": 3,
        "tmp_channels": 512,
    }

    embedder_config = {
        "path": "DNABERT-2-117M",
        "max_length": 750
    }

    dnaClassifier_config = {
        "tmp_channel": 768,
        "deepness": 3
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

    classifiers = build_classifiers(config_classifiers)

    return {
        "config_backbone": backbone_config,
        "config_embedder": embedder_config,
        "config_classifiers": config_classifiers,
        "dnaClassifier_config": dnaClassifier_config
    }


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