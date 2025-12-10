import torch
import torch.nn as nn
from torch.nn import functional as F
from classifiers.dna_classifier_back_bone import ResBlock

def gram_schmidt_component(vector, basis):
    """
    Get the orthogonal component of 'vector' relative to 'basis'
    """
    # Normalize basis
    basis_norm = F.normalize(basis, p=2, dim=-1)
    
    # Project vector onto basis
    projection = (vector * basis_norm).sum(dim=-1, keepdim=True) * basis_norm
    
    # Orthogonal component = vector - projection
    orthogonal = vector - projection
    
    return orthogonal

class DNAClassifier(nn.Module):
    def __init__(self, embedder, backbone, rank_classifiers, config):
        super().__init__()
        
        self.embedder = embedder
        self.backbone = backbone
        self.rank_classifiers = rank_classifiers

        tmp_channel = config["tmp_channel"]
        deepness = config["deepness"]
        embbeder_size = embedder.get_embedding_dim()

        backbone_length = backbone.get_backbone_length(embbeder_size)
        backbone_channels = backbone.get_backbone_channels()
        cant_features = int(backbone_length * backbone_channels)

        #Pre procesamiento propio de cada rank
        self.rank_classifiers_pre_process = nn.ModuleList()
        for rank_classifier in rank_classifiers:
            pre_process = nn.ModuleList()
            
            for i in range(deepness):
                pre_process.append(ResBlock(backbone_channels, tmp_channel))
            
            pre_process.append(nn.Flatten())
            self.rank_classifiers_pre_process.append(nn.Sequential(*pre_process)) 


        #Clasificacion propia de cada rank
        self.rank_classifiers_end = nn.ModuleList()
        for rank_classifier in rank_classifiers:
            end = nn.Sequential(
                                nn.Linear(cant_features, rank_classifier.classification_in_features),
                                rank_classifier.classification_end
                               )
            
            self.rank_classifiers_end.append(end)
            

    def forward(self, sequences):
        
        embeddings = self.embedder(sequences)
        features = self.backbone(embeddings)
        
        pre_processing = []
        for rank_preprocess in self.rank_classifiers_pre_process:
            pre_processing.append(rank_preprocess(features))

        # Add orthogonal amplification
        amplified_features = []
        for idx in range(len(pre_processing)):
            if idx == 0:
                # First rank - no previous to compare
                amplified_features.append(pre_processing[idx])
            else:
                # Get orthogonal component relative to previous rank
                ortho_component = gram_schmidt_component(
                    pre_processing[idx], 
                    pre_processing[idx - 1]
                )
                
                # Amplify the difference
                amplification_scale = 0.5  # tune this hyperparameter
                amplified = pre_processing[idx] + amplification_scale * ortho_component
                amplified_features.append(amplified)
        
        # Use amplified features for predictions
        predictions = []
        for idx, rank_end in enumerate(self.rank_classifiers_end):
            predictions.append(rank_end(amplified_features[idx]))
        
        return predictions

    def get_loss_weight(self, rank_index):
        return self.rank_classifiers[rank_index].loss_weight
    
    def get_ranks(self):
        return len(self.rank_classifiers)

    def get_dynamic_loss_weights(self, current_epoch, total_epochs):
        """
        Linear shift from lower to higher ranks
        """
        num_ranks = len(self.rank_classifiers_end)
        progress = min(current_epoch / total_epochs, 1.0)
        
        def cuadratic_f(x, offset, roots):
            x -= offset
            result =  1 - (x/roots) * (x/roots) 
            return max(0, result)
        
        # var = (1 - progress) + 0.001
        var = 1
        offset = min(1 , (progress * 1.3))

        weights = []
        for rank_idx in range(num_ranks):

            norm_rank = rank_idx / (num_ranks - 1) # 0 a 1
            coeff = cuadratic_f(norm_rank, offset, var)
            coeff = max(0.2, coeff)
            
            base_weight = self.get_loss_weight(rank_idx)

            weights.append(base_weight * coeff)

        return weights