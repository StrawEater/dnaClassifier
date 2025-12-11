import torch
import random
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.fasta_dataset import FastaDataset, pre_process_batch
from genetic_algorithm_dna import GeneticAlgorithmDNA, analyze_solution
from classifiers.dna_classifier_basic import DNAClassifier
from classifiers.dna_classifier_back_bone import CNNBackbone
from classifiers.rank_classifier import RankClassifer, RankClassiferEnd, RankClassiferCosine
from dnabert_embedder import DNABERTEmbedder
from build_classifier import get_model_config, build_model
import os

import numpy as np


def load_model(model_path, number_of_classes, device='cuda'):
    print(number_of_classes)
    """Load trained model from checkpoint"""
    
    model_configuration = get_model_config(number_of_classes)
    classifier = build_model(model_configuration)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    
    classifier.to(device)    
    classifier.eval()

    def classify(sequences):
        probs_by_rank = []

        with torch.no_grad():
            predictions = classifier(sequences)
                
            for rank_pred in predictions:
                # Get probabilities
                probs = torch.softmax(rank_pred, dim=-1).cpu().numpy()[0]
                probs_by_rank.append(probs)
            
            return probs_by_rank
    
    return classify


def plot_evolution_history(history, save_path='evolution_history.png'):
    """Plot evolution history including continuity metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Best fitness
    axes[0, 0].plot(history['best_fitness'], label='Best Fitness', linewidth=2)
    axes[0, 0].plot(history['avg_fitness'], label='Avg Fitness', alpha=0.7)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Entropy
    axes[0, 1].plot(history['best_entropy'], color='green', linewidth=2)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].set_title('Best Entropy Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Continuity score
    axes[0, 2].plot(history['best_continuity'], color='purple', linewidth=2)
    axes[0, 2].set_xlabel('Generation')
    axes[0, 2].set_ylabel('Continuity Score')
    axes[0, 2].set_title('Continuity Evolution (1.0 = fully continuous)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1.1])
    
    # N count
    axes[1, 0].plot(history['best_n_count'], color='red', linewidth=2)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Number of Ns')
    axes[1, 0].set_title('N Count Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Number of segments
    axes[1, 1].plot(history['best_num_segments'], color='orange', linewidth=2)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Number of Segments')
    axes[1, 1].set_title('Segment Count Evolution (lower = more continuous)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Pareto front (entropy vs n_count, colored by continuity)
    scatter = axes[1, 2].scatter(history['best_n_count'], history['best_entropy'], 
                                  c=history['best_continuity'], cmap='RdYlGn', 
                                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[1, 2].set_xlabel('Number of Ns')
    axes[1, 2].set_ylabel('Entropy')
    axes[1, 2].set_title('Entropy vs N Count (color = continuity)')
    axes[1, 2].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 2])
    cbar.set_label('Continuity Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEvolution history plot saved to: {save_path}")


def visualize_n_positions(sequence, individual, save_path='n_positions_viz.png'):
    """Visualize where N positions are placed in the sequence"""
    seq_len = len(sequence)
    segments = individual.get_continuous_segments()
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 8))
    
    # 1. Position map
    position_array = np.zeros(seq_len)
    for pos in individual.n_positions:
        position_array[pos] = 1
    
    axes[0].fill_between(range(seq_len), position_array, alpha=0.7, color='red')
    axes[0].set_xlabel('Sequence Position')
    axes[0].set_ylabel('N present')
    axes[0].set_title(f'N Position Map (Total: {len(individual.n_positions)} Ns)')
    axes[0].set_ylim([0, 1.2])
    axes[0].grid(True, alpha=0.3)
    
    # 2. Segment visualization
    y_positions = np.zeros(seq_len)
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    
    for idx, (start, length) in enumerate(segments):
        axes[1].barh(0, length, left=start, height=0.8, 
                     color=colors[idx], edgecolor='black', linewidth=1,
                     label=f'Seg {idx+1}: {length}bp')
    
    axes[1].set_xlim([0, seq_len])
    axes[1].set_xlabel('Sequence Position')
    axes[1].set_title(f'Continuous Segments (Total: {len(segments)} segments)')
    axes[1].set_yticks([])
    if len(segments) <= 10:  # Only show legend if not too many segments
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 3. Segment length histogram
    if segments:
        segment_lengths = [length for _, length in segments]
        axes[2].hist(segment_lengths, bins=min(20, max(segment_lengths)), 
                     color='steelblue', edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Segment Length (bp)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Segment Lengths')
        axes[2].grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {np.mean(segment_lengths):.1f} bp\nMedian: {np.median(segment_lengths):.1f} bp\nMax: {max(segment_lengths)} bp'
        axes[2].text(0.98, 0.97, stats_text, transform=axes[2].transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"N positions visualization saved to: {save_path}")


def run_ga_experiment(
    model,
    test_sequences,
    n_sequences=5,
    population_size=50,
    n_generations=100,
    max_n_positions=50,
    n_penalty_weight=0.01,
    continuity_penalty_weight=0.05,
    device='cuda'
):
    """Run GA on multiple test sequences"""
    
    results = []
    
    for idx, sequence in enumerate(test_sequences[:n_sequences]):
        print(f"\n{'='*60}")
        print(f"Processing sequence {idx+1}/{n_sequences}")
        print(f"Sequence length: {len(sequence)}")
        print(f"{'='*60}")
        
        
        
        # Analyze results
        analysis = analyze_solution(model, sequence, best_solution, device)
        
        # Plot evolution
        plot_evolution_history(history, f'evolution_seq_{idx+1}.png')
        
        # Visualize N positions
        visualize_n_positions(sequence, best_solution, f'n_positions_seq_{idx+1}.png')
        
        results.append({
            'sequence_idx': idx,
            'sequence_length': len(sequence),
            'best_solution': best_solution,
            'history': history,
            'analysis': analysis
        })
    
    return results


def main():
    """Main execution"""
    
    random.seed(128)


    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = 'models/best_model_correct.pt'  # Adjust date
    SPLITS_PATH = 'data/splits.pkl'
    
    print(f"Using device: {DEVICE}")
    
    # Load data splits
    print("\nLoading data splits...")
    with open(SPLITS_PATH, "rb") as f:
        data = pickle.load(f)
    
    X_val = data['X_val']
    y_val = data['y_val']
    ranks_to_label = data['ranks_to_label']
    
    val_dataset = FastaDataset(X_val, y_val, max_length=750)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=True, collate_fn=pre_process_batch)

    # Get number of classes per rank
    number_of_classes = [len(labels) for labels in ranks_to_label.values()]
    print(f"Number of classes per rank: {number_of_classes}")
    
    # Load model
    print("\nLoading model...")
    classifier = load_model(MODEL_PATH, number_of_classes, DEVICE)
    print("Model loaded successfully!")

    # Run GA experiments
    print("\n" + "="*60)
    print("STARTING GENETIC ALGORITHM EXPERIMENTS")
    print("="*60)
    
    # Create GA
    ga = GeneticAlgorithmDNA(
        classifier,
        [0],
        population_size = 20,
        
        max_n_count = 500,
        max_sequence_lengths = 900,
        
        n_penalty_weight = 0.01,
        discontinuity_penalty_weight = 0.05,
        
        mutation_rate = 0.3,
        crossover_rate = 0.7,
        tournament_size = 4,
        elitism_count = 3
    )
    
    # Run GA
    best_solution, history = ga.run(
        val_loader, 
        epochs = 10
    )
    
    # # Summary statistics
    # print("\n" + "="*60)
    # print("SUMMARY STATISTICS")
    # print("="*60)
    
    # entropy_increases = [r['analysis']['entropy_increase'] for r in results]
    # n_counts = [r['analysis']['n_count'] for r in results]
    # n_percentages = [r['analysis']['n_percentage'] for r in results]
    # continuity_scores = [r['analysis']['continuity_score'] for r in results]
    # num_segments = [r['analysis']['num_segments'] for r in results]
    
    # print(f"\nAverage entropy increase: {np.mean(entropy_increases):.4f} ± {np.std(entropy_increases):.4f}")
    # print(f"Average N count: {np.mean(n_counts):.1f} ± {np.std(n_counts):.1f}")
    # print(f"Average N percentage: {np.mean(n_percentages):.2f}% ± {np.std(n_percentages):.2f}%")
    # print(f"Average continuity score: {np.mean(continuity_scores):.4f} ± {np.std(continuity_scores):.4f}")
    # print(f"Average number of segments: {np.mean(num_segments):.1f} ± {np.std(num_segments):.1f}")
    
    # # Save results
    # with open('ga_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    # print("\nResults saved to ga_results.pkl")


if __name__ == "__main__":
    main()
