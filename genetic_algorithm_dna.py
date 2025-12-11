import torch
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import random
from scipy.stats import entropy
from copy import deepcopy
from dna_sequence_perturber import DNASequencePerturber
from dna_sequence_perturber_mutators import *


class GeneticAlgorithmDNA:
    """Genetic Algorithm to find optimal N positions for maximizing prediction entropy"""
    
    def __init__(
        self,
        classifier,
        ranks_for_entropy,
        
        population_size: int = 50,
        
        max_n_count: int = 50,
        max_sequence_lengths: int = 900,
        
        n_penalty_weight: float = 0.01,
        discontinuity_penalty_weight: float = 0.05,
        
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        elitism_count: int = 5
    ):
        """
        Args:
            model: Trained DNA classifier model
            population_size: Number of individuals in population
            max_n_count: Maximum number of N positions allowed
            n_penalty_weight: Weight for penalizing number of Ns
            continuity_penalty_weight: Weight for penalizing discontinuous segments
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
            elitism_count: Number of best individuals to preserve
        """
        self.classifier = classifier
        self.ranks_for_entropy = ranks_for_entropy
        self.population_size = population_size
        self.max_n_count = max_n_count
        self.max_sequence_lengths = max_sequence_lengths
        self.n_penalty_weight = n_penalty_weight
        self.discontinuity_penalty_weight = discontinuity_penalty_weight
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
    
    def calculate_prediction_entropy(self, sequences: List[str]) -> float:
        """Calculate entropy of model predictions for a sequence"""
            
        predictions = self.classifier(sequences)

        entropies = []
        
        for rank_idx in self.ranks_for_entropy:

            rank_probs = predictions[rank_idx]

            # Calculate entropy
            rank_entropy = entropy(rank_probs)
            entropies.append(rank_entropy)
        
        # Average entropy across all ranks
        avg_entropy = np.mean(entropies)
            
        return avg_entropy
    
    def evaluate_fitness(self, individual: DNASequencePerturber, original_sequences: List[str]) -> float:
        
        # Apply N positions to sequence

        entropies = []

        for sequence in original_sequences:

            perturbed_seq = individual.apply_to_sequence([sequence])
            entropy_score = self.calculate_prediction_entropy(perturbed_seq)
            entropies.append(entropy_score)

        mean_entropy = np.mean(entropies)
        
        # Penalize number of Ns
        ns_score = len(individual.n_positions) / self.max_n_count
        ns_penalty = self.n_penalty_weight * ns_score
        
        # Penalize discontinuity (more segments = higher penalty)
        continuity_score = individual.calculate_continuity_score()
        discontinuity_penalty = self.discontinuity_penalty_weight * (1 - continuity_score)
        
        # Fitness is entropy minus penalties
        fitness = entropy_score - ns_penalty - discontinuity_penalty
        
        # Store components for analysis
        individual.entropy_score = entropy_score
        individual.n_count_penalty = ns_penalty
        individual.continuity_penalty = discontinuity_penalty
        individual.fitness = fitness
        
        return fitness
    
    def initialize_population(self) -> List[DNASequencePerturber]:
        """Initialize random population with preference for continuous segments"""
        population = []
        
        for i in range(self.population_size):
            
            if i < self.population_size // 2:
                n_positions = generate_random_continuous_positions(self.max_sequence_lengths//10, self.max_sequence_lengths)
            
            else:
                n_positions = generate_few_segments(self.max_n_count, self.max_sequence_lengths, 80)
            
            individual = DNASequencePerturber(self.max_sequence_lengths, n_positions)
            population.append(individual)
        
        return population
    
    def tournament_selection(self, population: List[DNASequencePerturber]) -> DNASequencePerturber:
        """Select individual using tournament selection"""
        tournament = random.sample(population, self.tournament_size)
        winner = max(tournament, key=lambda ind: ind.fitness)
        return winner.copy()
    
    def crossover(
        self, 
        parent1: DNASequencePerturber, 
        parent2: DNASequencePerturber
    ) -> Tuple[DNASequencePerturber, DNASequencePerturber]:
        
        chance = random.random() 
        """Perform crossover between two parents - segment-aware"""
        if chance > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Get segments from both parents
        segments1 = parent1.get_continuous_segments()
        segments2 = parent2.get_continuous_segments()
        
        if not segments1 and not segments2:
            return parent1.copy(), parent2.copy()
        
        # Randomly select segments from each parent
        child1_positions = []
        child2_positions = []
        
        all_segments = segments1 + segments2
        random.shuffle(all_segments)
        
        # Split segments between children
        for i, (start, length) in enumerate(all_segments):
            segment_positions = generate_continues_positions(start, length)
            
            if i % 2 == 0:
                child1_positions.extend(segment_positions)
            else:
                child2_positions.extend(segment_positions)
        
        child1_positions = remove_duplicates_and_fix_size(child1_positions, self.max_n_count)
        child2_positions = remove_duplicates_and_fix_size(child2_positions, self.max_n_count)
        
        child1 = DNASequencePerturber(parent1.sequence_length, child1_positions)
        child2 = DNASequencePerturber(parent2.sequence_length, child2_positions)
        
        return child1, child2
    
    def mutate(self, individual: DNASequencePerturber) -> DNASequencePerturber:
        
        """Mutate an individual - continuity-aware mutations"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.copy()
        # Choose mutation type with bias toward continuity-preserving mutations
        mutation_types = ['add_continuous','remove_segment', 'extend_segment','merge_segments','split_segment','shift_segment','shift_segment']

        mutation_type = random.choice(mutation_types)
        
        if mutation_type == 'add_continuous':
            add_continuous(mutated, self.max_n_count, 10)

        elif mutation_type == 'remove_segment':
            remove_segment(mutated)

        elif mutation_type == 'extend_segment':
            extend_segment(mutated, self.max_n_count, 10)

        elif mutation_type == 'merge_segments':
            merge_segments(mutated, self.max_n_count, 15)
        
        elif mutation_type == 'split_segment':
            split_segment(mutated, 40, 20)
        
        elif mutation_type == 'shift_segment':
            shift_segment(mutated, self.max_n_count, 10)

        else:
            assert False, "Hubo un problemas con las mutaciones"
        
        return mutated

    def evolve_generation(
        self, 
        population: List[DNASequencePerturber],
        original_sequences: List[str]
    ) -> List[DNASequencePerturber]:
        """Evolve one generation"""
        
        # Evaluate fitness for all individuals
        for ind in population:
            if ind.fitness is None:
                self.evaluate_fitness(ind, original_sequences)
        
        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Elitism: keep best individuals
        new_population = [ind.copy() for ind in population[:self.elitism_count]]
        
        # Generate rest of population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # Add to new population
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:self.population_size]
    
    def run(
        self, 
        dataloader, 
        epochs: int = 10,
        verbose: bool = True
    ) -> Tuple[DNASequencePerturber, Dict]:
        """
        Run the genetic algorithm
        
        Returns:
            best_individual: Best solution found
            history: Dictionary with evolution history
        """
        
        # Initialize population
        population = self.initialize_population()
        
        # History tracking
        history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_entropy': [],
            'best_n_count': [],
            'best_continuity': [],
            'best_num_segments': []
        }
        
        for epoch in range(epochs):

            pbar = tqdm(dataloader, desc="Evolution")
            for sequences, _ in pbar:
                population = self.evolve_generation(population, sequences)
            

            best_ind = max(population, key=lambda ind: ind.fitness)
            avg_fitness = np.mean([ind.fitness for ind in population])

            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print("\nGenetic Algorithm Status")
            print(f"  Best Fitness:              {best_ind.fitness:.4f}")
            print(f"  Average Fitness:           {avg_fitness:.4f}")
            print(f"  Best Entropy Score:        {best_ind.entropy_score:.4f}")
            print(f"  Best N Count:              {len(best_ind.n_positions)}")
            print(f"  Best N Count Penalty:      {best_ind.n_count_penalty:.4f}")
            print(f"  Best Continuity Penalty:   {best_ind.continuity_score:.4f}")

            history['best_fitness'].append(best_ind.fitness)
            history['avg_fitness'].append(avg_fitness)
            history['best_entropy'].append(best_ind.entropy_score)
            history['best_n_count'].append(len(best_ind.n_positions))
            history['best_n_count_penalty'].append(best_ind.n_count_penalty)
            history['best_continuity_penalty'].append(best_ind.continuity_score)
            
        # Return best individual
        best_individual = max(population, key=lambda ind: ind.fitness)
        return best_individual, history


def analyze_solution(
    model,
    original_sequence: str,
    best_individual: DNASequencePerturber,
):
    """Analyze the best solution found"""
    print("\n" + "="*60)
    print("SOLUTION ANALYSIS")
    print("="*60)
    
    # Original sequence predictions
    model.eval()
    with torch.no_grad():
        orig_preds = model([original_sequence])
        orig_entropies = []
        
        for rank_pred in orig_preds:
            probs = torch.softmax(rank_pred, dim=-1).cpu().numpy()[0]
            orig_entropies.append(entropy(probs))
    
    # Perturbed sequence
    perturbed_seq = best_individual.apply_to_sequence(original_sequence)
    
    with torch.no_grad():
        pert_preds = model([perturbed_seq])
        pert_entropies = []
        
        for rank_pred in pert_preds:
            probs = torch.softmax(rank_pred, dim=-1).cpu().numpy()[0]
            pert_entropies.append(entropy(probs))
    
    # Continuity analysis
    segments = best_individual.get_continuous_segments()
    continuity_score = best_individual.calculate_continuity_score()
    
    print(f"\n{'='*60}")
    print("N POSITIONS ANALYSIS")
    print(f"{'='*60}")
    print(f"Number of N positions: {len(best_individual.n_positions)}")
    print(f"Percentage of sequence: {100*len(best_individual.n_positions)/len(original_sequence):.2f}%")
    print(f"\nContinuity Score: {continuity_score:.4f} (1.0 = fully continuous)")
    print(f"Number of segments: {len(segments)}")
    
    print(f"\nSegment details:")
    for i, (start, length) in enumerate(segments):
        print(f"  Segment {i+1}: position {start}-{start+length-1} (length={length})")
    
    print(f"\n{'='*60}")
    print("ENTROPY ANALYSIS")
    print(f"{'='*60}")
    print(f"Original sequence entropy (avg): {np.mean(orig_entropies):.4f}")
    print(f"Perturbed sequence entropy (avg): {np.mean(pert_entropies):.4f}")
    print(f"Entropy increase: {np.mean(pert_entropies) - np.mean(orig_entropies):.4f}")
    
    print(f"\nEntropy by rank:")
    for i, (orig_e, pert_e) in enumerate(zip(orig_entropies, pert_entropies)):
        print(f"  Rank {i}: {orig_e:.4f} -> {pert_e:.4f} (Δ={pert_e-orig_e:.4f})")
    
    print(f"\n{'='*60}")
    print("FITNESS COMPONENTS")
    print(f"{'='*60}")
    print(f"Entropy score: {best_individual.entropy_score:.4f}")
    print(f"N count penalty: {best_individual.n_count_penalty:.4f}")
    print(f"Continuity penalty: {best_individual.continuity_penalty:.4f}")
    print(f"Total fitness: {best_individual.fitness:.4f}")
    
    return {
        'original_entropy': np.mean(orig_entropies),
        'perturbed_entropy': np.mean(pert_entropies),
        'entropy_increase': np.mean(pert_entropies) - np.mean(orig_entropies),
        'n_count': len(best_individual.n_positions),
        'n_percentage': 100*len(best_individual.n_positions)/len(original_sequence),
        'continuity_score': continuity_score,
        'num_segments': len(segments),
        'segments': segments
    }