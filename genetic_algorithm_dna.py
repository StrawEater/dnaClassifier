import torch
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import random
from scipy.stats import entropy
from copy import deepcopy


class DNASequenceIndividual:
    """Represents a solution: positions where to place 'N' in a DNA sequence"""
    
    def __init__(self, sequence_length: int, n_positions: List[int] = None):
        self.sequence_length = sequence_length
        self.n_positions = n_positions if n_positions is not None else []
        self.fitness = None
        self.entropy_score = None
        self.n_count_penalty = None
        self.continuity_penalty = None
    
    def apply_to_sequence(self, original_sequence: str) -> str:
        """Apply the N positions to create a perturbed sequence"""
        seq_list = list(original_sequence)
        for pos in self.n_positions:
            if pos < len(seq_list):
                seq_list[pos] = 'N'
        return ''.join(seq_list)
    
    def get_continuous_segments(self) -> List[Tuple[int, int]]:
        """
        Get continuous segments of N positions
        Returns list of (start, length) tuples
        """
        if not self.n_positions:
            return []
        
        segments = []
        current_start = self.n_positions[0]
        current_length = 1
        
        for i in range(1, len(self.n_positions)):
            if self.n_positions[i] == self.n_positions[i-1] + 1:
                # Continuous
                current_length += 1
            else:
                # Break in continuity
                segments.append((current_start, current_length))
                current_start = self.n_positions[i]
                current_length = 1
        
        # Don't forget last segment
        segments.append((current_start, current_length))
        
        return segments
    
    def calculate_continuity_score(self) -> float:
        """
        Calculate continuity score - rewards continuous segments
        Returns a value between 0 (all isolated) and 1 (all continuous)
        """
        if len(self.n_positions) <= 1:
            return 1.0
        
        segments = self.get_continuous_segments()
        
        # Ideal case: all Ns in one segment
        # Worst case: all Ns isolated (n_positions segments)
        num_segments = len(segments)
        max_segments = len(self.n_positions)  # Worst case
        min_segments = 1  # Best case
        
        # Normalize: 0 = worst (many segments), 1 = best (few segments)
        if max_segments == min_segments:
            return 1.0
        
        continuity = 1.0 - (num_segments - min_segments) / (max_segments - min_segments)
        return continuity
    
    def calculate_discontinuity_penalty(self) -> float:
        """
        Calculate penalty for discontinuous N positions
        Penalty increases with number of separate segments
        """
        if len(self.n_positions) <= 1:
            return 0.0
        
        segments = self.get_continuous_segments()
        num_segments = len(segments)
        
        # Penalty is proportional to number of segments
        # More segments = more penalty
        penalty = num_segments - 1  # 0 if all continuous (1 segment)
        
        return penalty
    
    def copy(self):
        """Create a deep copy of the individual"""
        new_ind = DNASequenceIndividual(self.sequence_length, self.n_positions.copy())
        new_ind.fitness = self.fitness
        new_ind.entropy_score = self.entropy_score
        new_ind.n_count_penalty = self.n_count_penalty
        new_ind.continuity_penalty = self.continuity_penalty
        return new_ind


class GeneticAlgorithmDNA:
    """Genetic Algorithm to find optimal N positions for maximizing prediction entropy"""
    
    def __init__(
        self,
        model,
        device: str = 'cuda',
        population_size: int = 50,
        max_n_positions: int = 50,
        n_penalty_weight: float = 0.01,
        continuity_penalty_weight: float = 0.05,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        elitism_count: int = 5
    ):
        """
        Args:
            model: Trained DNA classifier model
            device: Device to run model on
            population_size: Number of individuals in population
            max_n_positions: Maximum number of N positions allowed
            n_penalty_weight: Weight for penalizing number of Ns
            continuity_penalty_weight: Weight for penalizing discontinuous segments
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament for selection
            elitism_count: Number of best individuals to preserve
        """
        self.model = model
        self.device = device
        self.population_size = population_size
        self.max_n_positions = max_n_positions
        self.n_penalty_weight = n_penalty_weight
        self.continuity_penalty_weight = continuity_penalty_weight
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        self.model.eval()
    
    def calculate_prediction_entropy(self, sequence: str) -> Tuple[float, List[np.ndarray]]:
        """Calculate entropy of model predictions for a sequence"""
        with torch.no_grad():
            predictions = self.model([sequence])
            
            entropies = []
            all_probs = []
            
            for rank_pred in predictions:
                # Get probabilities
                probs = torch.softmax(rank_pred, dim=-1).cpu().numpy()[0]
                all_probs.append(probs)
                
                # Calculate entropy
                rank_entropy = entropy(probs)
                entropies.append(rank_entropy)
            
            # Average entropy across all ranks
            avg_entropy = np.mean(entropies)
            
        return avg_entropy, all_probs
    
    def evaluate_fitness(self, individual: DNASequenceIndividual, original_sequence: str) -> float:
        """
        Evaluate fitness of an individual
        Fitness = entropy_score - n_penalty_weight * n_count - continuity_penalty_weight * discontinuity_penalty
        
        The continuity penalty penalizes having many separate N segments
        """
        # Apply N positions to sequence
        perturbed_seq = individual.apply_to_sequence(original_sequence)
        
        # Calculate entropy
        entropy_score, _ = self.calculate_prediction_entropy(perturbed_seq)
        
        # Penalize number of Ns
        n_count = len(individual.n_positions)
        n_penalty = self.n_penalty_weight * n_count
        
        # Penalize discontinuity (more segments = higher penalty)
        discontinuity_penalty = individual.calculate_discontinuity_penalty()
        continuity_penalty = self.continuity_penalty_weight * discontinuity_penalty
        
        # Fitness is entropy minus penalties
        fitness = entropy_score - n_penalty - continuity_penalty
        
        # Store components for analysis
        individual.entropy_score = entropy_score
        individual.n_count_penalty = n_penalty
        individual.continuity_penalty = continuity_penalty
        individual.fitness = fitness
        
        return fitness
    
    def initialize_population(self, sequence_length: int) -> List[DNASequenceIndividual]:
        """Initialize random population with preference for continuous segments"""
        population = []
        
        for i in range(self.population_size):
            # Mix of strategies
            if i < self.population_size // 3:
                # Random scattered positions
                n_count = random.randint(0, self.max_n_positions)
                n_positions = sorted(random.sample(range(sequence_length), n_count))
            
            elif i < 2 * self.population_size // 3:
                # Continuous segments
                n_positions = self._generate_continuous_positions(sequence_length)
            
            else:
                # Few continuous segments (2-5 segments)
                n_positions = self._generate_few_segments(sequence_length)
            
            individual = DNASequenceIndividual(sequence_length, n_positions)
            population.append(individual)
        
        return population
    
    def _generate_continuous_positions(self, sequence_length: int) -> List[int]:
        """Generate a single continuous segment of Ns"""
        segment_length = random.randint(1, min(self.max_n_positions, sequence_length // 4))
        start_pos = random.randint(0, sequence_length - segment_length)
        return list(range(start_pos, start_pos + segment_length))
    
    def _generate_few_segments(self, sequence_length: int) -> List[int]:
        """Generate 2-5 continuous segments"""
        num_segments = random.randint(2, 5)
        positions = []
        
        remaining_n = self.max_n_positions
        
        for _ in range(num_segments):
            if remaining_n <= 0:
                break
            
            segment_length = random.randint(1, max(1, remaining_n // (num_segments + 1)))
            start_pos = random.randint(0, max(0, sequence_length - segment_length - 1))
            
            segment = list(range(start_pos, start_pos + segment_length))
            positions.extend(segment)
            remaining_n -= segment_length
        
        # Remove duplicates and sort
        positions = sorted(list(set(positions)))[:self.max_n_positions]
        return positions
    
    def tournament_selection(self, population: List[DNASequenceIndividual]) -> DNASequenceIndividual:
        """Select individual using tournament selection"""
        tournament = random.sample(population, self.tournament_size)
        winner = max(tournament, key=lambda ind: ind.fitness)
        return winner.copy()
    
    def crossover(
        self, 
        parent1: DNASequenceIndividual, 
        parent2: DNASequenceIndividual
    ) -> Tuple[DNASequenceIndividual, DNASequenceIndividual]:
        """Perform crossover between two parents - segment-aware"""
        if random.random() > self.crossover_rate:
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
            segment_positions = list(range(start, start + length))
            
            if i % 2 == 0:
                child1_positions.extend(segment_positions)
            else:
                child2_positions.extend(segment_positions)
        
        # Limit and sort
        child1_positions = sorted(list(set(child1_positions)))[:self.max_n_positions]
        child2_positions = sorted(list(set(child2_positions)))[:self.max_n_positions]
        
        child1 = DNASequenceIndividual(parent1.sequence_length, child1_positions)
        child2 = DNASequenceIndividual(parent2.sequence_length, child2_positions)
        
        return child1, child2
    
    def mutate(self, individual: DNASequenceIndividual) -> DNASequenceIndividual:
        """Mutate an individual - continuity-aware mutations"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.copy()
        
        # Choose mutation type with bias toward continuity-preserving mutations
        mutation_types = ['add_continuous', 'remove_segment', 'extend_segment', 
                         'merge_segments', 'split_segment', 'shift_segment']
        mutation_type = random.choice(mutation_types)
        
        segments = mutated.get_continuous_segments()
        
        if mutation_type == 'add_continuous' and len(mutated.n_positions) < self.max_n_positions:
            # Add a new continuous segment
            segment_length = random.randint(1, min(5, self.max_n_positions - len(mutated.n_positions)))
            start_pos = random.randint(0, mutated.sequence_length - segment_length)
            new_positions = list(range(start_pos, start_pos + segment_length))
            mutated.n_positions.extend(new_positions)
            mutated.n_positions = sorted(list(set(mutated.n_positions)))[:self.max_n_positions]
        
        elif mutation_type == 'remove_segment' and segments:
            # Remove an entire segment
            segment_to_remove = random.choice(segments)
            start, length = segment_to_remove
            positions_to_remove = set(range(start, start + length))
            mutated.n_positions = [p for p in mutated.n_positions if p not in positions_to_remove]
        
        elif mutation_type == 'extend_segment' and segments and len(mutated.n_positions) < self.max_n_positions:
            # Extend a random segment by 1-3 positions
            segment = random.choice(segments)
            start, length = segment
            
            # Extend left or right
            if random.random() < 0.5 and start > 0:
                mutated.n_positions.append(start - 1)
            else:
                end = start + length
                if end < mutated.sequence_length:
                    mutated.n_positions.append(end)
            
            mutated.n_positions = sorted(list(set(mutated.n_positions)))[:self.max_n_positions]
        
        elif mutation_type == 'merge_segments' and len(segments) >= 2:
            # Try to merge two nearby segments by filling the gap
            segments_sorted = sorted(segments, key=lambda x: x[0])
            
            for i in range(len(segments_sorted) - 1):
                start1, len1 = segments_sorted[i]
                start2, len2 = segments_sorted[i + 1]
                
                gap = start2 - (start1 + len1)
                
                if 1 <= gap <= 5 and len(mutated.n_positions) + gap <= self.max_n_positions:
                    # Fill the gap
                    gap_positions = list(range(start1 + len1, start2))
                    mutated.n_positions.extend(gap_positions)
                    mutated.n_positions = sorted(list(set(mutated.n_positions)))
                    break
        
        elif mutation_type == 'split_segment' and segments:
            # Split a segment by removing a position in the middle
            long_segments = [s for s in segments if s[1] > 3]
            if long_segments:
                segment = random.choice(long_segments)
                start, length = segment
                
                # Remove a position in the middle
                middle_pos = start + length // 2
                if middle_pos in mutated.n_positions:
                    mutated.n_positions.remove(middle_pos)
        
        elif mutation_type == 'shift_segment' and segments:
            # Shift an entire segment left or right
            segment = random.choice(segments)
            start, length = segment
            
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_start = max(0, min(mutated.sequence_length - length, start + shift))
            
            if new_start != start:
                # Remove old segment
                old_positions = set(range(start, start + length))
                mutated.n_positions = [p for p in mutated.n_positions if p not in old_positions]
                
                # Add new segment
                new_positions = list(range(new_start, new_start + length))
                mutated.n_positions.extend(new_positions)
                mutated.n_positions = sorted(list(set(mutated.n_positions)))[:self.max_n_positions]
        
        return mutated
    
    def evolve_generation(
        self, 
        population: List[DNASequenceIndividual],
        original_sequence: str
    ) -> List[DNASequenceIndividual]:
        """Evolve one generation"""
        # Evaluate fitness for all individuals
        for ind in population:
            if ind.fitness is None:
                self.evaluate_fitness(ind, original_sequence)
        
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
        original_sequence: str, 
        n_generations: int = 100,
        verbose: bool = True
    ) -> Tuple[DNASequenceIndividual, Dict]:
        """
        Run the genetic algorithm
        
        Returns:
            best_individual: Best solution found
            history: Dictionary with evolution history
        """
        sequence_length = len(original_sequence)
        
        # Initialize population
        population = self.initialize_population(sequence_length)
        
        # History tracking
        history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_entropy': [],
            'best_n_count': [],
            'best_continuity': [],
            'best_num_segments': []
        }
        
        # Evolution loop
        iterator = tqdm(range(n_generations), desc="Evolving") if verbose else range(n_generations)
        
        for generation in iterator:
            # Evolve
            population = self.evolve_generation(population, original_sequence)
            
            # Track statistics
            best_ind = max(population, key=lambda ind: ind.fitness)
            avg_fitness = np.mean([ind.fitness for ind in population])
            
            continuity_score = best_ind.calculate_continuity_score()
            num_segments = len(best_ind.get_continuous_segments())
            
            history['best_fitness'].append(best_ind.fitness)
            history['avg_fitness'].append(avg_fitness)
            history['best_entropy'].append(best_ind.entropy_score)
            history['best_n_count'].append(len(best_ind.n_positions))
            history['best_continuity'].append(continuity_score)
            history['best_num_segments'].append(num_segments)
            
            if verbose and generation % 10 == 0:
                iterator.set_postfix({
                    'best_fit': f'{best_ind.fitness:.4f}',
                    'entropy': f'{best_ind.entropy_score:.4f}',
                    'n_count': len(best_ind.n_positions),
                    'segments': num_segments
                })
        
        # Return best individual
        best_individual = max(population, key=lambda ind: ind.fitness)
        
        return best_individual, history


def analyze_solution(
    model,
    original_sequence: str,
    best_individual: DNASequenceIndividual,
    device: str = 'cuda'
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