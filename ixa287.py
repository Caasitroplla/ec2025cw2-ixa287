import argparse
import random
import time
from typing import List, Tuple

def check_satisfiability(assignment: str, clause: str) -> int:
    """
    Check if a clause is satisfied by a given assignment.

    Args:
        assignment: A bitstring representing the assignment (e.g., "1010")
        clause: A string representing the clause (e.g., "0 5 2 1 -3 -4 0")

    Returns:
        1 if the clause is satisfied, 0 otherwise
    """
    # Split the clause into individual literals and convert to integers
    # Skip the first value (0) and the last value (0)
    literals = [int(x) for x in clause.split()[1:-1]]

    # For each literal in the clause
    for literal in literals:
        # Get the variable index (1-based)
        var_index = abs(literal) - 1

        # Check if the variable is within the assignment length
        if var_index >= len(assignment):
            continue

        # Get the value of the variable from the assignment
        var_value = int(assignment[var_index])

        # If literal is positive and var_value is 1, clause is satisfied
        # If literal is negative and var_value is 0, clause is satisfied
        if (literal > 0 and var_value == 1) or (literal < 0 and var_value == 0):
            return 1

    # If no literal satisfied the clause, return 0
    return 0

def count_satisfied_clauses(wdimacs_file: str, assignment: str) -> int:
    """
    Count the weighted sum of satisfied clauses in a WDIMACS file for a given assignment.

    Args:
        wdimacs_file: Path to the WDIMACS format file
        assignment: Assignment as a bitstring

    Returns:
        Weighted sum of satisfied clauses
    """
    satisfied_sum = 0

    with open(wdimacs_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('c'):
                continue

            # Skip the problem line (starts with 'p')
            if line.startswith('p'):
                continue

            # Skip weight lines (start with 'w')
            if line.startswith('w'):
                continue

            # Process clause lines
            if line:
                tokens = line.split()

                # Get weight and clause
                weight = 1  # Default weight if not specified
                if tokens and tokens[0].isdigit():
                    weight = int(tokens[0])
                    clause = "0 " + " ".join(tokens[1:])
                else:
                    clause = line

                # Add weight to sum if clause is satisfied
                if check_satisfiability(assignment, clause):
                    satisfied_sum += weight

    return satisfied_sum

def evolutionary_algorithm(wdimacs_file: str, time_budget: float, population_size: int = 100) -> Tuple[int, int, str]:
    """
    Run the evolutionary algorithm for MAXSAT.

    Returns:
        Tuple of (runtime, number of satisfied clauses, best solution)
    """

    def get_num_variables(wdimacs_file: str) -> int:
        """Get the number of variables from the WDIMACS file."""
        with open(wdimacs_file, 'r') as f:
            for line in f:
                if line.startswith('p'):
                    return int(line.split()[2])
        raise ValueError("No problem line found in WDIMACS file")

    def create_random_individual(num_variables: int) -> str:
        """Create a random individual as a bitstring."""
        return ''.join(random.choice('01') for _ in range(num_variables))

    def tournament_selection(population: List[str], fitness_values: List[int], tournament_size: int = 3) -> str:
        """Select an individual using tournament selection."""
        tournament_idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_idx]
        winner_idx = tournament_idx[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]

    def crossover(parent1: str, parent2: str) -> str:
        """Perform uniform crossover between two parents."""
        return ''.join(random.choice(p1 + p2) for p1, p2 in zip(parent1, parent2))

    def mutation(individual: str, mutation_rate: float = 0.1) -> str:
        """Perform bit-flip mutation on an individual."""
        return ''.join('1' if bit == '0' and random.random() < mutation_rate else
                    '0' if bit == '1' and random.random() < mutation_rate else
                    bit for bit in individual)

    num_variables = get_num_variables(wdimacs_file)
    start_time = time.time()

    # Initialize population
    population = [create_random_individual(num_variables) for _ in range(population_size)]
    fitness_values = [count_satisfied_clauses(wdimacs_file, ind) for ind in population]
    best_solution = population[fitness_values.index(max(fitness_values))]
    best_fitness = max(fitness_values)
    generations = 0

    while time.time() - start_time < time_budget:
        # Create new population
        new_population = []
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutation(child)

            new_population.append(child)

        # Evaluate new population
        new_fitness = [count_satisfied_clauses(wdimacs_file, ind) for ind in new_population]

        # Update best solution
        max_new_fitness = max(new_fitness)
        if max_new_fitness > best_fitness:
            best_fitness = max_new_fitness
            best_solution = new_population[new_fitness.index(max_new_fitness)]

        # Update population and fitness values
        population = new_population
        fitness_values = new_fitness
        generations += 1

    runtime = generations * population_size
    return runtime, best_fitness, best_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAT solver utilities')
    parser.add_argument('-question', required=True, help='Question number (e.g., "1")')
    parser.add_argument('-assignment', required=False, help='Assignment as a bitstring (e.g., "1010")')
    parser.add_argument('-clause', required=False, help='Clause description (e.g., "0 5 2 1 -3 -4 0")')
    parser.add_argument('-wdimacs', required=False, help='Path to WDIMACS format file')
    parser.add_argument('-time_budget', required=False, type=float, help='Time budget in seconds per repetition')
    parser.add_argument('-repetitions', required=False, type=int, help='Number of repetitions')

    args = parser.parse_args()

    if args.question == "1":
        if not args.clause or not args.assignment:
            raise ValueError("Clause and assignment arguments are required for question 1")
        result = check_satisfiability(args.assignment, args.clause)
        print(result)  # Just print the result (1 or 0) as required
    elif args.question == "2":
        if not args.wdimacs or not args.assignment:
            raise ValueError("WDIMACS file and assignment arguments are required for question 2")
        result = count_satisfied_clauses(args.wdimacs, args.assignment)
        print(result)  # Print the number of satisfied clauses
    elif args.question == "3":
        if not args.wdimacs or not args.time_budget or not args.repetitions:
            raise ValueError("WDIMACS file, time budget, and repetitions arguments are required for question 3")

        for _ in range(args.repetitions):
            runtime, nsat, xbest = evolutionary_algorithm(args.wdimacs, args.time_budget)
            print(f"{runtime}\t{nsat}\t{xbest}")
    else:
        raise ValueError(f"Invalid question number: {args.question}")