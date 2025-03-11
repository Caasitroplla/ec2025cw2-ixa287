from ixa287 import evolutionary_algorithm
import matplotlib.pyplot as plt
import os

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Problem instance
original_file = "data/TWComp_1c75_N69.wcnf"

# Parameter settings to test
default_params = {
    'population_size': 20,
    'time_budget': 2.0,
    'mutation_rate': 0.2
}

# Parameter ranges to test
param_ranges = {
    'population_size': [10, 20, 50, 100],
    'time_budget': [0.5, 1.0, 2.0, 5.0],
    'mutation_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
}

repetitions = 100

def run_experiment(param_name, param_value):
    results = []
    for rep in range(repetitions):
        # Create parameter dictionary with default values
        params = default_params.copy()
        # Override the parameter being tested
        params[param_name] = param_value

        runtime, nsat, xbest = evolutionary_algorithm(
            original_file,
            params['time_budget'],
            population_size=params['population_size'],
            mutation_rate=params['mutation_rate']
        )
        results.append(nsat)
        if rep % 10 == 0:  # Print progress every 10 repetitions
            print(f"Rep {rep+1}/{repetitions}: nsat = {nsat} time = {runtime}")
    return results

def create_boxplot(results, labels, param_name, param_values):
    plt.figure(figsize=(10, 6))
    plt.boxplot(results, tick_labels=labels)

    # Customize labels based on parameter
    if param_name == 'population_size':
        plt.xlabel('Population Size')
    elif param_name == 'time_budget':
        plt.xlabel('Time Budget (seconds)')
    else:  # mutation_rate
        plt.xlabel('Mutation Rate')

    plt.ylabel('Number of Satisfied Clauses')

    # Create title with default parameters
    title = f'Impact of {param_name.replace("_", " ").title()} on Solution Quality\n'
    title += f'(MAXSAT Instance: TWComp_1c75_N69.wcnf, '
    title += f'Default: {default_params["population_size"]} pop, '
    title += f'{default_params["time_budget"]}s, '
    title += f'{default_params["mutation_rate"]} mut)'
    plt.title(title)

    # Set y-axis limits based on data range with padding
    all_values = [val for sublist in results for val in sublist]
    y_min = min(all_values)
    y_max = max(all_values)
    padding = (y_max - y_min) * 0.1  # 10% padding
    plt.ylim(y_min - padding, y_max + padding)

    plt.tight_layout()
    plt.savefig(f'images/{param_name}_impact.png')
    plt.close()

# Test each parameter independently
for param_name, param_values in param_ranges.items():
    print(f"\nTesting {param_name}...")
    results = []
    labels = []

    for param_value in param_values:
        print(f"\nTesting {param_name} = {param_value}")
        instance_results = run_experiment(param_name, param_value)
        results.append(instance_results)
        labels.append(str(param_value))

    # Create and save boxplot for this parameter
    create_boxplot(results, labels, param_name, param_values)

print("\nAll experiments completed. Results saved in the images folder.")

