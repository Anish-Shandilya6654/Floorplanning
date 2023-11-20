import numpy as np
import matplotlib.pyplot as plt


# Constants
NUM_MODULES = 8
MAX_CHIP_SIZE = 20
POPULATION_SIZE = 50
NUM_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1


# Module dimensions (width, height)
module_dimensions = np.array([(2, 3), (4, 2), (3, 4), (2, 2), (5, 3), (4, 3), (6, 2), (3, 5)])


# Function to generate a random floorplan
def generate_random_floorplan():
    return np.random.randint(0, MAX_CHIP_SIZE, size=(NUM_MODULES, 2))


# Function to plot the floorplan
def plot_floorplan(floorplan):
    plt.figure()

    for i, (width, height) in enumerate(module_dimensions):
        rect = plt.Rectangle((floorplan[i, 0], floorplan[i, 1]), width, height, fc=np.random.rand(3,), alpha=0.7)
        plt.gca().add_patch(rect)

    plt.title('Floorplan')
    plt.xlim(0, MAX_CHIP_SIZE)
    plt.ylim(0, MAX_CHIP_SIZE)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Function to calculate the cost (unused area) of a floorplan
def calculate_cost(floorplan):
    min_x = np.min(floorplan[:, 0])
    min_y = np.min(floorplan[:, 1])
    max_x = np.max(floorplan[:, 0] + module_dimensions[:, 0])
    max_y = np.max(floorplan[:, 1] + module_dimensions[:, 1])

    # Check if modules are outside the chip boundaries
    if min_x < 0 or min_y < 0 or max_x > MAX_CHIP_SIZE or max_y > MAX_CHIP_SIZE:
        return float('inf')  # Penalize floorplans that exceed chip boundaries

    # Check for overlaps and assign penalties
    penalty = 0
    for i in range(NUM_MODULES):
        for j in range(i + 1, NUM_MODULES):
            if (
                floorplan[i, 0] < floorplan[j, 0] + module_dimensions[j, 0] and
                floorplan[i, 0] + module_dimensions[i, 0] > floorplan[j, 0] and
                floorplan[i, 1] < floorplan[j, 1] + module_dimensions[j, 1] and
                floorplan[i, 1] + module_dimensions[i, 1] > floorplan[j, 1]
            ):
                penalty += 1

    # Calculate the unused area within the bounding rectangle
    width = max_x - min_x
    height = max_y - min_y

    # Ensure non-negative dimensions
    width = max(0, width)
    height = max(0, height)

    # Calculate the unused area within the bounding rectangle
    unused_area = width * height - np.sum(module_dimensions[:, 0] * module_dimensions[:, 1])

    # Add penalty to the cost
    cost = max(0, unused_area) + penalty * 30  # Adjust penalty weight as needed

    return cost


# Function to perform crossover
def crossover(parent1, parent2):
    mask = np.random.rand(NUM_MODULES) > 0.5
    child1 = np.where(mask[:, None], parent1, parent2)
    child2 = np.where(mask[:, None], parent2, parent1)
    return child1, child2


# Function to perform mutation
def mutate(floorplan):
    mask = np.random.rand(NUM_MODULES, 2) < MUTATION_RATE
    mutated_floorplan = floorplan + np.random.randint(-1, 2, size=(NUM_MODULES, 2)) * mask

    # Ensure modules do not go beyond chip boundaries
    mutated_floorplan = np.clip(mutated_floorplan, 0, MAX_CHIP_SIZE - 1)

    # Resolve overlaps
    for i in range(NUM_MODULES):
        for j in range(i + 1, NUM_MODULES):
            if (
                mutated_floorplan[i, 0] < mutated_floorplan[j, 0] + module_dimensions[j, 0] and
                mutated_floorplan[i, 0] + module_dimensions[i, 0] > mutated_floorplan[j, 0] and
                mutated_floorplan[i, 1] < mutated_floorplan[j, 1] + module_dimensions[j, 1] and
                mutated_floorplan[i, 1] + module_dimensions[i, 1] > mutated_floorplan[j, 1]
            ):
                # Overlapping modules, move them away from each other
                dx = mutated_floorplan[i, 0] - mutated_floorplan[j, 0]
                dy = mutated_floorplan[i, 1] - mutated_floorplan[j, 1]

                move = max(module_dimensions[i, 0], module_dimensions[i, 1], module_dimensions[j, 0], module_dimensions[j, 1])

                if abs(dx) < abs(dy):
                    mutated_floorplan[i, 0] += np.sign(dx) * move
                else:
                    mutated_floorplan[i, 1] += np.sign(dy) * move

    return mutated_floorplan


# Initialize population
population = [generate_random_floorplan() for _ in range(POPULATION_SIZE)]
plot_floorplan(population[0])


# Evolution loop
for generation in range(NUM_GENERATIONS):
    # Calculate costs for each individual in the population
    costs = [calculate_cost(individual) for individual in population]

    # Select parents based on tournament selection
    parents_indices = np.random.choice(range(POPULATION_SIZE), size=(POPULATION_SIZE // 2, 2), replace=False)
    parents = [population[i] for indices in parents_indices for i in indices]

    # Perform crossover and mutation
    new_population = []
    for i in range(0, len(parents), 2):
        if np.random.rand() < CROSSOVER_RATE:
            child1, child2 = crossover(parents[i], parents[i + 1])
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

    # Select the best individuals for the next generation
    population = population + new_population
    costs = [calculate_cost(individual) for individual in population]
    population = [population[i] for i in np.argsort(costs)[:POPULATION_SIZE]]

    # Print best cost in each generation
    best_cost = min(costs)
    print(f"Generation {generation+1}, Best Cost: {best_cost}")

# Get the best floorplan
best_floorplan = population[np.argmin(costs)]


# Plot the best floorplan
plot_floorplan(best_floorplan)