import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_MODULES = 8
MAX_CHIP_SIZE = 20
INITIAL_TEMPERATURE = 1000
COOLING_RATE = 0.9
NUM_ITERATIONS = 1000
T_MIN=0.0001
# Module dimensions (width, height)
module_dimensions = np.array([(2, 3), (4, 2), (3, 4), (2, 2), (5, 3), (4, 3), (6, 2), (3, 5)])


# Function to generate a random floorplan
def generate_random_floorplan():
    return np.random.randint(0, MAX_CHIP_SIZE, size=(NUM_MODULES, 2))

colors = np.random.rand(NUM_MODULES, 3)

def plot_floorplan(floorplan):
    plt.figure()

    for i, (width, height) in enumerate(module_dimensions):
        rect = plt.Rectangle((floorplan[i, 0], floorplan[i, 1]), width, height, fc=colors[i], alpha=0.7)
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

    penalty = 0

    # Check if modules are outside the chip boundaries
    if min_x < 0 or min_y < 0 or max_x > MAX_CHIP_SIZE or max_y > MAX_CHIP_SIZE:
        penalty+= 100# Penalize floorplans that exceed chip boundaries

    # Check for overlaps and assign penalties
    
    for i in range(NUM_MODULES):
        for j in range(i + 1, NUM_MODULES):
            if (
                floorplan[i, 0] < floorplan[j, 0] + module_dimensions[j, 0] and
                floorplan[i, 0] + module_dimensions[i, 0] > floorplan[j, 0] and
                floorplan[i, 1] < floorplan[j, 1] + module_dimensions[j, 1] and
                floorplan[i, 1] + module_dimensions[i, 1] > floorplan[j, 1]
            ):
                penalty += 2

    # Calculate the unused area within the bounding rectangle
    width = max_x - min_x
    height = max_y - min_y

    # Ensure non-negative dimensions
    width = max(0, width)
    height = max(0, height)

    # Calculate the unused area within the bounding rectangle
    unused_area = width * height - np.sum(module_dimensions[:, 0] * module_dimensions[:, 1])

    # Add penalty to the cost
    cost = max(0, unused_area) + penalty * 10  # Adjust penalty weight as needed

    return cost


# Simulated Annealing Algorithm
def simulated_annealing(initial_floorplan, initial_temperature, cooling_rate, num_iterations):
    current_floorplan = initial_floorplan
    best_floorplan = initial_floorplan
    current_cost = calculate_cost(current_floorplan)
    best_cost = current_cost

    temperature=initial_temperature
    while temperature>T_MIN:
        print("Best cost : ",best_cost)
        for i in range(num_iterations):
            

            # Generate a neighboring solution (perturbation)
            neighbor_floorplan = generate_neighbor(current_floorplan)

            # Calculate the cost of the neighboring solution
            neighbor_cost = calculate_cost(neighbor_floorplan)
            if current_cost < best_cost:
                    best_floorplan = current_floorplan
                    best_cost = current_cost

            # Decide whether to accept the neighboring solution
            if neighbor_cost < current_cost or np.random.rand() < np.exp((current_cost - neighbor_cost) / temperature):
                current_floorplan = neighbor_floorplan
                current_cost = neighbor_cost

                # Update the best solution if applicable
                
        temperature=temperature * COOLING_RATE 

    return best_floorplan, best_cost

 
# Function to generate a neighboring floorplan (perturbation)
def generate_neighbor(current_floorplan):
    neighbor_floorplan = current_floorplan.copy()

    # Select a random module to move
    module_index = np.random.randint(NUM_MODULES)

    # Randomly generate new coordinates within chip boundaries
    new_x = np.random.randint(0, MAX_CHIP_SIZE - module_dimensions[module_index][0] + 1)
    new_y = np.random.randint(0, MAX_CHIP_SIZE - module_dimensions[module_index][1] + 1)

    # Update the position of the selected module
    neighbor_floorplan[module_index] = [new_x, new_y]

    return neighbor_floorplan




# Initialize the floorplan
initial_floorplan = generate_random_floorplan()
plot_floorplan(initial_floorplan)

# Run the simulated annealing algorithm
best_solution, best_cost = simulated_annealing(initial_floorplan, INITIAL_TEMPERATURE, COOLING_RATE, NUM_ITERATIONS)

# Plot the best floorplan
plot_floorplan(best_solution)
