import networkx as nx
import random
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import heapq
import math


# Function to generate a random directed graph
def generate_random_directed_graph(num_nodes, edge_prob):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < edge_prob:
                G.add_edge(i, j)
    return G


# Function to calculate the influence set I_d(u)
def influence_set(G, u, d):
    return set(nx.single_source_shortest_path_length(G, u, cutoff=d).keys())


# Function to calculate the influence set I_d(U) for a set of nodes U
def combined_influence_set(G, U, d):
    combined_set = set()
    for u in U:
        combined_set.update(influence_set(G, u, d))
    return combined_set


# Biased Random Key Genetic Algorithm (BRKGA) to solve the k-dDSP
def brkga_k_dDSP(G, k, d, population_size=50, generations=100, elite_size=0.2, mutant_size=0.1, bias=0.7):
    def fitness(solution):
        return len(combined_influence_set(G, solution, d))

    def decode(chromosome):
        node_list = list(G.nodes())
        selected_indices = sorted(range(len(chromosome)), key=lambda i: chromosome[i])[:k]
        return set(node_list[i] for i in selected_indices)

    def initialize_population():
        return [[random.random() for _ in range(len(G.nodes()))] for _ in range(population_size)]

    def select_parents(population):
        elite_count = int(population_size * elite_size)
        return population[:elite_count]

    def crossover(parent1, parent2):
        return [parent1[i] if random.random() < bias else parent2[i] for i in range(len(parent1))]

    def mutate():
        return [random.random() for _ in range(len(G.nodes()))]

    population = initialize_population()

    for generation in range(generations):
        population = sorted(population, key=lambda chromosome: fitness(decode(chromosome)), reverse=True)
        next_population = select_parents(population)

        while len(next_population) < population_size - int(population_size * mutant_size):
            parent1, parent2 = random.sample(select_parents(population), 2)
            child = crossover(parent1, parent2)
            next_population.append(child)

        while len(next_population) < population_size:
            next_population.append(mutate())

        population = next_population

    best_chromosome = max(population, key=lambda chromosome: fitness(decode(chromosome)))
    best_solution = decode(best_chromosome)
    return best_solution, combined_influence_set(G, best_solution, d)


# Genetic Algorithm to solve the k-dDSP
def genetic_algorithm_k_dDSP(G, k, d, population_size=50, generations=100, mutation_rate=0.1):
    def fitness(solution):
        return len(combined_influence_set(G, solution, d))

    def selection(population):
        return sorted(population, key=fitness, reverse=True)[:population_size // 2]

    def crossover(parent1, parent2):
        parent1 = list(parent1)
        parent2 = list(parent2)
        point = random.randint(1, k - 1)
        child = parent1[:point] + parent2[point:]
        return set(child)

    def mutate(solution):
        solution = list(solution)
        if random.random() < mutation_rate:
            solution.pop()
            solution.append(random.choice(list(set(G.nodes()) - set(solution))))
        return set(solution)

    # Initialize population
    node_list = list(G.nodes())
    population = [set(random.sample(node_list, k)) for _ in range(population_size)]

    for _ in range(generations):
        selected_population = selection(population)
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population

    best_solution = max(population, key=fitness)
    return best_solution, combined_influence_set(G, best_solution, d)


# Simulated Annealing to solve the k-dDSP
def simulated_annealing_k_dDSP(G, k, d, initial_temp=1000, cooling_rate=0.95, max_iter=1000):
    def fitness(solution):
        return len(combined_influence_set(G, solution, d))

    def neighbor(solution):
        new_solution = set(solution)
        new_solution.remove(random.choice(tuple(new_solution)))
        new_solution.add(random.choice(list(set(G.nodes()) - new_solution)))
        return new_solution

    current_solution = set(random.sample(list(G.nodes()), k))
    current_fitness = fitness(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness
    temperature = initial_temp

    for _ in range(max_iter):
        if temperature <= 0:
            break
        new_solution = neighbor(current_solution)
        new_fitness = fitness(new_solution)
        if new_fitness > current_fitness or math.exp((new_fitness - current_fitness) / temperature) > random.random():
            current_solution = new_solution
            current_fitness = new_fitness
            if new_fitness > best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
        temperature *= cooling_rate

    return best_solution, combined_influence_set(G, best_solution, d)


# Lazy Greedy algorithm to solve the k-dDSP
def lazy_greedy_k_dDSP(G, k, d):
    all_nodes = list(G.nodes())
    influence_sets = {node: influence_set(G, node, d) for node in all_nodes}
    max_heap = [(-len(influence_sets[node]), node) for node in all_nodes]
    heapq.heapify(max_heap)

    selected_nodes = set()
    combined_influence = set()

    while len(selected_nodes) < k:
        while True:
            current_max = heapq.heappop(max_heap)
            node = current_max[1]
            if -current_max[0] == len(influence_sets[node] - combined_influence):
                selected_nodes.add(node)
                combined_influence.update(influence_sets[node])
                break
            else:
                updated_influence = influence_sets[node] - combined_influence
                heapq.heappush(max_heap, (-len(updated_influence), node))

    return selected_nodes, combined_influence


# Parameters for the experiment
num_nodes_list = [10, 20, 50, 100, 200]
edge_prob_list = [0.1, 0.2, 0.3, 0.4]
k_list = [2, 3, 4, 5]
d_list = [1, 2, 3]
repeats = 3

# Run the experiment using all four algorithms
results_ga = []
results_sa = []
results_lg = []
results_brkga = []

fixed_edge_prob = 0.2

for num_nodes in num_nodes_list:
    for edge_prob in edge_prob_list:
        for k in k_list:
            for d in d_list:
                for _ in range(repeats):
                    G = generate_random_directed_graph(num_nodes, edge_prob)

                    # Genetic Algorithm
                    start_time = time.time()
                    selected_nodes, influence = genetic_algorithm_k_dDSP(G, k, d)
                    end_time = time.time()
                    computing_time = end_time - start_time
                    num_influenced_nodes = len(influence)
                    results_ga.append(
                        (num_nodes, edge_prob, k, d, 'Genetic Algorithm', computing_time, num_influenced_nodes))

                    # Simulated Annealing
                    start_time = time.time()
                    selected_nodes, influence = simulated_annealing_k_dDSP(G, k, d)
                    end_time = time.time()
                    computing_time = end_time - start_time
                    num_influenced_nodes = len(influence)
                    results_sa.append(
                        (num_nodes, edge_prob, k, d, 'Simulated Annealing', computing_time, num_influenced_nodes))

                    # Lazy Greedy
                    start_time = time.time()
                    selected_nodes, influence = lazy_greedy_k_dDSP(G, k, d)
                    end_time = time.time()
                    computing_time = end_time - start_time
                    num_influenced_nodes = len(influence)
                    results_lg.append((num_nodes, edge_prob, k, d, 'Lazy Greedy', computing_time, num_influenced_nodes))

                    # Biased Random Key Genetic Algorithm
                    start_time = time.time()
                    selected_nodes, influence = brkga_k_dDSP(G, k, d)
                    end_time = time.time()
                    computing_time = end_time - start_time
                    num_influenced_nodes = len(influence)
                    results_brkga.append((num_nodes, edge_prob, k, d, 'BRKGA', computing_time, num_influenced_nodes))

# Combine all results
results = results_ga + results_sa + results_lg + results_brkga
df = pd.DataFrame(results,
                  columns=['num_nodes', 'edge_prob', 'k', 'd', 'algorithm', 'computing_time', 'num_influenced_nodes'])

# Filter data for the fixed edge probability
df_fixed_p = df[df['edge_prob'] == fixed_edge_prob]

# Filter data for 100 nodes
fixed_num_nodes = 100
df_fixed_n = df[df['num_nodes'] == fixed_num_nodes]

# Plot computing time vs. number of nodes
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_fixed_n, x='num_nodes', y='computing_time', hue='algorithm', style='edge_prob', markers=True,
             dashes=False)
plt.title('Computing Time vs. Number of Nodes (Fixed 100 Nodes)')
plt.xlabel('Number of Nodes')
plt.ylabel('Computing Time (seconds)')
plt.legend(title='Algorithm and Edge Probability')
plt.grid(True)
plt.show()

# Plot quality of the solution vs. number of nodes for fixed edge probability
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_fixed_p, x='num_nodes', y='num_influenced_nodes', hue='algorithm', markers=True, dashes=False)
plt.title(f'Quality of Solution vs. Number of Nodes (p={fixed_edge_prob})')
plt.xlabel('Number of Nodes')
plt.ylabel('Number of Influenced Nodes')
plt.legend(title='Algorithm')
plt.grid(True)
plt.show()

# Plot computing time vs. number of influential nodes (k)
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_fixed_n, x='k', y='computing_time', hue='algorithm', style='edge_prob', markers=True, dashes=False)
plt.title('Computing Time vs. Number of Influential Nodes (k) (Fixed 100 Nodes)')
plt.xlabel('Number of Influential Nodes (k)')
plt.ylabel('Computing Time (seconds)')
plt.legend(title='Algorithm and Edge Probability')
plt.grid(True)
plt.show()

# Plot quality of the solution vs. number of influential nodes (k) for fixed edge probability
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_fixed_p, x='k', y='num_influenced_nodes', hue='algorithm', markers=True, dashes=False)
plt.title(f'Quality of Solution vs. Number of Influential Nodes (k) (p={fixed_edge_prob})')
plt.xlabel('Number of Influential Nodes (k)')
plt.ylabel('Number of Influenced Nodes')
plt.legend(title='Algorithm')
plt.grid(True)
plt.show()

# Plot computing time vs. edge probability for fixed number of nodes
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_fixed_n, x='edge_prob', y='computing_time', hue='algorithm', markers=True, dashes=False)
plt.title('Computing Time vs. Edge Probability (Fixed 100 Nodes)')
plt.xlabel('Edge Probability')
plt.ylabel('Computing Time (seconds)')
plt.legend(title='Algorithm')
plt.grid(True)
plt.show()

# Plot quality of the solution vs. edge probability for fixed number of nodes (e.g., 100 nodes)
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_fixed_n, x='edge_prob', y='num_influenced_nodes', hue='algorithm', markers=True, dashes=False)
plt.title(f'Quality of Solution vs. Edge Probability (Number of Nodes = {fixed_num_nodes})')
plt.xlabel('Edge Probability')
plt.ylabel('Number of Influenced Nodes')
plt.legend(title='Algorithm')
plt.grid(True)
plt.show()
