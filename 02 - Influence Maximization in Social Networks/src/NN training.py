import networkx as nx
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import heapq
import matplotlib.pyplot as plt


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


# Data generation using Lazy Greedy Algorithm
def generate_training_data(num_samples, num_nodes_list, edge_prob_list, k_list, d_list):
    data = []
    print("Generating training data...")
    for i in range(num_samples):
        if i % 100 == 0 and i > 0:
            print(f"Generated {i} samples...")
        num_nodes = random.choice(num_nodes_list)
        edge_prob = random.choice(edge_prob_list)
        k = random.choice(k_list)
        d = random.choice(d_list)
        G = generate_random_directed_graph(num_nodes, edge_prob)
        node_features = [G.degree(n) for n in G.nodes()]

        # Use Lazy Greedy Algorithm to generate solutions
        selected_nodes_lg, _ = lazy_greedy_k_dDSP(G, k, d)

        # Calculate influence scores for LGA
        influence_score_lg = len(combined_influence_set(G, selected_nodes_lg, d))

        # Append data
        data.append((G, node_features, list(selected_nodes_lg), influence_score_lg))
    print("Training data generation complete.")
    return data


def normalize_features(df):
    for i, row in df.iterrows():
        features = np.array(row['node_features'])
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            df.at[i, 'node_features'] = [(x - mean) / std for x in features]
        else:
            df.at[i, 'node_features'] = features  # No normalization if std is zero
    return df


# Example usage
num_samples = 5000  # Increased number of samples
num_nodes_list = [10, 20, 50, 100]
edge_prob_list = [0.1, 0.2, 0.3, 0.4]
k_list = [2, 3, 4, 5]
d_list = [1, 2, 3]
data = generate_training_data(num_samples, num_nodes_list, edge_prob_list, k_list, d_list)

# Convert data to a DataFrame for easier handling
df = pd.DataFrame(data, columns=['graph', 'node_features', 'selected_nodes', 'influence_score'])
df = normalize_features(df)
df.to_pickle("training_data.pkl")


# Define the Graph Neural Network
class InfluenceMaximizationGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(InfluenceMaximizationGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# Hyperparameters
input_dim = 1  # Degree of nodes
hidden_dim = 256  # Increased hidden units
output_dim = 1  # Influence score
learning_rate = 0.001
batch_size = 64
num_epochs = 30  # Increased number of epochs

model = InfluenceMaximizationGNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Prepare the dataset for PyTorch Geometric
def prepare_data(df):
    data_list = []
    print("Preparing data for PyTorch Geometric...")
    for i, row in df.iterrows():
        if i % 100 == 0 and i > 0:
            print(f"Prepared {i} samples...")
        G = row['graph']
        x = torch.tensor([[d] for d in row['node_features']], dtype=torch.float)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        y = torch.tensor([row['influence_score']], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    print("Data preparation complete.")
    return data_list


data_list = prepare_data(df)
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Split the data into training and validation sets
train_size = int(0.8 * len(data_list))
val_size = len(data_list) - train_size
train_data, val_data = torch.utils.data.random_split(data_list, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Training loop with loss tracking
train_losses = []
val_losses = []
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch)
            loss = criterion(output, batch.y)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
print("Training complete.")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Save the model
torch.save(model.state_dict(), "influence_maximization_gnn.pth")
print("Model saved.")

# Generate test data and prepare for evaluation
print("Generating test data...")
test_data = generate_training_data(200, num_nodes_list, edge_prob_list, k_list, d_list)
test_df = pd.DataFrame(test_data, columns=['graph', 'node_features', 'selected_nodes', 'influence_score'])
test_df = normalize_features(test_df)
test_list = prepare_data(test_df)
test_loader = DataLoader(test_list, batch_size=32, shuffle=False)

# Evaluation loop with prediction tracking
print("Starting evaluation...")
model.eval()
total_loss = 0
predictions = []
actuals = []
with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        loss = criterion(output, batch.y)
        total_loss += loss.item()
        predictions.extend(output.cpu().numpy())
        actuals.extend(batch.y.cpu().numpy())
avg_test_loss = total_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss}')

# Plot predicted vs actual influence scores with 45-degree line
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.6, label='Predictions')
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--',
         label='45-degree line')
plt.xlabel('Actual Influence Score')
plt.ylabel('Predicted Influence Score')
plt.title('Predicted vs Actual Influence Scores')
plt.legend()
plt.grid(True)
plt.show()
