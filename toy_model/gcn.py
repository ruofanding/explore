# dataset: https://snap.stanford.edu/data/twitch-social-networks.html
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import random
from sklearn.metrics import roc_auc_score, precision_score


# 1. Load the dataset
data_dir = '/Users/ruofanding/Downloads/twitch/ENGB/'
# Load node features and labels
target_df = pd.read_csv(data_dir + 'musae_ENGB_target.csv')

# Load the edge list (friendship network)
edges_df = pd.read_csv(data_dir + 'musae_ENGB_edges.csv')

# Load the features
with open(data_dir + 'musae_ENGB_features.json') as f:
    node_features = json.load(f)

# Get a list of all unique features
all_features = set()
for features in node_features.values():
    all_features.update(features)
all_features = sorted(list(all_features))  # Sort to maintain consistent indexing

# Create a mapping from feature ID to index in the binary feature vector
feature_to_idx = {feature: idx for idx, feature in enumerate(all_features)}

# Convert node features to a binary feature matrix
num_nodes = len(node_features)
num_features = len(all_features)
X = torch.zeros((num_nodes, num_features))

for node_id, features in node_features.items():
    for feature in features:
        if feature in feature_to_idx:
            X[int(node_id), feature_to_idx[feature]] = 1

# Create edge list tensor
edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
print('done loading data')
# 2. Edge masking: Hide some edges for training and use them for evaluation
num_edges = edge_index.size(1)
num_hidden_edges = int(0.10 * num_edges)  # Hide 10% of edges for validation
print('num edges:', num_edges, 'num hidden edges:', num_hidden_edges)
# Shuffle the edges and split into training and test sets
edge_perm = torch.randperm(num_edges)
hidden_edges = edge_index[:, edge_perm[:num_hidden_edges]]  # Hidden for testing
train_edges = edge_index[:, edge_perm[num_hidden_edges:]]   # Used for training


def negative_sampling(edge_index, num_nodes, x):
    """
    For each node (in-degree), sample `x` unique out-degree nodes that are not connected to it.
    """
    # Convert existing edges into a set for quick lookup
    existing_edges = set((u, v) for u, v in edge_index.t().tolist())

    neg_edges = []

    # Loop through each in-degree node (first node in the edge pair)
    for in_node in range(num_nodes):
        out_nodes = set()

        # Sample `x` unique out-degree nodes that are not already connected to the in-degree node
        while len(out_nodes) < x:
            candidate_out_node = torch.randint(0, num_nodes, (1,)).item()
            # Ensure no self-loop and the edge doesn't already exist
            if candidate_out_node != in_node and (in_node, candidate_out_node) not in existing_edges:
                out_nodes.add(candidate_out_node)

        # Add the sampled negative edges for this in-degree node
        for out_node in out_nodes:
            neg_edges.append((in_node, out_node))

    # Return the negative edges as a tensor
    return torch.tensor(neg_edges, dtype=torch.long).t()

num_negative_samples = num_hidden_edges  # Equal number of negative samples as hidden edges
neg_edge_index = negative_sampling(train_edges, num_nodes, 1)


num_neg_edges = neg_edge_index.size(1)
edge_perm = torch.randperm(num_neg_edges)
num_neg_hidden_edges = int(0.10 * num_neg_edges)  # Hide 10% of edges for validation
neg_hidden_edges = neg_edge_index[:, edge_perm[:num_neg_hidden_edges]]  # Hidden for testing
neg_train_edges = neg_edge_index[:, edge_perm[num_neg_hidden_edges:]]   # Used for training


print('Done negative sampling')
# 3. Prepare graph data in PyTorch Geometric format
data = Data(x=X, edge_index=train_edges)

# 4. Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        return x

# Edge prediction function: Calculate link probabilities
def link_prediction(z, edge_index):
    # Inner product of node embeddings for each edge (dot product)
    return (F.normalize(z[edge_index[0]]) * F.normalize(z[edge_index[1]])).sum(dim=1)

# 5. Initialize model, optimizer, and loss function
hidden_dim = 32
model = GCN(num_node_features=X.size(1), hidden_dim=hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Binary cross-entropy loss function
loss_fn = torch.nn.BCEWithLogitsLoss()

# 6. Training function
def train():
    model.train()
    optimizer.zero_grad()

    # Forward pass: Get node embeddings from GCN
    z = model(data)

    # Predict scores for positive edges
    pos_score = link_prediction(z, train_edges)

    # Predict scores for negative edges (non-existent edges)
    neg_score = link_prediction(z, neg_train_edges)

    # Labels: 1 for positive edges, 0 for negative edges
    labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])

    # Concatenate scores for both positive and negative edges
    scores = torch.cat([pos_score, neg_score])

    # Compute loss
    loss = loss_fn(scores, labels)
    loss.backward()

    # grad clip is not necessary after changing dot product to cosine similiarity in link_prediction
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

# 7. Evaluation function (on hidden edges)
def evaluate():
    model.eval()
    with torch.no_grad():
        z = model(data)

        # Predict scores for positive hidden edges
        pos_score = link_prediction(z, hidden_edges)

        # Predict scores for negative edges
        neg_hidden_edges = neg_train_edges
        neg_score = link_prediction(z, neg_hidden_edges)

        # Labels: 1 for positive edges, 0 for negative edges
        labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])

        # Concatenate scores for both positive and negative edges
        scores = torch.cat([pos_score, neg_score])

        # Use binary classification accuracy (threshold = 0.5)
        pred = (scores > 0).float()
        acc = (pred == labels).float().mean().item()
        auc = roc_auc_score(labels, scores)

        return acc, auc, torch.mean(pos_score).item(), torch.mean(neg_score).item()

# 8. Train the model for 100 epochs
for epoch in range(10000):
    loss = train()
    if epoch % 100 == 0:
        acc, auc, pos_prob, neg_prob = evaluate()
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}, AUC: {auc:.4f}, '
              f'Avg. Pos. Score: {pos_prob:.4f}, Avg. Neg. Score: {neg_prob:.4f}')