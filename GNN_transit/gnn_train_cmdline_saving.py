import os

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
import time
from random import shuffle

import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Primary device: {device}")

# Define DotDict class for easy attribute access
class DotDict(dict):
    """Dictionary wrapper to access keys as attributes."""
    def __getattr__(self, key):
        return self.get(key)

# Define GoogleTSPReader class
class GoogleTSPReader:
    """Iterator to read and process TSP dataset files in mini-batches."""

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath):
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath

        # Read and shuffle the data
        with open(filepath, "r") as f:
            self.filedata = f.readlines()
        shuffle(self.filedata)

        self.max_iter = len(self.filedata) // batch_size

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx])

    def process_batch(self, lines):
        """Convert raw lines into a mini-batch as a DotDict."""
        batch_size = len(lines)
        nodes_coord = np.zeros((batch_size, self.num_nodes, 2), dtype=np.float32)
        batch_x_mid = np.zeros(batch_size, dtype=np.float32)
        batch_y_mid = np.zeros(batch_size, dtype=np.float32)
        batch_edges = np.zeros((batch_size, self.num_nodes, self.num_nodes), dtype=np.float32)
        batch_edges_values = np.zeros((batch_size, self.num_nodes, self.num_nodes), dtype=np.float32)
        batch_edges_target = np.zeros((batch_size, self.num_nodes, self.num_nodes), dtype=np.float32)
        batch_nodes = np.ones((batch_size, self.num_nodes), dtype=np.float32)  # All ones for TSP
        batch_nodes_target = np.zeros((batch_size, self.num_nodes), dtype=np.float32)
        batch_tour_nodes = np.zeros((batch_size, self.num_nodes), dtype=np.int64)
        batch_tour_len = np.zeros(batch_size, dtype=np.float32)

        for b, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 2 * self.num_nodes:
                raise ValueError(f"Line {b + 1} is malformed. Expected at least {2 * self.num_nodes} coordinates.")

            # Extract node coordinates
            coords = np.array(parts[:2 * self.num_nodes], dtype=np.float32).reshape(self.num_nodes, 2)
            nodes_coord[b] = coords

            # Compute midpoints
            x_mid = coords[:, 0].mean()
            y_mid = coords[:, 1].mean()
            batch_x_mid[b] = x_mid
            batch_y_mid[b] = y_mid

            # Assign quadrants
            quadrants = np.ones(self.num_nodes, dtype=int)
            quadrants[coords[:, 0] < x_mid] = 2
            quadrants[coords[:, 1] < y_mid] = 4
            quadrants[(coords[:, 0] < x_mid) & (coords[:, 1] < y_mid)] = 3

            # Compute distance matrix
            W_val = squareform(pdist(coords, metric='euclidean'))
            batch_edges_values[b] = W_val

            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)
            else:
                W = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
                knns = np.argpartition(W_val, self.num_neighbors, axis=1)[:, :self.num_neighbors]
                W[np.arange(self.num_nodes)[:, None], knns] = 1
            np.fill_diagonal(W, 2)  # Self-connections
            batch_edges[b] = W

            # Extract tour nodes
            try:
                output_idx = parts.index('output')
                tour_nodes = [int(node) - 1 for node in parts[output_idx + 1:] if node.isdigit()]
                if len(tour_nodes) != self.num_nodes + 1 or tour_nodes[0] != tour_nodes[-1]:
                    raise ValueError
                tour_nodes = tour_nodes[:-1]
            except (ValueError, IndexError):
                raise ValueError(f"Error processing line {b + 1}: Invalid tour information.")
            batch_tour_nodes[b] = tour_nodes

            # Compute node and edge targets
            nodes_target = np.arange(self.num_nodes, dtype=np.float32)
            batch_nodes_target[b] = nodes_target
            tour_len = 0.0
            for idx in range(self.num_nodes):
                current = tour_nodes[idx]
                next_node = tour_nodes[(idx + 1) % self.num_nodes]
                if quadrants[current] != quadrants[next_node]:
                    batch_edges_target[b, current, next_node] = 1
                    batch_edges_target[b, next_node, current] = 1  # Undirected
                    tour_len += W_val[current, next_node]
            batch_tour_len[b] = tour_len

        return DotDict(
            edges=batch_edges,
            edges_values=batch_edges_values,
            edges_target=batch_edges_target,
            nodes=batch_nodes,
            nodes_target=batch_nodes_target,
            nodes_coord=nodes_coord,
            tour_nodes=batch_tour_nodes,
            tour_len=batch_tour_len,
            x_mid=batch_x_mid,
            y_mid=batch_y_mid
        )

# Define Batch Normalization layers
class BatchNormNode(nn.Module):
    """Batch normalization for node features."""

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        # x: (B, V, H)
        x = x.transpose(1, 2).contiguous()  # (B, H, V)
        x = self.batch_norm(x)  # Apply BatchNorm1d on H
        x = x.transpose(1, 2).contiguous()  # (B, V, H)
        return x

class BatchNormEdge(nn.Module):
    """Batch normalization for edge features."""

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        # e: (B, V, V, H)
        e = e.permute(0, 3, 1, 2).contiguous()  # (B, H, V, V)
        e = self.batch_norm(e)  # Apply BatchNorm2d on H
        e = e.permute(0, 2, 3, 1).contiguous()  # (B, V, V, H)
        return e

# Define MLP layer
class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction."""

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(L - 1)]
        self.U = nn.ModuleList(layers)
        self.V = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.U:
            x = F.relu(layer(x))
        return self.V(x)

# Define NodeFeatures
class NodeFeatures(nn.Module):
    """Convnet features for nodes."""

    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_gate):
        # x: (B, V, H)
        # edge_gate: (B, V, V, H)
        Ux = self.U(x)  # (B, V, H)
        Vx = self.V(x).unsqueeze(1)  # (B, 1, V, H)
        gateVx = edge_gate * Vx  # (B, V, V, H)
        if self.aggregation == "mean":
            aggregation = gateVx.sum(dim=2) / (edge_gate.sum(dim=2) + 1e-20)  # (B, V, H)
        else:
            aggregation = gateVx.sum(dim=2)  # (B, V, H)
        return Ux + aggregation  # (B, V, H)

# Define EdgeFeatures
class EdgeFeatures(nn.Module):
    """Convnet features for edges."""

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, e):
        # x: (B, V, H)
        # e: (B, V, V, H)
        Ue = self.U(e)  # (B, V, V, H)
        Vx_i = self.V(x).unsqueeze(2)  # (B, V, 1, H)
        Vx_j = self.V(x).unsqueeze(1)  # (B, 1, V, H)
        return Ue + Vx_i + Vx_j  # (B, V, V, H)

# Define ResidualGatedGCNLayer
class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection."""

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        # x: (B, V, H)
        # e: (B, V, V, H)
        e_new = self.edge_feat(x, e)  # (B, V, V, H)
        edge_gate = torch.sigmoid(e_new)  # (B, V, V, H)
        x_new = self.node_feat(x, edge_gate)  # (B, V, H)
        e_new = self.bn_edge(e_new)  # (B, V, V, H)
        x_new = self.bn_node(x_new)  # (B, V, H)
        e_new = F.relu(e_new)  # (B, V, V, H)
        x_new = F.relu(x_new)  # (B, V, H)
        return x + x_new, e + e_new  # Residual connections

# Define ResidualGatedGCNModel
class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for predicting edge adjacency matrices."""

    def __init__(self, config):
        super(ResidualGatedGCNModel, self).__init__()
        self.num_nodes = config.num_nodes
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.aggregation = config.aggregation

        # Node coordinate embedding
        self.nodes_coord_embedding = nn.Linear(config.node_dim, self.hidden_dim, bias=False)

        # Edge value and edge type embedding
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(config.voc_edges_in, self.hidden_dim // 2)

        # GCN Layers
        self.gcn_layers = nn.ModuleList([
            ResidualGatedGCNLayer(self.hidden_dim, self.aggregation) for _ in range(self.num_layers)
        ])

        # MLP classifier
        self.mlp_edges = MLP(self.hidden_dim, config.voc_edges_out, config.mlp_layers)

    def loss_edges(self, y_pred_edges, y_edges, edge_cw):
        """Compute loss for edge predictions."""
        y = F.log_softmax(y_pred_edges, dim=3)  # (B, V, V, O)
        y = y.permute(0, 3, 1, 2).contiguous()  # (B, O, V, V)
        return nn.NLLLoss(weight=edge_cw)(y, y_edges)

    def forward(self, x_edges, x_edges_values, x_nodes_coord):
        # Embeddings
        x = self.nodes_coord_embedding(x_nodes_coord)  # (B, V, H)
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(-1))  # (B, V, V, H//2)
        e_tags = self.edges_embedding(x_edges)  # (B, V, V, H//2)
        e = torch.cat((e_vals, e_tags), dim=-1)  # (B, V, V, H)

        # GCN layers
        for layer in self.gcn_layers:
            x, e = layer(x, e)  # (B, V, H), (B, V, V, H)

        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # (B, V, V, O)
        return y_pred_edges

# Define plotting function
def plot_tsp_ground_truth_prediction_classifier(nodes_coord, edges_target, edges_pred_probs, edges_classified, x_mid, y_mid, title="TSP Visualization"):
    """Plot ground truth edges, prediction heatmap, and classifier output for a TSP instance."""
    num_nodes = nodes_coord.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.Reds

    # Ground Truth Edges
    ax = axes[0]
    ax.set_title("Ground Truth Edges", fontsize=16)
    ax.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', s=50, zorder=5)
    lines = [[nodes_coord[i], nodes_coord[j]] for i in range(num_nodes) for j in range(i+1, num_nodes) if edges_target[i, j] == 1]
    if lines:
        lc = LineCollection(lines, colors='red', linewidths=2, alpha=0.8)
        ax.add_collection(lc)
    ax.axvline(x=x_mid, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=y_mid, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("X Coordinate", fontsize=14)
    ax.set_ylabel("Y Coordinate", fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')

    # Prediction Heatmap
    ax = axes[1]
    ax.set_title("Prediction Heatmap", fontsize=16)
    ax.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', s=50, zorder=5)
    lines = []
    colors = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            lines.append([nodes_coord[i], nodes_coord[j]])
            colors.append(edges_pred_probs[i, j])
    lines = np.array(lines)
    colors = np.array(colors)
    sorted_indices = np.argsort(colors)
    lc = LineCollection(lines[sorted_indices], cmap=cmap, norm=norm, linewidths=1.5)
    lc.set_array(colors[sorted_indices])
    lc.set_alpha(0.8)
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Edge Probability', fontsize=14)
    ax.axvline(x=x_mid, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=y_mid, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("X Coordinate", fontsize=14)
    ax.set_ylabel("Y Coordinate", fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')

    # Classifier Output Edges
    ax = axes[2]
    ax.set_title("Classifier Output Edges", fontsize=16)
    ax.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', s=50, zorder=5)
    lines = [[nodes_coord[i], nodes_coord[j]] for i in range(num_nodes) for j in range(i+1, num_nodes) if edges_classified[i, j] == 1]
    if lines:
        lc = LineCollection(lines, colors='green', linewidths=2, alpha=0.8)
        ax.add_collection(lc)
    ax.axvline(x=x_mid, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=y_mid, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("X Coordinate", fontsize=14)
    ax.set_ylabel("Y Coordinate", fontsize=14)
    ax.grid(True)
    ax.set_aspect('equal')

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    return fig  # Return the figure for TensorBoard logging

# Define Edge Classifier
def classify_edges(probs, threshold=0.99, max_edges=2):
    """
    Classify edges based on probability threshold and degree constraints.

    Args:
        probs (torch.Tensor): Edge probabilities, shape (B, V, V)
        threshold (float): Probability threshold to classify an edge as True
        max_edges (int): Maximum number of edges per node

    Returns:
        torch.Tensor: Binary tensor indicating classified edges, shape (B, V, V)
    """
    # Apply threshold
    binary = (probs > threshold).float()

    # Ensure symmetry
    binary = torch.max(binary, binary.transpose(1, 2))

    # Remove self-loops using masking
    V = binary.size(1)
    mask = 1 - torch.eye(V, device=probs.device).unsqueeze(0)  # (1, V, V)
    binary = binary * mask  # (B, V, V)

    # Get topk edges
    topk_values, topk_indices = probs.topk(max_edges, dim=2, largest=True, sorted=True)  # (B, V, max_edges)

    # Create mask for topk
    mask_topk = torch.zeros_like(binary)
    B, V, K = topk_indices.shape
    mask_topk.scatter_(2, topk_indices, 1.0)  # Scatter 1s at topk indices

    # Apply mask and threshold
    binary = binary * mask_topk  # (B, V, V)

    # Ensure symmetry again
    binary = torch.max(binary, binary.transpose(1, 2))

    return binary

# Define Precision and Recall Computation
def compute_precision_recall(pred, target):
    """
    Compute precision and recall.

    Args:
        pred (torch.Tensor): Predicted edges, binary tensor, shape (B, V, V)
        target (torch.Tensor): Ground truth edges, binary tensor, shape (B, V, V)

    Returns:
        tuple: (precision, recall)
    """
    # Get the number of nodes
    V = pred.size(1)
    
    # Create a mask to remove self-loops
    mask = 1 - torch.eye(V, device=pred.device).unsqueeze(0)  # Shape: (1, V, V)
    
    # Apply the mask to both pred and target to remove self-loops
    pred = pred * mask  # Shape: (B, V, V)
    target = target * mask  # Shape: (B, V, V)
    
    # Ensure symmetry
    pred = torch.max(pred, pred.transpose(1, 2))
    target = torch.max(target, target.transpose(1, 2))
    
    # Compute True Positives, False Positives, and False Negatives
    TP = (pred * target).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()
    
    # Compute Precision and Recall with epsilon to avoid division by zero
    precision = TP / (TP + FP + 1e-20)
    recall = TP / (TP + FN + 1e-20)
    
    return precision, recall

# Define training function
def train_one_epoch(net, optimizer, config, writer, epoch, logger=None):
    """Train the model for one epoch."""
    net.train()
    dataset = GoogleTSPReader(config.num_nodes, config.num_neighbors, config.batch_size, config.train_filepath)
    batches_per_epoch = min(config.batches_per_epoch, dataset.max_iter) if config.batches_per_epoch != -1 else dataset.max_iter
    dataset = iter(dataset)
    edge_cw = None
    running_loss = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_nb_data = 0
    start_epoch = time.time()

    progress_bar = tqdm(range(batches_per_epoch), desc='Training', unit='batch', file=sys.stderr)
    for batch_num in progress_bar:
        try:
            batch = next(dataset)
        except StopIteration:
            break

        # Move data to device
        x_edges = torch.tensor(batch.edges, dtype=torch.long, device=device)
        x_edges_values = torch.tensor(batch.edges_values, dtype=torch.float, device=device)
        x_nodes_coord = torch.tensor(batch.nodes_coord, dtype=torch.float, device=device)
        y_edges = torch.tensor(batch.edges_target, dtype=torch.long, device=device)

        # Compute class weights once
        if edge_cw is None:
            edge_labels = y_edges.view(-1).cpu().numpy()
            edge_cw_np = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            edge_cw = torch.tensor(edge_cw_np, dtype=torch.float, device=device)

        # Forward pass
        y_pred_edges = net(x_edges, x_edges_values, x_nodes_coord)

        # Compute loss
        if isinstance(net, nn.DataParallel):
            loss = net.module.loss_edges(y_pred_edges, y_edges, edge_cw)
        else:
            loss = net.loss_edges(y_pred_edges, y_edges, edge_cw)

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()
        optimizer.zero_grad()

        # Classification
        y_pred_probs = F.softmax(y_pred_edges, dim=3)[:, :, :, 1]  # (B, V, V)
        y_classified = classify_edges(y_pred_probs, threshold=0.99, max_edges=2)  # (B, V, V)

        # Compute precision and recall
        precision, recall = compute_precision_recall(y_classified, y_edges.float())

        # Update running metrics
        running_loss += loss.item() * config.batch_size
        running_precision += precision * config.batch_size
        running_recall += recall * config.batch_size
        running_nb_data += config.batch_size

        # Logging to TensorBoard every 100 batches
        if (batch_num + 1) % 100 == 0 or (batch_num + 1) == batches_per_epoch:
            current_loss = running_loss / running_nb_data
            current_precision = running_precision / running_nb_data
            current_recall = running_recall / running_nb_data
            global_step = epoch * batches_per_epoch + batch_num
            writer.add_scalar('Train/Loss', current_loss, global_step)
            writer.add_scalar('Train/Precision', current_precision, global_step)
            writer.add_scalar('Train/Recall', current_recall, global_step)

            progress_bar.set_postfix({'Loss': f'{current_loss:.4f}', 'Precision': f'{current_precision:.4f}', 'Recall': f'{current_recall:.4f}'})

    avg_loss = running_loss / running_nb_data
    avg_precision = running_precision / running_nb_data
    avg_recall = running_recall / running_nb_data
    epoch_time = time.time() - start_epoch

    # Log to TensorBoard at the end of the epoch
    writer.add_scalar('Train/Average_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Average_Precision', avg_precision, epoch)
    writer.add_scalar('Train/Average_Recall', avg_recall, epoch)

    return epoch_time, avg_loss, avg_precision, avg_recall

# Define evaluation function
def evaluate(net, config, writer, epoch, mode='test'):
    """Evaluate the model on validation or test set."""
    net.eval()
    filepath = config.val_filepath if mode == 'val' else config.test_filepath
    dataset = GoogleTSPReader(config.num_nodes, config.num_neighbors, config.batch_size, filepath)
    batches = min(config.batches_per_epoch, dataset.max_iter) if config.batches_per_epoch != -1 else dataset.max_iter
    dataset = iter(dataset)
    edge_cw = None
    running_loss = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_nb_data = 0
    start_time = time.time()

    progress_bar = tqdm(range(batches), desc=f'Evaluating ({mode})', unit='batch', file=sys.stderr)
    with torch.no_grad():
        for batch_num in progress_bar:
            try:
                batch = next(dataset)
            except StopIteration:
                break

            # Move data to device
            x_edges = torch.tensor(batch.edges, dtype=torch.long, device=device)
            x_edges_values = torch.tensor(batch.edges_values, dtype=torch.float, device=device)
            x_nodes_coord = torch.tensor(batch.nodes_coord, dtype=torch.float, device=device)
            y_edges = torch.tensor(batch.edges_target, dtype=torch.long, device=device)

            # Compute class weights once
            if edge_cw is None:
                edge_labels = y_edges.view(-1).cpu().numpy()
                edge_cw_np = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
                edge_cw = torch.tensor(edge_cw_np, dtype=torch.float, device=device)

            # Forward pass
            y_pred_edges = net(x_edges, x_edges_values, x_nodes_coord)

            # Compute loss
            if isinstance(net, nn.DataParallel):
                loss = net.module.loss_edges(y_pred_edges, y_edges, edge_cw)
            else:
                loss = net.loss_edges(y_pred_edges, y_edges, edge_cw)

            # Classification
            y_pred_probs = F.softmax(y_pred_edges, dim=3)[:, :, :, 1]  # (B, V, V)
            y_classified = classify_edges(y_pred_probs, threshold=0.99, max_edges=2)  # (B, V, V)

            # Compute precision and recall
            precision, recall = compute_precision_recall(y_classified, y_edges.float())

            # Update running metrics
            running_loss += loss.item() * config.batch_size
            running_precision += precision * config.batch_size
            running_recall += recall * config.batch_size
            running_nb_data += config.batch_size

            # Logging to TensorBoard every 100 batches
            if (batch_num + 1) % 100 == 0 or (batch_num + 1) == batches:
                current_loss = running_loss / running_nb_data
                current_precision = running_precision / running_nb_data
                current_recall = running_recall / running_nb_data
                global_step = epoch * batches + batch_num
                writer.add_scalar(f'{mode.capitalize()}/Loss', current_loss, global_step)
                writer.add_scalar(f'{mode.capitalize()}/Precision', current_precision, global_step)
                writer.add_scalar(f'{mode.capitalize()}/Recall', current_recall, global_step)

                progress_bar.set_postfix({'Loss': f'{current_loss:.4f}', 'Precision': f'{current_precision:.4f}', 'Recall': f'{current_recall:.4f}'})

    avg_loss = running_loss / running_nb_data
    avg_precision = running_precision / running_nb_data
    avg_recall = running_recall / running_nb_data
    eval_time = time.time() - start_time

    # Log to TensorBoard at the end of the evaluation
    writer.add_scalar(f'{mode.capitalize()}/Average_Loss', avg_loss, epoch)
    writer.add_scalar(f'{mode.capitalize()}/Average_Precision', avg_precision, epoch)
    writer.add_scalar(f'{mode.capitalize()}/Average_Recall', avg_recall, epoch)

    return eval_time, avg_loss, avg_precision, avg_recall

# Define learning rate update function
def update_learning_rate(optimizer, lr):
    """Update the learning rate for the optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# ... [Previous code remains unchanged] ...

def main():
    # Initialize the TensorBoard writer
    writer = SummaryWriter(log_dir='runs/gnn_training')

    # Define configurations
    config = DotDict(
        num_nodes=50,
        num_neighbors=20,
        train_filepath="tsp-data/tsp50_train_concorde.txt",
        val_filepath="tsp-data/tsp50_val_concorde.txt",
        test_filepath="tsp-data/tsp50_test_concorde.txt",
        node_dim=2,
        voc_edges_in=3,
        voc_edges_out=2,
        hidden_dim=300,
        num_layers=30,
        mlp_layers=3,
        aggregation='mean',
        batch_size=40,
        learning_rate=0.001,
        decay_rate=1.01,
        max_epochs=4500,
        batches_per_epoch=500,
        accumulation_steps=1,
        val_every=5,
        test_every=25
    )

    # Log device information to TensorBoard
    device_info = f'Using device: {device}, {torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"}'
    writer.add_text('Device_Info', device_info, 0)
    print(device_info)

    # Initialize the model
    net = ResidualGatedGCNModel(config).to(device)

    # Utilize multiple GPUs if available
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        gpu_info = f"Model is using {torch.cuda.device_count()} GPUs with DataParallel."
        writer.add_text('GPU_Info', gpu_info, 0)
        print(gpu_info)
    else:
        gpu_info = "Model is using a single GPU or CPU."
        writer.add_text('GPU_Info', gpu_info, 0)
        print(gpu_info)

    # Log number of parameters to TensorBoard
    nb_param = sum(p.numel() for p in net.parameters())
    param_info = f'Number of parameters: {nb_param}'
    writer.add_text('Model_Info', param_info, 0)
    print(param_info)

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    val_loss_old = None

    # Initialize loss and metric trackers
    train_losses = []
    train_precisions = []
    train_recalls = []
    val_losses = []
    val_precisions = []
    val_recalls = []
    test_losses = []
    test_precisions = []
    test_recalls = []

    # Epoch loop
    for epoch in trange(1, config.max_epochs + 1, desc='Epochs', unit='epoch'):
        # Train
        train_time, train_loss, train_precision, train_recall = train_one_epoch(net, optimizer, config, writer, epoch)
        train_losses.append(train_loss)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")

        # Validation
        if epoch % config.val_every == 0 or epoch == config.max_epochs:
            val_time, val_loss, val_precision, val_recall = evaluate(net, config, writer, epoch, mode='val')
            val_losses.append(val_loss)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            print(f"Epoch: {epoch}, Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

            # Adjust learning rate if validation loss doesn't improve
            if val_loss_old is not None and val_loss > 0.99 * val_loss_old:
                config.learning_rate /= config.decay_rate
                optimizer = update_learning_rate(optimizer, config.learning_rate)
                lr_info = f"Learning rate updated to {config.learning_rate:.6f}"
                writer.add_text('Learning_Rate', lr_info, epoch)
                print(lr_info)
            val_loss_old = val_loss

        # Testing
        if epoch % config.test_every == 0 or epoch == config.max_epochs:
            test_time, test_loss, test_precision, test_recall = evaluate(net, config, writer, epoch, mode='test')
            test_losses.append(test_loss)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)
            print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

    net.eval()

    # Visualization of test samples
    num_samples = 2
    test_dataset = GoogleTSPReader(config.num_nodes, config.num_neighbors, 1, config.test_filepath)
    test_iter = iter(test_dataset)

    for sample_idx in range(num_samples):
        try:
            batch = next(test_iter)
        except StopIteration:
            print("No more samples in the test dataset.")
            break

        # Move data to device
        x_edges = torch.tensor(batch.edges, dtype=torch.long, device=device)
        x_edges_values = torch.tensor(batch.edges_values, dtype=torch.float, device=device)
        x_nodes_coord = torch.tensor(batch.nodes_coord, dtype=torch.float, device=device)
        y_edges = torch.tensor(batch.edges_target, dtype=torch.long, device=device)

        # Forward pass
        with torch.no_grad():
            y_pred_edges = net(x_edges, x_edges_values, x_nodes_coord)

        # Convert predictions to probabilities
        y_pred_probs = F.softmax(y_pred_edges, dim=3)[:, :, :, 1]  # (B, V, V)
        y_classified = classify_edges(y_pred_probs, threshold=0.99, max_edges=2)  # (B, V, V)

        # Convert tensors to numpy
        y_pred_probs_np = y_pred_probs.squeeze(0).cpu().numpy()  # (V, V)
        y_classified_np = y_classified.squeeze(0).cpu().numpy()  # (V, V)
        y_edges_np = y_edges.squeeze(0).cpu().numpy()  # (V, V)
        nodes_coord_np = x_nodes_coord.squeeze(0).cpu().numpy()  # (V, 2)
        x_mid = batch.x_mid[0]
        y_mid = batch.y_mid[0]

        # Plot and save
        plot_title = f"TSP Sample {sample_idx + 1}"
        fig = plot_tsp_ground_truth_prediction_classifier(
            nodes_coord=nodes_coord_np,
            edges_target=y_edges_np,
            edges_pred_probs=y_pred_probs_np,
            edges_classified=y_classified_np,
            x_mid=x_mid,
            y_mid=y_mid,
            title=plot_title
        )
        plot_filename = f'./test_inference_{sample_idx + 1}.png'
        plt.savefig(plot_filename)
        plt.close()

        # Log the plots to TensorBoard
        img = plt.imread(plot_filename)
        writer.add_image(f'Test/Sample_{sample_idx + 1}', img, dataformats='HWC')

    # Plot loss and metric curves
    epochs = range(1, config.max_epochs + 1)
    val_epochs = [epoch for epoch in range(1, config.max_epochs + 1) if epoch % config.val_every == 0 or epoch == config.max_epochs]
    test_epochs_plot = [epoch for epoch in range(1, config.max_epochs + 1) if epoch % config.test_every == 0 or epoch == config.max_epochs]

    plt.figure(figsize=(15, 10))

    # Loss Plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(val_epochs, val_losses, 'ro-', label='Validation Loss')
    plt.plot(test_epochs_plot, test_losses, 'go-', label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Precision and Recall Plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_precisions, 'b*-', label='Training Precision')
    plt.plot(epochs, train_recalls, 'b.-', label='Training Recall')
    plt.plot(val_epochs, val_precisions, 'r*-', label='Validation Precision')
    plt.plot(val_epochs, val_recalls, 'r.-', label='Validation Recall')
    plt.plot(test_epochs_plot, test_precisions, 'g*-', label='Test Precision')
    plt.plot(test_epochs_plot, test_recalls, 'g.-', label='Test Recall')
    plt.title('Precision and Recall Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.tight_layout()
    metrics_plot_filename = './metrics_hist.png'
    plt.savefig(metrics_plot_filename)
    plt.close()

    # Log the metrics plots to TensorBoard
    metrics_img = plt.imread(metrics_plot_filename)
    writer.add_image('Metrics/Loss_Precision_Recall', metrics_img, dataformats='HWC')

    # **Save the Model After Training**
    # Ensure the /models directory exists
    os.makedirs('models', exist_ok=True)

    # Define the model save path
    model_save_path = os.path.join('models', 'model_final.pth')

    # Save the model's state_dict
    if isinstance(net, nn.DataParallel):
        torch.save(net.module.state_dict(), model_save_path)
    else:
        torch.save(net.state_dict(), model_save_path)

    print(f"Model saved to {model_save_path}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()