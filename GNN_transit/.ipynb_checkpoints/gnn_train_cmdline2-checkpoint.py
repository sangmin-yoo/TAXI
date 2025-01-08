#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # For remote plotting without GUI
import matplotlib.pyplot as plt

###############################################################################
# ARGPARSE
###############################################################################
parser = argparse.ArgumentParser(description="TSP Transformer with custom train sampling")
parser.add_argument("--gpu_id", type=int, default=0, help="Index of single GPU to use, e.g. 0")
args = parser.parse_args()

gpu_id = args.gpu_id
if torch.cuda.is_available():
    device_str = f"cuda:{gpu_id}"
else:
    device_str = "cpu"
device = torch.device(device_str)
print(f"Using device: {device}")

###############################################################################
# PARSE TSP FILE
###############################################################################
def parse_tsp_file(file_path):
    """
    Reads .txt file: x1 y1 x2 y2 ... xN yN output r1 r2 ... rN r1
    Returns list of dict with:
      - 'points': list of (x,y)
      - 'solution_edges': set((u,v))
      - 'x_mid','y_mid'
    """
    instances = []
    print(f"Parsing file: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    from tqdm import tqdm
    for line in tqdm(lines, desc="Parsing lines"):
        parts = line.strip().split()
        if "output" not in parts:
            continue

        output_index = parts.index("output")
        coord_values = parts[:output_index]
        route_values = parts[output_index + 1:]

        if len(coord_values) % 2 != 0:
            raise ValueError("Coord values must be even (x,y).")

        N = len(coord_values) // 2
        points = [(float(coord_values[2*i]), float(coord_values[2*i+1])) for i in range(N)]

        x_mid = np.median([p[0] for p in points])
        y_mid = np.median([p[1] for p in points])

        route = list(map(int, route_values))
        if len(route) != N + 1 or route[0] != route[-1]:
            raise ValueError("Route must have N+1 entries and start/end with same city.")

        sol_edges = set()
        for i in range(len(route) - 1):
            u = route[i] - 1
            v = route[i+1] - 1
            if u != v:
                sol_edges.add((u, v))

        instances.append({
            'points': points,
            'solution_edges': sol_edges,
            'x_mid': x_mid,
            'y_mid': y_mid
        })
    return instances

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def quadrant(x, y, x_mid, y_mid):
    if x >= x_mid and y >= y_mid: return 0
    elif x < x_mid and y >= y_mid: return 1
    elif x < x_mid and y < y_mid: return 2
    else: return 3

from scipy.spatial import cKDTree

def build_kd_tree(points):
    return cKDTree(points)

def get_k_neighbors(tree, query_point, k):
    dist, idx = tree.query(query_point, k=k)
    if k == 1:
        dist = np.array([dist])
        idx = np.array([idx])
    return dist, idx

def compute_instance_thresholds(points, x_mid, y_mid, percentile=20):
    dx_vals = [abs(px - x_mid) for (px,py) in points]
    dy_vals = [abs(py - y_mid) for (px,py) in points]
    d_x = np.percentile(dx_vals, percentile)
    d_y = np.percentile(dy_vals, percentile)
    return d_x, d_y

def is_axis_close(x, y, x_mid, y_mid, d_x, d_y):
    return (abs(x - x_mid) <= d_x) or (abs(y - y_mid) <= d_y)

def compute_instance_dist_threshold(points, x_mid, y_mid, d_x, d_y, percentile=90):
    axis_close_mask = [is_axis_close(px,py, x_mid, y_mid, d_x, d_y) for (px,py) in points]
    axis_close_indices = [i for i,flag in enumerate(axis_close_mask) if flag]
    if len(axis_close_indices) < 2: return float('inf')

    coords = np.array(points)
    close_coords = coords[axis_close_indices]
    dist_list = []
    M = close_coords.shape[0]
    for i in range(M):
        for j in range(i+1, M):
            dx = close_coords[j,0] - close_coords[i,0]
            dy = close_coords[j,1] - close_coords[i,1]
            dist_ij = np.sqrt(dx*dx + dy*dy)
            dist_list.append(dist_ij)

    if len(dist_list) == 0: return float('inf')
    return np.percentile(dist_list, percentile)

from torch_geometric.data import Data

def build_graph_instance(points, x_mid, y_mid, d_x, d_y, k=3,
                         solution_edges=None, add_solution_edges=False,
                         dist_threshold=float('inf')):
    import numpy as np
    coords = np.array(points)
    N = len(points)
    quads = [quadrant(px,py, x_mid, y_mid) for (px,py) in coords]

    axis_mask = [is_axis_close(px,py, x_mid,y_mid, d_x,d_y) for (px,py) in coords]
    if solution_edges is not None and add_solution_edges:
        for (u,v) in solution_edges:
            if quads[u] != quads[v]:
                axis_mask[u] = True
                axis_mask[v] = True

    axis_indices = np.where(axis_mask)[0]
    quad_points = [[] for _ in range(4)]
    quad_indices= [[] for _ in range(4)]
    for i,(px,py) in enumerate(coords):
        Q = quads[i]
        quad_points[Q].append((px,py))
        quad_indices[Q].append(i)

    from torch_geometric.data import Data
    quad_points = [np.array(qp) if len(qp)>0 else np.zeros((0,2)) for qp in quad_points]
    quad_trees = [build_kd_tree(qp) if len(qp)>0 else None for qp in quad_points]

    edge_src, edge_dst, edge_labels = [], [], []

    for global_idx in axis_indices:
        px, py = coords[global_idx]
        Qsrc = quads[global_idx]
        for Qtgt in range(4):
            if Qtgt == Qsrc: continue
            if quad_trees[Qtgt] is None or len(quad_points[Qtgt]) == 0: continue
            k_eff = min(k, len(quad_points[Qtgt]))
            dist_arr, idx_arr = get_k_neighbors(quad_trees[Qtgt], [px,py], k_eff)
            for neigh_i in range(k_eff):
                nbr_global = quad_indices[Qtgt][idx_arr[neigh_i]]
                d_ij = dist_arr[neigh_i]
                if d_ij > dist_threshold: continue

                lbl = 0.0
                if solution_edges is not None:
                    if ((global_idx,nbr_global) in solution_edges 
                        or (nbr_global,global_idx) in solution_edges):
                        lbl=1.0

                edge_src.append(global_idx)
                edge_dst.append(nbr_global)
                edge_labels.append(lbl)
                edge_src.append(nbr_global)
                edge_dst.append(global_idx)
                edge_labels.append(lbl)

    x_t = torch.tensor(coords, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y_t = torch.tensor(edge_labels, dtype=torch.float).view(-1,1)

    edge_attr = torch.zeros_like(y_t)
    return Data(x=x_t, edge_index=edge_index, y=y_t, edge_attr=edge_attr)

###############################################################################
# A SMALL TRANSFORMER FOR NODE EMBEDDINGS
###############################################################################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,N,H)
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  
        x2 = x + self.dropout(attn_out)
        x2_norm = self.ln2(x2)
        ff_out = self.ffn(x2_norm)
        x3 = x2 + self.dropout(ff_out)
        return x3

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

###############################################################################
# EDGE CLASSIFIER w/ TRANSFORMER
###############################################################################
class EdgeClassifierTransformer(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=64, num_layers=3, num_heads=4):
        super().__init__()
        self.node_embed = nn.Linear(in_channels, hidden_dim)
        self.transformer = TransformerEncoder(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
        self.edge_fc = nn.Linear(hidden_dim, 1)  # final => single logit

    def forward(self, x, edge_index, edge_attr=None):
        device = x.device
        N = x.size(0)
        E = edge_index.size(1)

        x_b = x.unsqueeze(0)              # (1,N,2)
        x_b = self.node_embed(x_b)        # (1,N,H)
        x_b = self.transformer(x_b)       # (1,N,H)

        row, col = edge_index
        row_emb = x_b[0, row, :]
        col_emb = x_b[0, col, :]
        edge_emb = row_emb + col_emb      # (E,H)
        logits = self.edge_fc(edge_emb)   # (E,1)
        probs = torch.sigmoid(logits)     # in [0,1]
        return probs

###############################################################################
# CUSTOM TRAIN EPOCH with random 10k subset => 500 batches of 20
###############################################################################
def train_epoch(model, train_data_list, optimizer, criterion, device, epoch_num):
    model.train()
    total_loss = 0.0

    # 1) Randomly select 10,000 from train_data_list
    #    We assume train_data_list is a huge list of 1 million Data objects
    subset_size = 10000
    batch_size = 20
    # pick random 10k indices
    indices_10k = random.sample(range(len(train_data_list)), subset_size)

    # 2) We'll chunk them into 500 mini-batches (because 10,000 / 20 = 500)
    # shuffle that subset (though random.sample already random)
    # create mini-batches of size 20
    # we'll do for b_idx in range(500): ...
    # extract that chunk, do forward/backward

    # minor detail: the order is random anyway, so we can just chunk in consecutive
    for b_idx in range(500):
        # subrange in indices_10k
        start = b_idx * batch_size
        end = start + batch_size
        batch_indices = indices_10k[start:end]

        # load data
        batch_data = [train_data_list[i].to(device) for i in batch_indices]

        # we do a forward pass for each instance => we can either do them individually or combine
        # Typically for PyG, we might do a 'Batch.from_data_list(batch_data)', but let's do individually for clarity
        # We'll accumulate loss across these 20
        loss_acc = 0.0
        for data in batch_data:
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            loss_acc += loss

        loss_acc = loss_acc / batch_size  # average loss in mini-batch

        optimizer.zero_grad()
        loss_acc.backward()
        optimizer.step()
        total_loss += loss_acc.item()

    # average over 500 steps
    return total_loss / 500.0

###############################################################################
# EVALUATE using standard DataLoader
###############################################################################
def evaluate(model, loader, criterion, device, mode="Val"):
    model.eval()
    total_loss = 0.0

    eps = 1e-9
    TP, FP, TN, FN = 0, 0, 0, 0

    from tqdm import tqdm
    loader_iter = tqdm(loader, desc=f"Evaluating ({mode})", leave=False)

    with torch.no_grad():
        for data in loader_iter:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            preds = (out > 0.5).float()
            y_true = data.y.int()
            y_pred = preds.int()

            TP += int(((y_pred == 1) & (y_true == 1)).sum().item())
            FP += int(((y_pred == 1) & (y_true == 0)).sum().item())
            TN += int(((y_pred == 0) & (y_true == 0)).sum().item())
            FN += int(((y_pred == 0) & (y_true == 1)).sum().item())

    avg_loss = total_loss / len(loader)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return avg_loss, precision, recall, f1

def lr_scheduler(optimizer, new_val_loss, best_val_loss, decay_factor=1.01):
    improvement_threshold = 0.01
    if new_val_loss >= best_val_loss*(1 - improvement_threshold):
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr / decay_factor
            param_group['lr'] = new_lr
        return new_val_loss, False
    else:
        return new_val_loss, True

###############################################################################
# MAIN
###############################################################################
def main():
    problem_size = 50
    train_file = f"tsp-data/tsp{problem_size}_train_concorde.txt"
    val_file   = f"tsp-data/tsp{problem_size}_val_concorde.txt"
    test_file  = f"tsp-data/tsp{problem_size}_test_concorde.txt"

    # typical hyperparams
    percentile_axis_close = 20
    percentile_dist = 50
    k_nn = 3
    rebuild_data = False
    epochs = 30

    if rebuild_data:
        train_instances = parse_tsp_file(train_file)
        val_instances   = parse_tsp_file(val_file)
        test_instances  = parse_tsp_file(test_file)
        from tqdm import tqdm

        def build_dataset(instances, is_train=False):
            data_list = []
            for inst in tqdm(instances, desc="Building Graphs"):
                points = inst['points']
                sol_edges = inst['solution_edges']
                x_mid_i   = inst['x_mid']
                y_mid_i   = inst['y_mid']

                d_x_i, d_y_i = compute_instance_thresholds(points, x_mid_i, y_mid_i, percentile_axis_close)
                dist_thr_i   = compute_instance_dist_threshold(points, x_mid_i, y_mid_i, d_x_i, d_y_i,
                                                               percentile=percentile_dist)

                data = build_graph_instance(points, x_mid_i, y_mid_i, d_x_i, d_y_i,
                                            k=k_nn,
                                            solution_edges=sol_edges,
                                            add_solution_edges=is_train,
                                            dist_threshold=dist_thr_i)
                data_list.append(data)
            return data_list

        train_data_list = build_dataset(train_instances, is_train=True)   # potentially 1 million
        val_data_list   = build_dataset(val_instances,   is_train=False)
        test_data_list  = build_dataset(test_instances,  is_train=False)

        torch.save(train_data_list, f"tsp-data/tsp{problem_size}_train_concorde.pt")
        torch.save(val_data_list,   f"tsp-data/tsp{problem_size}_val_concorde.pt")
        torch.save(test_data_list,  f"tsp-data/tsp{problem_size}_test_concorde.pt")
    else:
        train_data_list = torch.load(f"tsp-data/tsp{problem_size}_train_concorde.pt")
        val_data_list   = torch.load(f"tsp-data/tsp{problem_size}_val_concorde.pt")
        test_data_list  = torch.load(f"tsp-data/tsp{problem_size}_test_concorde.pt")

    # We'll do standard DataLoader for val/test only
    val_loader  = DataLoader(val_data_list,  batch_size=512, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=512, shuffle=False)

    # Build the Transformer model
    from torch.optim import Adam
    model = EdgeClassifierTransformer(in_channels=2, hidden_dim=512, num_layers=10, num_heads=4).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history   = []
    val_precision_history = []
    val_recall_history    = []
    val_f1_history        = []

    for epoch in range(epochs):
        # Custom training procedure:
        # random subset(10k) => 500 mini-batches of 20
        train_loss = train_epoch(model, train_data_list, optimizer, criterion, device, epoch)
        # Evaluate
        val_loss, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device, "Val")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_precision_history.append(val_prec)
        val_recall_history.append(val_rec)
        val_f1_history.append(val_f1)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Prec: {val_prec:.4f}, "
              f"Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            _, improved = lr_scheduler(optimizer, val_loss, best_val_loss, decay_factor=1.01)

    # Final test
    test_loss, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device, "Test")
    print(f"Final Test Loss: {test_loss:.4f}, Test Prec: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    # Plots
    plt.figure(figsize=(8,6))
    plt.title("Train & Validation Loss")
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history,   label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_val_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.title("Validation Metrics")
    plt.plot(val_precision_history, label='Val Precision')
    plt.plot(val_recall_history,    label='Val Recall')
    plt.plot(val_f1_history,        label='Val F1')
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_metrics.png", dpi=150)
    plt.close()

    print("All done. Exiting.")

if __name__ == "__main__":
    main()