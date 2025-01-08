#!/usr/bin/env python3

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # For remote plotting without GUI
import matplotlib.pyplot as plt

#########################################
# ARGPARSE FOR GPU ID
#########################################
parser = argparse.ArgumentParser(description="TSP GNN Training Script")
parser.add_argument("--gpu_id", type=int, default=0,
                    help="Index of the single GPU to use, e.g. 0")
args = parser.parse_args()

gpu_id = args.gpu_id

##################################################
# GPU SELECTION
##################################################
if torch.cuda.is_available():
    device_str = f"cuda:{gpu_id}"
else:
    device_str = "cpu"
device = torch.device(device_str)
print(f"Using device: {device}")

##################################################
# PARSE TSP FILE (PER-INSTANCE x_mid, y_mid)
##################################################

def parse_tsp_file(file_path):
    """
    Reads a .txt file where each line represents an N-city TSP instance.
    Format (single line):
      x1 y1 x2 y2 ... xN yN output r1 r2 ... rN r1

    Returns list of dict, each containing:
      - 'points': list of (x,y)
      - 'solution_edges': set((u,v)) 0-based edges
      - 'x_mid', 'y_mid': medians for that instance
    """
    instances = []
    print(f"Parsing file: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Parsing lines"):
        parts = line.strip().split()
        if "output" not in parts:
            continue

        output_index = parts.index("output")
        coord_values = parts[:output_index]
        route_values = parts[output_index + 1:]

        if len(coord_values) % 2 != 0:
            raise ValueError("Number of coordinate values must be even (x, y pairs).")

        N = len(coord_values) // 2
        points = [(float(coord_values[2*i]), float(coord_values[2*i+1])) for i in range(N)]

        x_mid = np.median([p[0] for p in points])
        y_mid = np.median([p[1] for p in points])

        route = list(map(int, route_values))
        if len(route) != N + 1 or route[0] != route[-1]:
            raise ValueError("Route must have N+1 entries and start/end with the same city.")

        sol_edges = set()
        for i in range(len(route) - 1):
            u = route[i] - 1
            v = route[i+1] - 1
            if not (0 <= u < N and 0 <= v < N):
                raise ValueError(f"Invalid route index {u},{v}. Must be between 1 and N.")
            if u != v:
                sol_edges.add((u, v))

        instances.append({
            'points': points,
            'solution_edges': sol_edges,
            'x_mid': x_mid,
            'y_mid': y_mid
        })
    return instances

##################################################
# HELPER FUNCTIONS
##################################################
from scipy.spatial import cKDTree

def quadrant(x, y, x_mid, y_mid):
    if x >= x_mid and y >= y_mid:
        return 0
    elif x < x_mid and y >= y_mid:
        return 1
    elif x < x_mid and y < y_mid:
        return 2
    else:
        return 3

def build_kd_tree(points):
    return cKDTree(points)

def get_k_neighbors(tree, query_point, k):
    dist, idx = tree.query(query_point, k=k)
    if k == 1:
        dist = np.array([dist])
        idx = np.array([idx])
    return dist, idx

def compute_instance_thresholds(points, x_mid, y_mid, percentile=20):
    dx_vals = [abs(x - x_mid) for (x,y) in points]
    dy_vals = [abs(y - y_mid) for (x,y) in points]
    d_x = np.percentile(dx_vals, percentile)
    d_y = np.percentile(dy_vals, percentile)
    return d_x, d_y

def is_axis_close(x, y, x_mid, y_mid, d_x, d_y):
    return (abs(x - x_mid) <= d_x) or (abs(y - y_mid) <= d_y)

def compute_instance_dist_threshold(points, x_mid, y_mid, d_x, d_y, percentile=90):
    axis_close_mask = [
        is_axis_close(x, y, x_mid, y_mid, d_x, d_y)
        for (x,y) in points
    ]
    axis_close_indices = [i for i, flag in enumerate(axis_close_mask) if flag]

    if len(axis_close_indices) < 2:
        return float('inf')

    coords = np.array(points)
    axis_close_coords = coords[axis_close_indices]
    dist_list = []
    M = axis_close_coords.shape[0]
    for i in range(M):
        for j in range(i+1, M):
            dx = axis_close_coords[j,0] - axis_close_coords[i,0]
            dy = axis_close_coords[j,1] - axis_close_coords[i,1]
            dist_ij = np.sqrt(dx*dx + dy*dy)
            dist_list.append(dist_ij)

    if len(dist_list) == 0:
        return float('inf')

    dist_threshold = np.percentile(dist_list, percentile)
    return dist_threshold

def build_graph_instance(points, x_mid, y_mid, d_x, d_y, k=3,
                         solution_edges=None, add_solution_edges=False,
                         dist_threshold=float('inf')):
    import numpy as np
    from torch_geometric.data import Data

    coords = np.array(points)
    N = len(points)
    quads = [quadrant(x, y, x_mid, y_mid) for (x,y) in coords]

    axis_close_mask = [
        is_axis_close(x, y, x_mid, y_mid, d_x, d_y)
        for (x,y) in coords
    ]
    if solution_edges is not None and add_solution_edges:
        for (u, v) in solution_edges:
            if quads[u] != quads[v]:
                axis_close_mask[u] = True
                axis_close_mask[v] = True

    axis_close_indices = np.where(axis_close_mask)[0]

    quad_points = [[] for _ in range(4)]
    quad_indices = [[] for _ in range(4)]
    for i, (x,y) in enumerate(coords):
        Q = quads[i]
        quad_points[Q].append((x,y))
        quad_indices[Q].append(i)

    quad_points = [np.array(qp) if len(qp)>0 else np.zeros((0,2)) for qp in quad_points]
    quad_trees = [build_kd_tree(qp) if len(qp)>0 else None for qp in quad_points]

    edge_src, edge_dst, edge_labels = [], [], []

    for global_idx in axis_close_indices:
        px, py = coords[global_idx]
        Qsrc = quads[global_idx]
        for Qtgt in range(4):
            if Qtgt == Qsrc:
                continue
            if quad_trees[Qtgt] is None or len(quad_points[Qtgt]) == 0:
                continue

            k_eff = min(k, len(quad_points[Qtgt]))
            if k_eff == 0:
                continue

            dist_arr, idx_arr = get_k_neighbors(quad_trees[Qtgt], [px, py], k_eff)
            for neigh_i in range(k_eff):
                nbr_global = quad_indices[Qtgt][idx_arr[neigh_i]]
                d_ij = dist_arr[neigh_i]
                if d_ij > dist_threshold:
                    continue

                edge_src.append(global_idx)
                edge_dst.append(nbr_global)
                if solution_edges is not None:
                    if (global_idx, nbr_global) in solution_edges or (nbr_global, global_idx) in solution_edges:
                        label = 1.0
                    else:
                        label = 0.0
                else:
                    label = 0.0
                edge_labels.append(label)

                edge_src.append(nbr_global)
                edge_dst.append(global_idx)
                edge_labels.append(label)

    x_t = torch.tensor(coords, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y_t = torch.tensor(edge_labels, dtype=torch.float).view(-1,1)

    data = Data(x=x_t, edge_index=edge_index, y=y_t)
    return data

##################################################
# DEEP GNN MODEL
##################################################
class EdgeClassifierGNN(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=256, num_layers=15):
        super().__init__()
        self.num_layers = num_layers

        # Pre-transform input from (N,2) -> (N,256) for residual
        self.input_fc = nn.Linear(in_channels, hidden_dim, bias=False)

        # GraphConv + BatchNorm layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GraphConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        x = self.input_fc(x)  # (N,2)->(N,256)
        for i in range(self.num_layers):
            h = self.convs[i](x, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            x = x + h  # residual
        row, col = edge_index
        edge_emb = torch.cat([x[row], x[col]], dim=-1)
        out = self.mlp(edge_emb)
        return out

##################################################
# TRAIN & EVAL
##################################################

def train_epoch(model, train_data_list, optimizer, criterion, device, epoch_num):
    """
    Modified training procedure:
      - Randomly sample 10k from the million training instances
      - Split into 500 mini-batches of 20 each
      - Optimize with Adam(lr=0.001) on each mini-batch
    """
    model.train()
    total_loss = 0

    # 1) Randomly sample 10k from the entire training dataset
    #    (We assume train_data_list has >= 1e6 items.)
    subset_size = 10000
    batch_size = 20
    subset = random.sample(train_data_list, subset_size)

    # 2) Create a loader for those 10k, with mini-batch size=20
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    loader_iter = tqdm(loader, desc=f"Epoch {epoch_num} (Train)")
    for data in loader_iter:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, mode="Val"):
    model.eval()
    total_loss = 0
    loader_iter = tqdm(loader, desc=f"Evaluating ({mode})", leave=False)

    eps = 1e-9
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        for data in loader_iter:
            data = data.to(device)
            out = model(data.x, data.edge_index)
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
    improvement_threshold = 0.01  # 1%
    if new_val_loss >= best_val_loss * (1 - improvement_threshold):
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr / decay_factor
            param_group['lr'] = new_lr
        return new_val_loss, False
    else:
        return new_val_loss, True

##################################################
# MAIN
##################################################

def main():
    problem_size = 50
    train_file = f"tsp-data/tsp{problem_size}_train_concorde.txt"
    val_file   = f"tsp-data/tsp{problem_size}_val_concorde.txt"
    test_file  = f"tsp-data/tsp{problem_size}_test_concorde.txt"

    percentile_axis_close = 20
    percentile_dist = 50
    k_nn = 3
    rebuild_data = False
    epochs = 30

    if rebuild_data:
        train_instances = parse_tsp_file(train_file)
        val_instances   = parse_tsp_file(val_file)
        test_instances  = parse_tsp_file(test_file)

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
                data = build_graph_instance(points, x_mid_i, y_mid_i, d_x_i, d_y_i, k=k_nn,
                                            solution_edges=sol_edges, add_solution_edges=is_train,
                                            dist_threshold=dist_thr_i)
                data_list.append(data)
            return data_list

        train_data_list = build_dataset(train_instances, is_train=True)
        val_data_list   = build_dataset(val_instances,   is_train=False)
        test_data_list  = build_dataset(test_instances,  is_train=False)

        torch.save(train_data_list, f"tsp-data/tsp{problem_size}_train_concorde.pt")
        torch.save(val_data_list,   f"tsp-data/tsp{problem_size}_val_concorde.pt")
        torch.save(test_data_list,  f"tsp-data/tsp{problem_size}_test_concorde.pt")
    else:
        train_data_list = torch.load(f"tsp-data/tsp{problem_size}_train_concorde.pt")
        val_data_list   = torch.load(f"tsp-data/tsp{problem_size}_val_concorde.pt")
        test_data_list  = torch.load(f"tsp-data/tsp{problem_size}_test_concorde.pt")

    # We do NOT create a standard train_loader of all data:
    # We'll do random sampling each epoch in train_epoch

    val_loader   = DataLoader(val_data_list,   batch_size=512,  shuffle=False)
    test_loader  = DataLoader(test_data_list,  batch_size=512,  shuffle=False)

    base_model = EdgeClassifierGNN(in_channels=2, hidden_dim=256, num_layers=10).to(device)
    # Adam with initial LR=0.001
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history   = []
    val_precision_history = []
    val_recall_history    = []
    val_f1_history        = []

    for epoch in range(epochs):
        # Modified train_epoch: random sample 10k from train_data_list, do mini-batches of 20
        train_loss = train_epoch(base_model, train_data_list, optimizer, criterion, device, epoch)
        val_loss, val_prec, val_rec, val_f1 = evaluate(base_model, val_loader, criterion, device, "Val")

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
    test_loss, test_precision, test_recall, test_f1 = evaluate(base_model, test_loader, criterion, device, "Test")
    print(f"Final Test Loss: {test_loss:.4f}, Test Prec: {test_precision:.4f}, "
          f"Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

    # PLOT & SAVE
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
