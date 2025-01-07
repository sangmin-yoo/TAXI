import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

##################################################
# 1. PARSE TSP FILE (PER-INSTANCE x_mid, y_mid)
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
# 2. HELPER FUNCTIONS
##################################################

def quadrant(x, y, x_mid, y_mid):
    # Q0: x >= x_mid, y >= y_mid
    # Q1: x <  x_mid, y >= y_mid
    # Q2: x <  x_mid, y <  y_mid
    # Q3: x >= x_mid, y <  y_mid
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
        idx = np.array([idx])
        dist = np.array([dist])
    return dist, idx

def compute_instance_thresholds(points, x_mid, y_mid, percentile=20):
    """
    For axis closeness:
      d_x, d_y = percentile-based thresholds for |x - x_mid| and |y - y_mid|.
    """
    dx_vals = [abs(x - x_mid) for (x,y) in points]
    dy_vals = [abs(y - y_mid) for (x,y) in points]
    d_x = np.percentile(dx_vals, percentile)
    d_y = np.percentile(dy_vals, percentile)
    return d_x, d_y

def is_axis_close(x, y, x_mid, y_mid, d_x, d_y):
    return (abs(x - x_mid) <= d_x) or (abs(y - y_mid) <= d_y)

##################################################
# 3. DISTANCE-BASED EDGE PRUNING: PERCENTILE THRESHOLD
##################################################

def compute_instance_dist_threshold(points, x_mid, y_mid, d_x, d_y, percentile=90):
    """
    1) Identify axis-close points (just as in build_graph_instance).
    
    2) Collect all pairwise distances among these axis-close points
       (if that set is large, be mindful of O(N^2) complexity).
    3) Return the distance at the specified percentile. 
       e.g., 90 => skip edges above the 90th percentile distance.
    """
    # Step A: basic axis closeness
    axis_close_mask = [
        is_axis_close(x, y, x_mid, y_mid, d_x, d_y)
        for (x,y) in points
    ]
    axis_close_indices = [i for i,flag in enumerate(axis_close_mask) if flag]

    # If no axis-close points, return a large threshold to avoid
    # pruning everything
    if len(axis_close_indices) < 2:
        return float('inf')  # no distance-based pruning possible

    coords = np.array(points)
    axis_close_coords = coords[axis_close_indices]  # shape (M,2)

    # Step B: gather pairwise distances (O(M^2))
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

    # Step C: percentile
    dist_threshold = np.percentile(dist_list, percentile)
    return dist_threshold

##################################################
# 4. BUILD GRAPH (k-NN + Dist Threshold)
##################################################

def build_graph_instance(points, 
                         x_mid, y_mid, 
                         d_x, d_y, 
                         k=3,
                         solution_edges=None,
                         add_solution_edges = False,
                         dist_threshold=float('inf')):
    """
    This function prunes edges in two ways:
      1) Only consider axis-close points as sources (plus endpoints of cross-quadrant solution edges).
      2) Among the k-NN in each quadrant, skip any edge with dist > dist_threshold.
    """
    coords = np.array(points)
    N = len(points)

    quads = [quadrant(x, y, x_mid, y_mid) for (x,y) in points]

    # Identify axis-close points
    axis_close_mask = [
        is_axis_close(x, y, x_mid, y_mid, d_x, d_y)
        for (x,y) in points
    ]

    # Enforce solution cross-quadrant edges
    if solution_edges is not None and add_solution_edges is True:
        for (u, v) in solution_edges:
            if quads[u] != quads[v]:
                axis_close_mask[u] = True
                axis_close_mask[v] = True

    axis_close_indices = np.where(axis_close_mask)[0]

    # kd-trees per quadrant
    quad_points = [[] for _ in range(4)]
    quad_indices = [[] for _ in range(4)]
    for i, (x,y) in enumerate(points):
        Q = quads[i]
        quad_points[Q].append((x,y))
        quad_indices[Q].append(i)

    quad_points = [np.array(qp) if len(qp)>0 else np.zeros((0,2)) for qp in quad_points]
    quad_trees = [
        build_kd_tree(qp) if len(qp)>0 else None
        for qp in quad_points
    ]

    edge_src, edge_dst, edge_labels = [], [], []

    # Build edges from axis-close points -> kNN in other quadrants
    for global_idx in axis_close_indices:
        px, py = points[global_idx]
        Qsrc = quads[global_idx]
        for Qtgt in range(4):
            if Qtgt == Qsrc:
                continue
            if quad_trees[Qtgt] is None or len(quad_points[Qtgt]) == 0:
                continue

            # effective k if quadrant has fewer than k points
            k_eff = min(k, len(quad_points[Qtgt]))
            if k_eff == 0:
                continue

            dist_arr, idx_arr = get_k_neighbors(quad_trees[Qtgt], [px, py], k_eff)
            for neigh_i in range(k_eff):
                nbr_global = quad_indices[Qtgt][idx_arr[neigh_i]]
                d_ij = dist_arr[neigh_i]

                # Skip if distance > dist_threshold
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

                # reverse edge
                edge_src.append(nbr_global)
                edge_dst.append(global_idx)
                edge_labels.append(label)

    x_t = torch.tensor(coords, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y_t = torch.tensor(edge_labels, dtype=torch.float).view(-1,1)

    return Data(x=x_t, edge_index=edge_index, y=y_t)

##################################################
# 5. GNN MODEL WITH SIGMOID OUTPUT
##################################################

class EdgeClassifierGNN(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))

        # We apply Sigmoid directly -> use BCELoss
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        row, col = edge_index
        edge_emb = torch.cat([x[row], x[col]], dim=-1)  # (M,2*hidden_dim)
        out = self.mlp(edge_emb)  # (M,1) in [0,1]
        return out
    
##################################################
# 6. TRAIN & EVALUATE (Using BCELoss)
##################################################

def train_epoch(model, loader, optimizer, criterion, device, epoch_num):
    model.train()
    total_loss = 0
    loader_iter = tqdm(loader, desc=f"Epoch {epoch_num} (Train)")
    for data in loader_iter:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # sigmoid probabilities
        loss = criterion(out, data.y)         # BCELoss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, mode="Val"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    loader_iter = tqdm(loader, desc=f"Evaluating ({mode})", leave=False)
    with torch.no_grad():
        for data in loader_iter:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            preds = (out > 0.5).float()
            correct += (preds == data.y).sum().item()
            total   += data.y.numel()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

##################################################
# 7. MAIN EXAMPLE
##################################################


# Suppose we have TSP files of size 50
problem_size = 50
train_file = f"tsp-data/tsp{problem_size}_train_concorde.txt"
val_file   = f"tsp-data/tsp{problem_size}_val_concorde.txt"
test_file  = f"tsp-data/tsp{problem_size}_test_concorde.txt"

train_instances = parse_tsp_file(train_file)
val_instances   = parse_tsp_file(val_file)
test_instances  = parse_tsp_file(test_file)


# For axis closeness
percentile_axis_close = 20  
# For distance threshold (e.g., skip edges above 90th percentile)
percentile_dist = 50  
k_nn = 3              

def build_dataset(instances, is_train = False):
    data_list = []
    for inst in tqdm(instances, desc="Building Graphs"):
        points = inst['points']
        sol_edges = inst['solution_edges']
        x_mid_i   = inst['x_mid']
        y_mid_i   = inst['y_mid']

        # Step 1: compute d_x, d_y
        d_x_i, d_y_i = compute_instance_thresholds(points, x_mid_i, y_mid_i, percentile_axis_close)

        # Step 2: compute the distance threshold for this instance
        dist_thr_i = compute_instance_dist_threshold(
            points, x_mid_i, y_mid_i, d_x_i, d_y_i,
            percentile=percentile_dist
        )

        # Step 3: build graph
        data = build_graph_instance(
            points,
            x_mid_i, y_mid_i,
            d_x_i, d_y_i,
            k=k_nn,
            solution_edges=sol_edges,
            add_solution_edges=is_train,
            dist_threshold=dist_thr_i
        )
        data_list.append(data)
    return data_list

train_data_list = build_dataset(train_instances, is_train=True)
val_data_list   = build_dataset(val_instances, is_train=False)
test_data_list  = build_dataset(test_instances, is_train=False)

train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data_list,   batch_size=1, shuffle=False)
test_loader  = DataLoader(test_data_list,  batch_size=1, shuffle=False)

# Use MPS or fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

model = EdgeClassifierGNN(in_channels=2, hidden_dim=256, num_layers=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Because final layer is Sigmoid
criterion = nn.BCELoss()

epochs = 10
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device, "Val")
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Final test
test_loss, test_acc = evaluate(model, test_loader, criterion, device, "Test")
print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")