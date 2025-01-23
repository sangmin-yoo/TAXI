import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from scipy.spatial.distance import pdist, squareform
from functools import partial
from tqdm import tqdm
import shutil

################################################################################
# Utility structures
################################################################################

class DotDict(dict):
    def __getattr__(self, key):
        return self.get(key)

################################################################################
# Build global vocabulary
################################################################################

def build_global_vocabulary(num_nodes=50):
    """
    Build a canonical ordering of edges (i < j) for up to num_nodes=50.
    Also add special tokens: <PAD>, <START>, <END>
    """
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edges.append((i, j))
    edges.sort(key=lambda x: (x[0], x[1]))
    edge_to_token = {}
    token_to_edge = {}

    for idx, (i, j) in enumerate(edges):
        edge_to_token[(i, j)] = idx
        token_to_edge[idx] = (i, j)
    
    PAD_TOKEN = len(edges)
    START_TOKEN = len(edges) + 1
    END_TOKEN = len(edges) + 2
    
    edge_to_token['<PAD>'] = PAD_TOKEN
    edge_to_token['<START>'] = START_TOKEN
    edge_to_token['<END>'] = END_TOKEN
    
    token_to_edge[PAD_TOKEN] = '<PAD>'
    token_to_edge[START_TOKEN] = '<START>'
    token_to_edge[END_TOKEN] = '<END>'
    
    return edge_to_token, token_to_edge

################################################################################
# Dataset
################################################################################

class TSPDataset(Dataset):
    """
    Expects lines: x1 y1 x2 y2 ... xN yN output 1 2 3 ... N 1
    We parse node coords, compute quadrant splits, build topDown/leftRight edge
    tokens, etc.
    """
    def __init__(self, filepath, num_nodes=50, edge_to_token=None, token_to_edge=None,
                 num_neighbors=-1, device='cpu', voc_size=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_to_token = edge_to_token
        self.token_to_edge = token_to_edge
        self.num_neighbors = num_neighbors
        self.device = device

        if voc_size is None:
            voc_size = len(edge_to_token)
        self.voc_size = voc_size

        with open(filepath, "r") as f:
            lines = f.readlines()
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split()
        if len(line) < 2*self.num_nodes:
            raise ValueError(f"Line {idx} too short for {2*self.num_nodes} coords.")

        coords = np.array(line[:2*self.num_nodes], dtype=np.float32).reshape(self.num_nodes, 2)
        coords_swapped = coords[:, [1, 0]]

        try:
            output_idx = line.index('output')
        except ValueError:
            raise ValueError("No 'output' in line.")
        sol_nodes = [int(x)-1 for x in line[output_idx+1:] if x.isdigit()]
        if len(sol_nodes) != self.num_nodes+1 or sol_nodes[0] != sol_nodes[-1]:
            raise ValueError("Solution tour not matching expected length or not cyclic.")
        sol_nodes = sol_nodes[:-1]

        x_mid = coords[:, 0].mean()
        y_mid = coords[:, 1].mean()

        # Quadrants
        quadrants = np.ones(self.num_nodes, dtype=int)
        quadrants[coords[:, 0] < x_mid] = 2
        quadrants[coords[:, 1] < y_mid] = 4
        quadrants[(coords[:, 0] < x_mid) & (coords[:, 1] < y_mid)] = 3

        # topDown edges
        topDownEdgePairs = []
        for i in range(self.num_nodes):
            j = (i + 1) % self.num_nodes
            q_i = quadrants[sol_nodes[i]]
            q_j = quadrants[sol_nodes[j]]
            if ((q_i in [1, 2] and q_j in [3, 4]) or (q_j in [1, 2] and q_i in [3, 4])):
                topDownEdgePairs.append((min(sol_nodes[i], sol_nodes[j]),
                                         max(sol_nodes[i], sol_nodes[j])))

        # leftRight edges
        leftRightEdgePairs = []
        for i in range(self.num_nodes):
            j = (i + 1) % self.num_nodes
            q_i = quadrants[sol_nodes[i]]
            q_j = quadrants[sol_nodes[j]]
            if ((q_i in [1, 4] and q_j in [2, 3]) or (q_j in [1, 4] and q_i in [2, 3])):
                leftRightEdgePairs.append((min(sol_nodes[i], sol_nodes[j]),
                                           max(sol_nodes[i], sol_nodes[j])))

        def sort_and_tokenize(edgePairs):
            edgePairs = list(set(edgePairs))
            edgePairs.sort(key=lambda x: (x[0], x[1]))
            tokens = [self.edge_to_token[(i, j)] for (i, j) in edgePairs]
            return tokens

        START_TOKEN = self.edge_to_token['<START>']
        END_TOKEN   = self.edge_to_token['<END>']
        topDownTokens = [START_TOKEN] + sort_and_tokenize(topDownEdgePairs) + [END_TOKEN]
        leftRightTokens = [START_TOKEN] + sort_and_tokenize(leftRightEdgePairs) + [END_TOKEN]

        final_token_seq = [START_TOKEN] + topDownTokens + leftRightTokens + [END_TOKEN]

        # Distances + adjacency
        W_val = squareform(pdist(coords, metric='euclidean'))  # (N,N)
        if self.num_neighbors == -1:
            W = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)
        else:
            W = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
            knns = np.argpartition(W_val, self.num_neighbors, axis=1)[:, :self.num_neighbors]
            W[np.arange(self.num_nodes)[:, None], knns] = 1
        np.fill_diagonal(W, 2)

        # edge_rank
        edge_rank = np.full((self.voc_size,), -999999, dtype=np.float32)
        for token_id in range(self.voc_size - 3):
            if token_id not in self.token_to_edge:
                continue
            i, j = self.token_to_edge[token_id]
            if i < self.num_nodes and j < self.num_nodes:
                if (W[i, j] == 1) or (W[j, i] == 1):
                    meanX = (coords[i, 0] + coords[j, 0]) / 2.0
                    diff = abs(meanX - x_mid)
                    edge_rank[token_id] = -diff

        return {
            'coords': torch.tensor(coords, dtype=torch.float, device=self.device),
            'coords_swapped': torch.tensor(coords_swapped, dtype=torch.float, device=self.device),
            'x_edges': torch.tensor(W, dtype=torch.long, device=self.device),
            'x_edges_values': torch.tensor(W_val, dtype=torch.float, device=self.device),
            'topDownTokens': torch.tensor(topDownTokens, dtype=torch.long, device=self.device),
            'leftRightTokens': torch.tensor(leftRightTokens, dtype=torch.long, device=self.device),
            'token_seq': torch.tensor(final_token_seq, dtype=torch.long, device=self.device),
            'edge_rank': torch.tensor(edge_rank, dtype=torch.float, device=self.device),
            'x_mid': torch.tensor(x_mid, dtype=torch.float, device=self.device),
            'y_mid': torch.tensor(y_mid, dtype=torch.float, device=self.device),
        }

################################################################################
# Collate function
################################################################################

def collate_fn(batch, edge_to_token):
    PAD_TOKEN = edge_to_token['<PAD>']
    
    coords          = torch.stack([s['coords'] for s in batch], dim=0)
    coords_swapped  = torch.stack([s['coords_swapped'] for s in batch], dim=0)
    x_edges         = torch.stack([s['x_edges'] for s in batch], dim=0)
    x_edges_values  = torch.stack([s['x_edges_values'] for s in batch], dim=0)
    edge_rank       = torch.stack([s['edge_rank'] for s in batch], dim=0)
    x_mid           = torch.stack([s['x_mid'] for s in batch], dim=0)
    y_mid           = torch.stack([s['y_mid'] for s in batch], dim=0)

    topDownTokens = pad_sequence([s['topDownTokens'] for s in batch],
                                 batch_first=True, padding_value=PAD_TOKEN)
    leftRightTokens = pad_sequence([s['leftRightTokens'] for s in batch],
                                   batch_first=True, padding_value=PAD_TOKEN)
    token_seq = pad_sequence([s['token_seq'] for s in batch],
                             batch_first=True, padding_value=PAD_TOKEN)

    return {
        'coords': coords,
        'coords_swapped': coords_swapped,
        'x_edges': x_edges,
        'x_edges_values': x_edges_values,
        'edge_rank': edge_rank,
        'x_mid': x_mid,
        'y_mid': y_mid,
        'topDownTokens': topDownTokens,
        'leftRightTokens': leftRightTokens,
        'token_seq': token_seq,
    }

################################################################################
# Model components
################################################################################

class BatchNormNode(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.bn(x)
        x = x.transpose(1, 2).contiguous()
        return x

class BatchNormEdge(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
    def forward(self, e):
        e = e.permute(0, 3, 1, 2).contiguous()
        e = self.bn(e)
        e = e.permute(0, 2, 3, 1).contiguous()
        return e

class NodeFeatures(nn.Module):
    def __init__(self, hidden_dim, aggregation="mean"):
        super().__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, edge_gate):
        Ux = self.U(x)
        Vx = self.V(x).unsqueeze(1)  # (B,1,V,H)
        gateVx = edge_gate * Vx      # (B,V,V,H)
        if self.aggregation == "mean":
            denom = edge_gate.sum(dim=2) + 1e-20
            agg = gateVx.sum(dim=2) / denom
        else:
            agg = gateVx.sum(dim=2)
        return Ux + agg

class EdgeFeatures(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x, e):
        Ue = self.U(e)
        Vx_i = self.V(x).unsqueeze(2)  # (B,V,1,H)
        Vx_j = self.V(x).unsqueeze(1)  # (B,1,V,H)
        return Ue + Vx_i + Vx_j

class ResidualGatedGCNLayer(nn.Module):
    def __init__(self, hidden_dim, aggregation="sum"):
        super().__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)
    def forward(self, x, e):
        e_new = self.edge_feat(x, e)
        edge_gate = torch.sigmoid(e_new)
        x_new = self.node_feat(x, edge_gate)
        e_new = self.bn_edge(e_new)
        x_new = self.bn_node(x_new)
        e_new = F.relu(e_new)
        x_new = F.relu(x_new)
        return x + x_new, e + e_new

class ResidualGatedGCNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_nodes = config.num_nodes
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.aggregation = config.aggregation

        self.nodes_coord_embedding = nn.Linear(config.node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(config.voc_edges_in, self.hidden_dim // 2)

        self.gcn_layers = nn.ModuleList([
            ResidualGatedGCNLayer(self.hidden_dim, self.aggregation) 
            for _ in range(self.num_layers)
        ])

    def forward(self, x_edges, x_edges_values, x_nodes_coord):
        x = self.nodes_coord_embedding(x_nodes_coord)
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(-1))
        e_tags = self.edges_embedding(x_edges)
        e = torch.cat((e_vals, e_tags), dim=-1)

        for layer in self.gcn_layers:
            x, e = layer(x, e)
        return x, e

################################################################################
# Transformer Decoder
################################################################################

class EdgeTokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
    def forward(self, token_seq):
        return self.emb(token_seq)

class TransformerEdgeDecoder(nn.Module):
    def __init__(self, d_model=256, vocab_size=1228, nhead=4, num_layers=4, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embedding = EdgeTokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            activation='relu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_tokens, node_emb, tgt_mask=None):
        B, L = tgt_tokens.shape
        tgt_emb = self.token_embedding(tgt_tokens)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(L, device=tgt_tokens.device)
        out = self.decoder(tgt=tgt_emb, memory=node_emb, tgt_mask=tgt_mask)
        logits = self.output_projection(out)
        return logits

    def generate_square_subsequent_mask(self, sz, device):
        # mask shape (sz, sz) => True => block
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
        return mask

class GCNTransformerEdgeModel(nn.Module):
    def __init__(self, config, edge_vocab_size):
        super().__init__()
        self.gcn = ResidualGatedGCNModel(config)
        d_model = config.decoder_d_model
        self.decoder = TransformerEdgeDecoder(
            d_model=d_model,
            vocab_size=edge_vocab_size,
            nhead=config.decoder_nhead,
            num_layers=config.decoder_layers,
            padding_idx=edge_vocab_size-3  # <PAD> is vocab_size-3
        )
        self.edge_vocab_size = edge_vocab_size
        self.d_model = d_model

    def forward(self, x_edges, x_edges_values, x_nodes_coord, token_seq, edge_context=None):
        node_emb, edge_emb = self.gcn(x_edges, x_edges_values, x_nodes_coord)
        if edge_context is not None:
            node_emb = torch.cat([node_emb, edge_context], dim=1)
        edge_emb_flat = edge_emb.view(edge_emb.size(0), -1, edge_emb.size(-1))
        logits = self.decoder(token_seq, edge_emb_flat)
        return logits

################################################################################
# Edge context utility
################################################################################

def generate_edge_context(decoder_embedding, topdown_tokens, edge_rank, N=2):
    B, L = topdown_tokens.shape
    rank_vals = edge_rank.gather(1, topdown_tokens)  # (B,L)
    _, topk_indices = torch.topk(rank_vals, N, dim=1, largest=True, sorted=True)
    topk_tokens = torch.gather(topdown_tokens, 1, topk_indices)
    context_emb = decoder_embedding(topk_tokens)
    return context_emb

################################################################################
# Training/Eval routines
################################################################################

def sequence_cross_entropy(logits, target_seq, pad_token):
    B, L, V = logits.shape
    logits_flat = logits.view(B*L, V)
    target_flat = target_seq.view(B*L)
    loss = F.cross_entropy(logits_flat, target_flat, ignore_index=pad_token, reduction='mean')
    return loss

def get_noam_scheduler(optimizer, d_model, warmup_steps=4000):
    """
    Create a Noam scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        d_model (int): The dimensionality of the model.
        factor (float): A scaling factor.
        warmup_steps (int): Number of warmup steps.

    Returns:
        LambdaLR: A PyTorch LambdaLR scheduler.
    """
    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def generate_padded_logit(logits_last_step, generated_seq, start_token, end_token, pad_token, device):
    B, _, C = logits_last_step.shape
    start_logits = torch.zeros((B, 1, C), device=device)
    start_logits[:, 0, start_token] = 100.0
    logits = torch.cat([start_logits, logits_last_step], dim=1)

    L = logits.size(1)
    mask_end = (generated_seq == end_token)
    has_end = mask_end.any(dim=1)
    end_indices = mask_end.float().argmax(dim=1)

    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    mask_after_end = (positions > end_indices.unsqueeze(1)) & has_end.unsqueeze(1)

    logits = logits.masked_fill(mask_after_end.unsqueeze(-1), 0.0)
    logits[:, :, pad_token] = logits[:, :, pad_token].masked_fill(mask_after_end, 100.0)
    return logits

def two_pass_train_single_epoch(model, dataloader, optimizer, scheduler, config, epoch, writer=None):
    """
    Single epoch training with batch-level TensorBoard logging every 5 batches.
    """
    model.train()
    running_loss = 0.0
    running_count = 0
    start_t = time.time()

    pad_token = config.special_tokens['PAD']
    log_every = 5  # Set logging frequency to every 5 batches

    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch
        coords = batch['coords']        
        x_edges = batch['x_edges']
        x_edges_values = batch['x_edges_values']
        token_seq_td = batch['topDownTokens']  # (B, L_td)

        # Forward pass
        logits_td = model(x_edges, x_edges_values, coords, token_seq_td, edge_context=None)
        target_seq_td = token_seq_td[:, 1:].contiguous()  # Shift targets
        logits_td = logits_td[:, :-1, :].contiguous()    # Align logits with targets

        # Compute loss
        loss_td = sequence_cross_entropy(logits_td, target_seq_td, pad_token)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_td.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        # Accumulate loss
        B = coords.size(0)
        running_loss += loss_td.item() * B
        running_count += B
        avg_loss = running_loss / running_count

        # Calculate global_step
        global_step = (epoch - 1) * len(dataloader) + batch_idx

        # Log batch loss to TensorBoard every 'log_every' batches
        if writer is not None and (batch_idx % log_every == 0):
            writer.add_scalar("Train/Batch_Loss", loss_td.item(), global_step)
            writer.add_scalar("Train/Learning_Rate", scheduler.get_last_lr()[0], global_step)
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss_td.item():.4f}")

    epoch_time = time.time() - start_t
    final_loss = running_loss / running_count
    if writer is not None:
        writer.add_scalar("Train/Epoch_Loss", final_loss, epoch)
    return epoch_time, final_loss

def two_pass_evaluate(model, dataloader, config, epoch, mode='Val', writer=None):
    """
    Evaluation with batch-level TensorBoard logging every 5 batches.
    """
    model.eval()
    running_loss = 0.0
    running_count = 0
    start_t = time.time()

    pad_token   = config.special_tokens['PAD']
    start_token = config.special_tokens['START']
    end_token   = config.special_tokens['END']
    log_every = 5  # Set logging frequency to every 5 batches

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            coords = batch['coords']
            x_edges = batch['x_edges']
            x_edges_values = batch['x_edges_values']
            edge_rank = batch['edge_rank']
            token_seq_td = batch['topDownTokens']
            token_seq_lr = batch['leftRightTokens']

            B = coords.size(0)

            # Pass 1: Predict top-down edges (greedy)
            generated_td = torch.full((B, 1), start_token, dtype=torch.long, device=coords.device)
            max_len_td = token_seq_td.size(1)
            logits_td_step = None
            for _ in range(max_len_td - 1):
                logits_td_step = model(x_edges, x_edges_values, coords, generated_td, edge_context=None)
                next_token = logits_td_step[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_td = torch.cat([generated_td, next_token], dim=1)

            logits_td = generate_padded_logit(
                logits_td_step, generated_td, start_token, end_token,
                pad_token, coords.device
            )
            loss_td = sequence_cross_entropy(logits_td, token_seq_td, pad_token)

            '''
            # Pass 2: Predict left-right edges with context
            real_edge_context = None
            if hasattr(model.module if isinstance(model, DDP) else model, "decoder"):
                submodel = model.module if isinstance(model, DDP) else model
                real_edge_context = generate_edge_context(
                    submodel.decoder.token_embedding.emb,
                    generated_td,
                    edge_rank,
                    N=2
                )

            generated_lr = torch.full((B, 1), start_token, dtype=torch.long, device=coords.device)
            max_len_lr = token_seq_lr.size(1)
            logits_lr_step = None
            for _ in range(max_len_lr - 1):
                logits_lr_step = model(
                    x_edges, x_edges_values, coords,
                    generated_lr, edge_context=real_edge_context
                )
                next_token = logits_lr_step[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_lr = torch.cat([generated_lr, next_token], dim=1)

            logits_lr = generate_padded_logit(
                logits_lr_step, generated_lr, start_token, end_token,
                pad_token, coords.device
            )
            loss_lr = sequence_cross_entropy(logits_lr, token_seq_lr, pad_token)
            '''

            # Accumulate loss
            total_loss = loss_td
            running_loss += total_loss.item() * B
            running_count += B

            # Calculate global_step
            global_step = (epoch - 1) * len(dataloader) + batch_idx

            # Log batch loss to TensorBoard every 'log_every' batches
            if writer is not None and (batch_idx % log_every == 0):
                writer.add_scalar(f"{mode}/Batch_Loss", total_loss.item(), global_step)
                print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], {mode} Loss: {total_loss.item():.4f}")

    epoch_time = time.time() - start_t
    final_loss = running_loss / running_count
    if writer is not None:
        writer.add_scalar(f"{mode}/Epoch_Loss", final_loss, epoch)
    return epoch_time, final_loss

################################################################################
# Main training loop
################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="tsp-data/tsp50_train_concorde.txt")
    parser.add_argument("--val_data",   type=str, default="tsp-data/tsp50_val_concorde.txt")
    parser.add_argument("--test_data",  type=str, default="tsp-data/tsp50_test_concorde.txt")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--test_freq", type=int, default=15)

    # Optionally specify a log dir for tensorboard
    parser.add_argument("--log_dir", type=str, default=None)
    return parser.parse_args()

def main_worker(args):
    """
    Main worker function for multi-GPU training with torchrun.
    Reads environment variables for local/global rank.
    """
    # 1) Read local rank, global rank, etc. from the environment.
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 2) Set device and init process group
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)

    # 3) Build vocab
    edge2tok, tok2edge = build_global_vocabulary(args.num_nodes)
    collate = partial(collate_fn, edge_to_token=edge2tok)

    # 4) Create dataset / sampler / dataloader
    train_dataset = TSPDataset(args.train_data, num_nodes=args.num_nodes,
                               edge_to_token=edge2tok, token_to_edge=tok2edge,
                               device=device)
    val_dataset = TSPDataset(args.val_data, num_nodes=args.num_nodes,
                             edge_to_token=edge2tok, token_to_edge=tok2edge,
                             device=device)
    test_dataset = TSPDataset(args.test_data, num_nodes=args.num_nodes,
                              edge_to_token=edge2tok, token_to_edge=tok2edge,
                              device=device)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size,
                                       rank=rank, shuffle=False)
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size,
                                       rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=val_sampler, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             sampler=test_sampler, collate_fn=collate)

    # 5) Build model + wrap with DDP
    cfg_autoreg = DotDict({
        'num_nodes': args.num_nodes,
        'hidden_dim': 296,
        'num_layers': 5,
        'aggregation':'mean',
        'node_dim': 2,
        'voc_edges_in': len(edge2tok),
        'decoder_d_model': 296,
        'decoder_nhead': 8,
        'decoder_layers': 1,
        'special_tokens': {
            'PAD': edge2tok['<PAD>'],
            'START': edge2tok['<START>'],
            'END': edge2tok['<END>']
        },
    })

    model = GCNTransformerEdgeModel(cfg_autoreg, edge_vocab_size=len(edge2tok)).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #number of warmup steps = total number of steps / 10
    n_warmup_steps = (len(train_dataset)/(world_size*args.batch_size)*args.epochs) // 10
    

    scheduler = get_noam_scheduler(
    optimizer,
    d_model=cfg_autoreg.hidden_dim,
    warmup_steps=n_warmup_steps)  

    # TensorBoard only on rank 0
    writer = None
    if rank == 0 and args.log_dir is not None:
        # clear logdir
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
        writer = SummaryWriter(log_dir=args.log_dir)

    # 6) Training loop
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        e_time, train_loss = two_pass_train_single_epoch(model, train_loader, optimizer, scheduler, cfg_autoreg, epoch, writer=writer)
        if rank == 0:
            print(f"[Rank 0] Epoch {epoch}/{args.epochs} | Train Loss = {train_loss:.4f}")

        # Validation
        if epoch % args.val_freq == 0:
            val_sampler.set_epoch(epoch)
            val_time, val_loss = two_pass_evaluate(model, val_loader, cfg_autoreg, epoch, mode='Val', writer=writer)
            if rank == 0:
                print(f"[Rank 0] Epoch {epoch} | Validation Loss = {val_loss:.4f}")

        # Test
        if epoch % args.test_freq == 0:
            test_sampler.set_epoch(epoch)
            test_time, test_loss = two_pass_evaluate(model, test_loader, cfg_autoreg, epoch, mode='Test', writer=writer)
            if rank == 0:
                print(f"[Rank 0] Epoch {epoch} | Test Loss = {test_loss:.4f}")

    # 7) Cleanup
    dist.destroy_process_group()
    if writer is not None:
        writer.close()

def main():
    args = parse_args()
    main_worker(args)

if __name__ == "__main__":
    """
    Usage (new recommended approach):
      torchrun --nproc_per_node=4 gnn_ar_ddp.py [--args...]
    """
    main()
