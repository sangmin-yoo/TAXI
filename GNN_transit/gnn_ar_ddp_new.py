#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import argparse
import math
from random import shuffle

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

from tqdm import tqdm
from functools import partial
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, LinearLR

import shutil

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


################################################################################
# 1) Basic Utilities
################################################################################

class DotDict(dict):
    """Dictionary wrapper to access keys as attributes."""
    def __getattr__(self, key):
        return self.get(key)

def build_global_vocabulary(num_nodes=50):
    """
    Build a canonical ordering of edges (i < j) for up to num_nodes=50.
    Also add special tokens: <PAD>, <START>, <END>
    Returns:
        edge_to_token: dict (i,j) -> token_id
        token_to_edge: dict token_id -> (i,j)
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

    # Special Tokens
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
# 2) Dataset & Collate
################################################################################

class TSPDataset(Dataset):
    """
    Expects lines: x1 y1 x2 y2 ... xN yN output 1 2 3 ... N 1
    Produces:
      - coords: node coordinates (N, 2)
      - x_edges, x_edges_values: adjacency + distance
      - topDownTokens, leftRightTokens, token_seq
      - edge_rank
      - x_mid, y_mid
      - y_edges_topdown: for pretraining (binary top-down edges)
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
            raise ValueError(f"Line {idx} too short to have {2*self.num_nodes} coords.")

        coords = np.array(line[:2*self.num_nodes], dtype=np.float32).reshape(self.num_nodes, 2)
        coords_swapped = coords[:, [1, 0]]

        # solution parse
        try:
            output_idx = line.index('output')
        except ValueError:
            raise ValueError("No 'output' in line.")
        sol_nodes = [int(x)-1 for x in line[output_idx+1:] if x.isdigit()]
        if len(sol_nodes) != self.num_nodes+1 or sol_nodes[0] != sol_nodes[-1]:
            raise ValueError("Solution tour not matching expected length or not cyclic.")
        sol_nodes = sol_nodes[:-1]  # remove the repeat

        x_mid = coords[:, 0].mean()
        y_mid = coords[:, 1].mean()

        # define quadrant
        quadrants = np.ones(self.num_nodes, dtype=int)
        quadrants[coords[:, 0] < x_mid] = 2
        quadrants[coords[:, 1] < y_mid] = 4
        quadrants[(coords[:, 0] < x_mid) & (coords[:, 1] < y_mid)] = 3

        # collect top-down edges
        topDownEdgePairs = []
        for i in range(self.num_nodes):
            j = (i + 1) % self.num_nodes
            q_i = quadrants[sol_nodes[i]]
            q_j = quadrants[sol_nodes[j]]
            if ((q_i in [1, 2] and q_j in [3, 4]) or (q_j in [1, 2] and q_i in [3, 4])):
                topDownEdgePairs.append((min(sol_nodes[i], sol_nodes[j]),
                                         max(sol_nodes[i], sol_nodes[j])))

        # collect left-right edges
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
        END_TOKEN = self.edge_to_token['<END>']

        topDownTokens = [START_TOKEN] + sort_and_tokenize(topDownEdgePairs) + [END_TOKEN]
        leftRightTokens = [START_TOKEN] + sort_and_tokenize(leftRightEdgePairs) + [END_TOKEN]
        final_token_seq = [START_TOKEN] + topDownTokens + leftRightTokens + [END_TOKEN]

        W_val = squareform(pdist(coords, metric='euclidean'))  # (N, N)
        if self.num_neighbors == -1:
            W = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)
        else:
            W = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
            knns = np.argpartition(W_val, self.num_neighbors, axis=1)[:, :self.num_neighbors]
            W[np.arange(self.num_nodes)[:, None], knns] = 1
        np.fill_diagonal(W, 2)

        # Generate y_edges_topdown for pretraining (binary classification)
        y_edges_topdown = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for (i, j) in topDownEdgePairs:
            y_edges_topdown[i, j] = 1
            y_edges_topdown[j, i] = 1

        # rank edges
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

        sample = {
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
            'y_edges_topdown': torch.tensor(y_edges_topdown, dtype=torch.float, device=self.device)
        }
        return sample

def collate_fn(batch, edge_to_token):
    coords = torch.stack([sample['coords'] for sample in batch], dim=0)
    coords_swapped = torch.stack([sample['coords_swapped'] for sample in batch], dim=0)
    x_edges = torch.stack([sample['x_edges'] for sample in batch], dim=0)
    x_edges_values = torch.stack([sample['x_edges_values'] for sample in batch], dim=0)
    edge_rank = torch.stack([sample['edge_rank'] for sample in batch], dim=0)
    x_mid = torch.stack([sample['x_mid'] for sample in batch], dim=0)
    y_mid = torch.stack([sample['y_mid'] for sample in batch], dim=0)

    PAD_TOKEN = edge_to_token['<PAD>']

    token_seqs = [sample['token_seq'] for sample in batch]
    token_seqs_padded = pad_sequence(token_seqs, batch_first=True, padding_value=PAD_TOKEN)

    td_token_seqs = [sample['topDownTokens'] for sample in batch]
    td_token_seqs_padded = pad_sequence(td_token_seqs, batch_first=True, padding_value=PAD_TOKEN)

    lr_token_seqs = [sample['leftRightTokens'] for sample in batch]
    lr_token_seqs_padded = pad_sequence(lr_token_seqs, batch_first=True, padding_value=PAD_TOKEN)

    y_edges_topdown = torch.stack([sample['y_edges_topdown'] for sample in batch], dim=0)

    return {
        'coords': coords,
        'coords_swapped': coords_swapped,
        'x_edges': x_edges,
        'x_edges_values': x_edges_values,
        'token_seq': token_seqs_padded,
        'topDownTokens': td_token_seqs_padded,
        'leftRightTokens': lr_token_seqs_padded,
        'edge_rank': edge_rank,
        'x_mid': x_mid,
        'y_mid': y_mid,
        'y_edges_topdown': y_edges_topdown
    }


################################################################################
# 3) GCN + MLP Pretraining Classes
################################################################################

class BatchNormNode(nn.Module):
    """Batch normalization for node features."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.batch_norm(x)
        x = x.transpose(1, 2).contiguous()
        return x

class BatchNormEdge(nn.Module):
    """Batch normalization for edge features."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)
    def forward(self, e):
        e = e.permute(0, 3, 1, 2).contiguous()
        e = self.batch_norm(e)
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
        Vx = self.V(x).unsqueeze(1)
        gateVx = edge_gate * Vx
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
        Vx_i = self.V(x).unsqueeze(2)
        Vx_j = self.V(x).unsqueeze(1)
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

class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction."""
    def __init__(self, hidden_dim, output_dim, L=2):
        super().__init__()
        layers = []
        for _ in range(L-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.U = nn.ModuleList(layers)
        self.V = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.U:
            x = F.relu(layer(x))
        return self.V(x)

################################################################################
# Pretrain Model (Stage 1)
################################################################################

class ResidualGatedGCNModelPretrain(nn.Module):
    """
    Residual Gated GCN Model for pretraining (learning good node embeddings + edge embeddings)
    with an MLP for predicting top-down edges as a classification problem.
    """
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

        # Output MLP over edge embeddings (binary classification => 2 classes)
        self.mlp_edges = MLP(self.hidden_dim, 2, config.mlp_layers)

    def forward(self, x_edges, x_edges_values, x_nodes_coord):
        x = self.nodes_coord_embedding(x_nodes_coord)
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(-1))
        e_tags = self.edges_embedding(x_edges)
        e = torch.cat((e_vals, e_tags), dim=-1)

        for layer in self.gcn_layers:
            x, e = layer(x, e)

        # Predict top-down edges
        # shape => (B, V, V, 2)
        y_pred_edges = self.mlp_edges(e)
        return y_pred_edges

    def loss_edges(self, y_pred_edges, y_edges, edge_cw):
        """
        y_pred_edges: (B, V, V, 2)
        y_edges: (B, V, V) with 0/1
        """
        # Cross Entropy across the 2 classes
        # We'll do a log_softmax -> NLLLoss approach
        y = F.log_softmax(y_pred_edges, dim=3)  # (B, V, V, 2)
        y = y.permute(0, 3, 1, 2).contiguous()  # (B, 2, V, V)
        return nn.NLLLoss(weight=edge_cw)(y, y_edges.long())


################################################################################
# 4) GCN + Transformer (Stage 2)
################################################################################

class ResidualGatedGCNModelTransformer(nn.Module):
    """
    Similar to the GCN used in pretraining, but we rename it here to indicate
    it's used in the Transformer model. Could differ internally if needed.
    """
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


class GCNEdgeEmbeddingLookup(nn.Module):
    """
    For each token t:
      - If t is <PAD>, <START>, <END>, look up from a special embedding
      - Else, find (i,j) from token->edge, fetch from e[b,i,j,:].
    """
    def __init__(self, token_to_edge, d_model, pad_idx, start_idx, end_idx):
        super().__init__()
        self.token_to_edge = token_to_edge
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.special_token_emb = nn.Embedding(3, d_model)
        self.special_token_map = {
            pad_idx: 0,
            start_idx: 1,
            end_idx: 2
        }

    def forward(self, tokens, edge_emb):
        B, L = tokens.shape
        device = tokens.device
        H = edge_emb.size(-1)
        out = torch.zeros((B, L, H), device=device)

        # Step 1: Identify special tokens
        special_mask = torch.full((B, L), -1, dtype=torch.long, device=device)
        for st_idx, sp_idx in self.special_token_map.items():
            st_mask = (tokens == st_idx)
            special_mask[st_mask] = sp_idx

        # Fill special tokens
        special_positions = (special_mask >= 0).nonzero(as_tuple=False)
        if special_positions.size(0) > 0:
            special_indices = special_mask[special_positions[:,0], special_positions[:,1]]
            special_vectors = self.special_token_emb(special_indices)
            out[special_positions[:,0], special_positions[:,1], :] = special_vectors

        # Step 2: GCN lookup for non-special tokens
        non_special_mask = (special_mask < 0)
        non_special_positions = non_special_mask.nonzero(as_tuple=False)
        if non_special_positions.size(0) > 0:
            non_special_tokens = tokens[non_special_positions[:,0], non_special_positions[:,1]]
            i_j = []
            for t_id in non_special_tokens.cpu().tolist():
                i_j.append(self.token_to_edge[t_id])
            i_j = torch.tensor(i_j, dtype=torch.long, device=device)
            i_idx = i_j[:,0]
            j_idx = i_j[:,1]
            b_idx = non_special_positions[:, 0]
            gcn_vectors = edge_emb[b_idx, i_idx, j_idx, :]
            out[b_idx, non_special_positions[:,1], :] = gcn_vectors
        return out


class TransformerGCNEdgeDecoder(nn.Module):
    def __init__(self, d_model=256, vocab_size=1228, nhead=4, num_layers=4,
                 pad_idx=0, start_idx=1, end_idx=2, token_to_edge=None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.embedding_lookup = GCNEdgeEmbeddingLookup(
            token_to_edge=token_to_edge,
            d_model=d_model,
            pad_idx=pad_idx,
            start_idx=start_idx,
            end_idx=end_idx
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4,
                                                   activation='relu',
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, tgt_tokens, edge_emb, node_emb=None, tgt_mask=None):
        B, L = tgt_tokens.shape
        device = tgt_tokens.device
        tgt_emb = self.embedding_lookup(tgt_tokens, edge_emb)

        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(L, device=device)

        out = self.decoder(tgt=tgt_emb, memory=node_emb, tgt_mask=tgt_mask)  # no cross-attn if node_emb=None
        logits = self.output_projection(out)  # (B, L, vocab_size)
        return logits

class GCNTransformerEdgeModel(nn.Module):
    """
    Stage 2 model: GCN + Transformer-based decoder for edge tokens.
    Uses ResidualGatedGCNModelTransformer internally for the GCN portion.
    """
    def __init__(self, config, edge_vocab_size, token_to_edge):
        super().__init__()
        self.gcn = ResidualGatedGCNModelTransformer(config)
        self.edge_vocab_size = edge_vocab_size
        self.pad_idx = config.special_tokens['PAD']
        self.start_idx = config.special_tokens['START']
        self.end_idx = config.special_tokens['END']

        self.decoder = TransformerGCNEdgeDecoder(
            d_model=config.decoder_d_model,
            vocab_size=edge_vocab_size,
            nhead=config.decoder_nhead,
            num_layers=config.decoder_layers,
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            token_to_edge=token_to_edge
        )

    def forward(self, x_edges, x_edges_values, x_nodes_coord, token_seq):
        node_emb, edge_emb = self.gcn(x_edges, x_edges_values, x_nodes_coord)
        logits = self.decoder(token_seq, edge_emb, node_emb=node_emb)
        return logits


################################################################################
# 5) Training/Evaluation Utils
################################################################################

def sequence_cross_entropy(logits, target_seq, pad_token):
    """
    logits: (B,L,vocab_size)
    target_seq: (B,L)
    """
    B,L,_ = logits.shape
    logits_flat = logits.view(B*L, -1)
    target_flat = target_seq.view(B*L)
    loss = F.cross_entropy(logits_flat, target_flat, ignore_index=pad_token, reduction='mean')
    return loss


################################################################################
# 6) Schedulers (Noam) & Trainer Functions
################################################################################

def get_noam_scheduler(optimizer, d_model, warmup_steps=4000):
    """
    Noam Scheduler: lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    Set your optimizer's base lr=1.0 so that the scheduler scales from there.
    """
    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def pretrain_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer=None, global_step_start=0):
    """
    Pretraining GCN+MLP for top-down edge classification.
    """
    model.train()
    running_loss = 0.0
    running_count = 0
    global_step = global_step_start

    edge_cw = None  # We'll compute once from the first batch
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        x_edges = batch['x_edges']
        x_edges_values = batch['x_edges_values']
        x_nodes_coord = batch['coords']
        y_edges = batch['y_edges_topdown']  # shape (B, V, V)

        # Compute class weights once from the first batch (if needed)
        if edge_cw is None:
            edge_labels_np = y_edges.view(-1).cpu().numpy()
            classes_np = np.unique(edge_labels_np)
            cw_np = compute_class_weight("balanced", classes=classes_np, y=edge_labels_np)
            # class_weight is an array with shape (#classes,)
            # but if classes_np isn't [0,1] in order, we might reorder. We'll assume it's [0,1].
            edge_cw = torch.tensor(cw_np, dtype=torch.float, device=device)

        # Forward pass
        y_pred_edges = model(x_edges, x_edges_values, x_nodes_coord)

        if isinstance(model, (torch.nn.DataParallel, DDP)):
            loss = model.module.loss_edges(y_pred_edges, y_edges, edge_cw)
        else:
            loss = model.loss_edges(y_pred_edges, y_edges, edge_cw)

        loss.backward()
        optimizer.step()

        B = x_nodes_coord.size(0)
        running_loss += loss.item() * B
        running_count += B

        # TensorBoard logging at batch level
        if writer is not None:
            writer.add_scalar("Pretrain/Batch_Loss", loss.item(), global_step)
        global_step += 1
    
    scheduler.step()

    final_loss = running_loss / running_count
    if writer is not None:
        writer.add_scalar("Pretrain/Epoch_Loss", final_loss, epoch)
        writer.add_scalar("Pretrain/Learning_Rate", scheduler.get_last_lr()[0], epoch)
    return final_loss, global_step


def train_transformer_one_epoch(model, dataloader, optimizer, scheduler, config,
                                epoch, writer=None, global_step_start=0):
    """
    Second stage training with GCN+Transformer. 
    Uses Noam scheduler => step() after each batch.
    """
    model.train()
    running_loss = 0.0
    running_count = 0
    pad_token = config.special_tokens['PAD']

    global_step = global_step_start
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        coords = batch['coords']
        x_edges = batch['x_edges']
        x_edges_values = batch['x_edges_values']
        token_seq_td = batch['topDownTokens']  # (B, L)

        logits_td = model(x_edges, x_edges_values, coords, token_seq_td)
        target_seq_td = token_seq_td[:, 1:].contiguous()
        logits_td = logits_td[:, :-1, :].contiguous()

        loss = sequence_cross_entropy(logits_td, target_seq_td, pad_token)

        loss.backward()
        optimizer.step()
        scheduler.step()  # Noam scheduler step after each batch

        B = coords.size(0)
        running_loss += loss.item() * B
        running_count += B

        # TensorBoard logging at batch level
        if writer is not None:
            writer.add_scalar("Transformer/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Transformer/Learning_Rate", scheduler.get_last_lr()[0], global_step)

        global_step += 1

    final_loss = running_loss / running_count
    if writer is not None:
        writer.add_scalar("Transformer/Epoch_Loss", final_loss, epoch)
    return final_loss, global_step

def load_pretrained_gcn(pretrain_model, final_model):
    """
    Load pretrained GCN weights into the final model's GCN module.
    
    Args:
        pretrain_model (ResidualGatedGCNModel): The pretrained GCN model with MLP.
        final_model (GCNTransformerEdgeModel): The final model with Transformer decoder.
    """
    with torch.no_grad():
        pretrain_state_dict = pretrain_model.state_dict()
        final_state_dict = final_model.state_dict()

        # Create a new OrderedDict for the final model's GCN
        new_gcn_state_dict = OrderedDict()

        for k, v in pretrain_state_dict.items():
            # Exclude MLP layers
            if 'mlp_edges' in k:
                continue
            # Prefix 'gcn.' to match the final model's GCN module
            new_key = 'gcn.' + k
            if new_key in final_state_dict:
                new_gcn_state_dict[new_key] = v
            else:
                print(f"Warning: Key {new_key} not found in the final model's state_dict.")

        # Load the new GCN state_dict into the final model
        missing_keys, unexpected_keys = final_model.load_state_dict(new_gcn_state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys in the final model's state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in the final model's state_dict: {unexpected_keys}")

        print("Pretrained GCN weights loaded successfully.")


################################################################################
# 7) Main DDP Script
################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="tsp-data/tsp50_train_concorde.txt")
    parser.add_argument("--val_data",   type=str, default="tsp-data/tsp50_val_concorde.txt")
    parser.add_argument("--test_data",  type=str, default="tsp-data/tsp50_test_concorde.txt")

    parser.add_argument("--epochs_pretrain", type=int, default=15)
    parser.add_argument("--epochs_transformer", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_transformer", type=float, default=1.0, 
                        help="Set base lr=1.0 so that Noam can scale it properly")
    #parser.add_argument("--warmup_steps", type=int, default=6000)

    parser.add_argument("--hidden_dim", type=int, default=296)
    parser.add_argument("--num_layers", type=int, default=30)
    parser.add_argument("--mlp_layers", type=int, default=3)
    parser.add_argument("--aggregation", type=str, default="mean")
    parser.add_argument("--node_dim", type=int, default=2)

    parser.add_argument("--decoder_d_model", type=int, default=296)
    parser.add_argument("--decoder_nhead", type=int, default=8)
    parser.add_argument("--decoder_layers", type=int, default=1)

    parser.add_argument("--log_dir", type=str, default=None)

    return parser.parse_args()


def main_worker():
    args = parse_args()

    # 1) Initialize Distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)

    # 2) Build Vocab
    edge2tok, tok2edge = build_global_vocabulary(50)  # example
    collate = partial(collate_fn, edge_to_token=edge2tok)

    # 3) Create Datasets & Samplers
    train_dataset = TSPDataset(filepath=args.train_data, 
                               num_nodes=50, 
                               edge_to_token=edge2tok, 
                               token_to_edge=tok2edge,
                               device=device)
    val_dataset = TSPDataset(filepath=args.val_data,
                             num_nodes=50,
                             edge_to_token=edge2tok,
                             token_to_edge=tok2edge,
                             device=device)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              collate_fn=collate,
                              drop_last=False)
    val_loader   = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              sampler=val_sampler,
                              collate_fn=collate,
                              drop_last=False)

    # 4) TensorBoard only on rank 0
    writer = None
    if rank == 0 and args.log_dir is not None:
        # clear logdir
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
        writer = SummaryWriter(log_dir=args.log_dir)

    ###########################################################################
    # STAGE 1: Pretrain GCN+MLP
    ###########################################################################
    cfg_pretrain = DotDict({
        'num_nodes': 50,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'mlp_layers': args.mlp_layers,
        'aggregation': args.aggregation,
        'node_dim': args.node_dim,
        'voc_edges_in': 3,
        'voc_edges_out': 2
    })

    model_pretrain = ResidualGatedGCNModelPretrain(cfg_pretrain).to(device)
    model_pretrain = DDP(model_pretrain, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer_pretrain = torch.optim.Adam(model_pretrain.parameters(), lr=args.lr_pretrain)

    scheduler_pretrain = torch.optim.lr_scheduler.LinearLR(
    optimizer_pretrain,
    start_factor=1.0,       # Start with the initial learning rate
    end_factor=0.1,         # Decay to 10% of the initial learning rate (1e-4 if initial was 1e-3)
    total_iters=args.epochs_pretrain)  # Number of epochs over which to decay

    global_step_pre = 0
    for epoch in range(1, args.epochs_pretrain + 1):
        train_sampler.set_epoch(epoch)
        epoch_loss, global_step_pre = pretrain_one_epoch(
            model=model_pretrain,
            dataloader=train_loader,
            optimizer=optimizer_pretrain,
            scheduler=scheduler_pretrain,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step_start=global_step_pre
        )
        if rank == 0:
            print(f"[Pretrain][Epoch {epoch}/{args.epochs_pretrain}] Loss={epoch_loss:.4f}")

    # After pretraining, we want to freeze GCN layers
    # We'll keep MLP detached or not used in stage 2.
    # We'll copy the GCN weights to the new transformer model and freeze them.
    # 1) Extract the state_dict from the pretrain model
    pretrain_sd = model_pretrain.module.state_dict()  # get original module's state_dict

    # 2) Build GCN+Transformer model for Stage 2
    cfg_transformer = DotDict({
        'num_nodes': 50,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'aggregation': args.aggregation,
        'node_dim': args.node_dim,
        'voc_edges_in': 3,
        'decoder_d_model': args.decoder_d_model,
        'decoder_nhead': args.decoder_nhead,
        'decoder_layers': args.decoder_layers,
        'special_tokens': {
            'PAD':  len(edge2tok) - 3,
            'START':len(edge2tok) - 2,
            'END':  len(edge2tok) - 1
        }
    })

    model_transformer = GCNTransformerEdgeModel(cfg_transformer, edge_vocab_size=len(edge2tok), token_to_edge=tok2edge).to(device)
    # rename 'gcn' -> 'gcn_layers' in state dict or we can map carefully
    
    # We only want to copy the GCN portion from the pretrain model
    # The pretrain model has .gcn_layers in 'model_pretrain.module.gcn_layers.*' or it might be something else
    # Actually in ResidualGatedGCNModelPretrain, we have 'gcn_layers' plus the embeddings
    # We'll do partial load: let's do a direct check
    with torch.no_grad():
        # Grab the portion of pretrain that belongs to gcn_layers, nodes_coord_embedding, edges_values_embedding, edges_embedding
        # in the new model, these are in model_transformer.gcn
        pretrain_dict = {}
        for k, v in pretrain_sd.items():
            # The typical keys might be 'gcn_layers.0.*', 'nodes_coord_embedding.*', etc.
            if k.startswith('nodes_coord_embedding') or k.startswith('edges_values_embedding') or k.startswith('edges_embedding') or k.startswith('gcn_layers'):
                # We'll rename them to match model_transformer.gcn.<layer_name>
                new_k = k.replace('module.', '')  # remove possible 'module.'
                # remove the prefix if any
                # in the transformer model, it's "gcn.nodes_coord_embedding..." etc.
                new_k = "gcn." + new_k  # so that it matches 'gcn.nodes_coord_embedding.weight', etc.
                pretrain_dict[new_k] = v

        model_transformer.load_state_dict(pretrain_dict, strict=False)

    #print(pretrain_dict.keys())
    
    #load_pretrained_gcn(model_pretrain, model_transformer)

    # 3) Freeze the GCN layers
    #for name, param in model_transformer.named_parameters():
    #    if name.startswith("gcn."):
    #        param.requires_grad = False  # freeze

    # Build DDP
    model_transformer = DDP(model_transformer, device_ids=[local_rank], output_device=local_rank)

    # 4) Optimizer + Noam Scheduler for Transformer stage
    #   We set base lr=1.0 for Noam
    #optimizer_transformer = torch.optim.Adam(
    #    filter(lambda p: p.requires_grad, model_transformer.parameters()),
    #    lr=args.lr_transformer, betas=(0.9, 0.98), eps=1e-9
    #)
    gcn_lr_factor = 0.1  #GCN LR is 10% of transformer LR
    # Initialize optimizer with separate parameter groups
    optimizer_transformer = torch.optim.Adam([
        {
            'params': model_transformer.module.gcn.parameters(),
            'lr': args.lr_transformer * gcn_lr_factor
        },
        {
            'params': model_transformer.module.decoder.parameters(),
            'lr': args.lr_transformer
        }
    ], betas=(0.9, 0.98), eps=1e-9)

    n_warmup_steps = (len(train_dataset)/(world_size*args.batch_size)*args.epochs_transformer) // 10

    scheduler_transformer = get_noam_scheduler(
        optimizer_transformer,
        d_model=args.decoder_d_model,
        warmup_steps=n_warmup_steps
    )

    ###########################################################################
    # STAGE 2: Train GCN+Transformer with Noam
    ###########################################################################
    global_step_trans = 0
    for epoch in range(1, args.epochs_transformer + 1):
        train_sampler.set_epoch(epoch)
        epoch_loss_trans, global_step_trans = train_transformer_one_epoch(
            model=model_transformer,
            dataloader=train_loader,
            optimizer=optimizer_transformer,
            scheduler=scheduler_transformer,
            config=cfg_transformer,
            epoch=epoch,
            writer=writer,
            global_step_start=global_step_trans
        )
        if rank == 0:
            print(f"[Transformer][Epoch {epoch}/{args.epochs_transformer}] Loss={epoch_loss_trans:.4f}")

    # Final cleanup
    dist.destroy_process_group()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    """
    Usage:
      torchrun --nproc_per_node=4 train_two_stage_ddp.py [--args...]
    """
    main_worker()
