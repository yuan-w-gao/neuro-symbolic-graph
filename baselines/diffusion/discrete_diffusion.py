"""Discrete denoising diffusion over labelled graphs, following DiGress (Vignac et al., ICLR 2023)."""

# Reimplemented because the official repo pins an older torch stack. Kept from the paper:
# marginal transitions Q_t = a_t I + (1 - a_t) 1 m^T, a cosine schedule, cycle and spectral
# features, the X/E/y graph transformer, the x0-marginalized posterior and EMA weights.
# Deviation: graphs are undirected (the edge type carries the EDS role, not its direction).

import argparse
import copy
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "common"))

from benchmarks import load_training_graphs  # noqa: E402

NO_EDGE = "<none>"
# build_vocabs and GraphTensors must agree on these, or unlabelled edges silently read as "no edge"
DEFAULT_NODE_LABEL = "*"
DEFAULT_EDGE_LABEL = "e"


class GraphTensors:
    """Dense node/edge type tensors for a corpus of small labelled graphs."""

    def __init__(self, graphs, node_vocab, edge_vocab, max_nodes: int):
        self.node_vocab = list(node_vocab)
        self.edge_vocab = list(edge_vocab)
        self.max_nodes = max_nodes
        self.node_index = {label: i for i, label in enumerate(self.node_vocab)}
        self.edge_index = {label: i for i, label in enumerate(self.edge_vocab)}

        count = len(graphs)
        self.nodes = torch.zeros(count, max_nodes, dtype=torch.long)
        self.edges = torch.zeros(count, max_nodes, max_nodes, dtype=torch.long)
        self.mask = torch.zeros(count, max_nodes, dtype=torch.bool)
        self.sizes = torch.zeros(count, dtype=torch.long)
        for i, graph in enumerate(graphs):
            position = {node: j for j, node in enumerate(graph.nodes())}
            self.sizes[i] = len(position)
            self.mask[i, :len(position)] = True
            for node, j in position.items():
                label = graph.nodes[node].get("label", DEFAULT_NODE_LABEL)
                self.nodes[i, j] = self.node_index.get(label, 0)
            for u, v, data in graph.edges(data=True):
                a, b = position[u], position[v]
                index = self.edge_index.get(data.get("label", DEFAULT_EDGE_LABEL), 1)
                self.edges[i, a, b] = index
                self.edges[i, b, a] = index

    @property
    def num_node_types(self) -> int:
        return len(self.node_vocab)

    @property
    def num_edge_types(self) -> int:
        return len(self.edge_vocab)

    def marginals(self):
        """Training marginals over node types and edge types (including "no edge")."""
        node_counts = torch.zeros(self.num_node_types)
        edge_counts = torch.zeros(self.num_edge_types)
        for i in range(len(self.sizes)):
            n = int(self.sizes[i])
            node_counts += torch.bincount(self.nodes[i, :n], minlength=self.num_node_types)
            upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            edge_counts += torch.bincount(self.edges[i, :n, :n][upper], minlength=self.num_edge_types)
        return node_counts / node_counts.sum(), edge_counts / edge_counts.sum()

    def size_distribution(self):
        counts = torch.bincount(self.sizes, minlength=self.max_nodes + 1).float()
        return counts / counts.sum()


def build_vocabs(graphs):
    node_labels = sorted({data.get("label", DEFAULT_NODE_LABEL)
                          for g in graphs for _, data in g.nodes(data=True)})
    edge_labels = sorted({data.get("label", DEFAULT_EDGE_LABEL)
                          for g in graphs for _, _, data in g.edges(data=True)})
    return node_labels, [NO_EDGE] + edge_labels


def marginal_transition(alpha, marginal):
    """Q = alpha I + (1 - alpha) 1 m^T, batched over alpha."""
    classes = marginal.numel()
    eye = torch.eye(classes, device=marginal.device).unsqueeze(0)
    outer = marginal.view(1, 1, classes).expand(1, classes, classes)
    alpha = alpha.view(-1, 1, 1)
    return alpha * eye + (1 - alpha) * outer


class MarginalDiffusion:
    def __init__(self, node_marginal, edge_marginal, timesteps=500, cosine_s=0.008):
        self.timesteps = timesteps
        self.node_marginal = node_marginal
        self.edge_marginal = edge_marginal
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alphas_bar = torch.cos(0.5 * math.pi * (steps + cosine_s) / (1 + cosine_s)) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        self.alphas = (alphas_bar[1:] / alphas_bar[:-1]).clamp(min=0.001, max=1.0).float()
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def _matrices(self, table, t):
        alpha = table.to(t.device)[t - 1]
        return (marginal_transition(alpha, self.node_marginal.to(t.device)),
                marginal_transition(alpha, self.edge_marginal.to(t.device)))

    def transition(self, t):
        return self._matrices(self.alphas, t)

    def cumulative(self, t):
        return self._matrices(self.alphas_bar, t)


def sample_categorical(probabilities):
    flat = probabilities.reshape(-1, probabilities.shape[-1]).clamp(min=1e-12)
    flat = flat / flat.sum(dim=-1, keepdim=True)
    return torch.multinomial(flat, 1).reshape(probabilities.shape[:-1])


def symmetrize(edges):
    upper = torch.triu(edges, diagonal=1)
    return upper + upper.transpose(1, 2)


def mask_graph(nodes, edges, mask):
    """Zero the node and edge types outside each graph's real nodes."""
    nodes = torch.where(mask, nodes, torch.zeros_like(nodes))
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    return nodes, torch.where(pair_mask, edges, torch.zeros_like(edges))


# Structural features (DiGress 3.3): at high noise the graph is a draw from the marginal,
# so global topology is only visible to the denoiser through these.
K_EIGENVECTORS = 4
K_EIGENVALUES = 5
NODE_FEATURE_DIM = 1 + 2 + K_EIGENVECTORS       # degree, k3/k4, eigenvectors
GRAPH_FEATURE_DIM = 1 + 1 + K_EIGENVALUES + 3   # size, components, eigenvalues, k3/k4/k5
PAD_DIAGONAL = 5.0  # puts padded nodes' eigenvalues above the [0, 2] of real ones


def cycle_counts(adjacency):
    """Per-node 3- and 4-cycle counts, and graph-level 3-, 4- and 5-cycle counts, scaled by 1/10."""
    degree = adjacency.sum(dim=-1)
    a2 = adjacency @ adjacency
    a3 = a2 @ adjacency
    a4 = a3 @ adjacency
    a5 = a4 @ adjacency
    diag3 = torch.diagonal(a3, dim1=-2, dim2=-1)
    diag4 = torch.diagonal(a4, dim1=-2, dim2=-1)
    diag5 = torch.diagonal(a5, dim1=-2, dim2=-1)

    k3 = diag3 / 2
    k4 = (diag4 - degree * (degree - 1) - (adjacency @ degree.unsqueeze(-1)).squeeze(-1)) / 2
    # 10 C5 = tr(A^5) - 5 tr(A^3) - 5 sum_i (d_i - 2) (A^3)_ii; no clean per-node form
    k5_graph = (diag5.sum(-1) - 5 * diag3.sum(-1) - 5 * ((degree - 2) * diag3).sum(-1)) / 10

    node = torch.stack([k3, k4], dim=-1).clamp(min=0) / 10.0
    graph = torch.stack([k3.sum(-1) / 3, k4.sum(-1) / 4, k5_graph], dim=-1).clamp(min=0) / 10.0
    return node, graph


def spectral_features(adjacency, mask):
    """Lowest nonzero normalized-Laplacian eigenvalues, their eigenvectors, and the component count."""
    batch, n, _ = adjacency.shape
    adjacency = adjacency * (mask.unsqueeze(1) & mask.unsqueeze(2))
    inv_sqrt = adjacency.sum(dim=-1).clamp(min=1e-6).pow(-0.5)
    eye = torch.eye(n, device=adjacency.device).unsqueeze(0)
    # a zero diagonal for an isolated real node makes it count as its own component
    attached = (adjacency.sum(-1) > 0).unsqueeze(-1)
    laplacian = eye * (mask.unsqueeze(-1) & attached) \
        - inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)
    laplacian = laplacian + (~mask).unsqueeze(-1) * eye * PAD_DIAGONAL

    values, vectors = torch.linalg.eigh(laplacian.double())
    values, vectors = values.float(), vectors.float()
    is_real = values < (PAD_DIAGONAL / 2)
    components = (is_real & (values < 1e-4)).sum(dim=-1, keepdim=True).float()

    shifted = torch.where(is_real & (values >= 1e-4), values, torch.full_like(values, 1e4))
    eigenvalues = shifted.sort(dim=-1).values[:, :K_EIGENVALUES]
    eigenvalues = torch.where(eigenvalues > 1e3, torch.zeros_like(eigenvalues), eigenvalues)

    order = shifted.argsort(dim=-1)[:, :K_EIGENVECTORS]
    selected = torch.gather(vectors, 2, order.unsqueeze(1).expand(batch, n, K_EIGENVECTORS))
    # sign-canonicalize, so eigh's arbitrary sign is not something the model has to average over
    peak = selected.abs().argmax(dim=1, keepdim=True)
    sign = torch.gather(selected, 1, peak).sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return eigenvalues, selected * sign * mask.unsqueeze(-1), components


def structural_features(edge_onehot, mask):
    n = edge_onehot.shape[1]
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    adjacency = (edge_onehot[..., 1:].sum(-1) * pair_mask).float()
    adjacency = adjacency * (1 - torch.eye(n, device=adjacency.device)).unsqueeze(0)

    sizes = mask.sum(-1, keepdim=True).float()
    degree = (adjacency.sum(-1, keepdim=True) / (sizes - 1).clamp(min=1).unsqueeze(-1)) * mask.unsqueeze(-1)
    cycles_node, cycles_graph = cycle_counts(adjacency)
    eigenvalues, eigenvectors, components = spectral_features(adjacency, mask)
    node_features = torch.cat([degree, cycles_node * mask.unsqueeze(-1), eigenvectors], dim=-1)
    graph_features = torch.cat([sizes / n, components / sizes, eigenvalues, cycles_graph], dim=-1)
    return node_features, graph_features


class XEyTransformerLayer(nn.Module):
    """DiGress's layer: the attention scores are the new edge features, and the graph-level
    vector y (which carries the noise level) modulates nodes and edges by FiLM."""

    def __init__(self, node_dim, edge_dim, y_dim, heads, ffn_node=256, ffn_edge=128, ffn_y=128):
        super().__init__()
        assert node_dim % heads == 0
        self.heads = heads
        self.head_dim = node_dim // heads

        self.query = nn.Linear(node_dim, node_dim)
        self.key = nn.Linear(node_dim, node_dim)
        self.value = nn.Linear(node_dim, node_dim)

        self.edge_mul = nn.Linear(edge_dim, node_dim)
        self.edge_add = nn.Linear(edge_dim, node_dim)
        self.edge_out = nn.Linear(node_dim, edge_dim)

        self.y_to_edge_mul = nn.Linear(y_dim, edge_dim)
        self.y_to_edge_add = nn.Linear(y_dim, edge_dim)
        self.y_to_node_mul = nn.Linear(y_dim, node_dim)
        self.y_to_node_add = nn.Linear(y_dim, node_dim)

        self.node_out = nn.Linear(node_dim, node_dim)
        self.node_to_y = nn.Linear(node_dim, y_dim)
        self.edge_to_y = nn.Linear(edge_dim, y_dim)
        self.y_to_y = nn.Linear(y_dim, y_dim)

        self.node_norm1, self.node_norm2 = nn.LayerNorm(node_dim), nn.LayerNorm(node_dim)
        self.edge_norm1, self.edge_norm2 = nn.LayerNorm(edge_dim), nn.LayerNorm(edge_dim)
        self.y_norm1, self.y_norm2 = nn.LayerNorm(y_dim), nn.LayerNorm(y_dim)

        self.node_ffn = nn.Sequential(nn.Linear(node_dim, ffn_node), nn.ReLU(), nn.Linear(ffn_node, node_dim))
        self.edge_ffn = nn.Sequential(nn.Linear(edge_dim, ffn_edge), nn.ReLU(), nn.Linear(ffn_edge, edge_dim))
        self.y_ffn = nn.Sequential(nn.Linear(y_dim, ffn_y), nn.ReLU(), nn.Linear(ffn_y, y_dim))

    def forward(self, x, e, y, mask):
        batch, n, _ = x.shape
        node_mask = mask.unsqueeze(-1)
        pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)

        q = self.query(x).view(batch, n, 1, self.heads, self.head_dim)
        k = self.key(x).view(batch, 1, n, self.heads, self.head_dim)
        scores = ((q * k) / math.sqrt(self.head_dim)).reshape(batch, n, n, -1)
        scores = scores * (self.edge_mul(e) + 1) + self.edge_add(e)

        new_e = self.edge_out(scores) * pair_mask
        new_e = self.y_to_edge_add(y).unsqueeze(1).unsqueeze(1) + \
            (self.y_to_edge_mul(y).unsqueeze(1).unsqueeze(1) + 1) * new_e
        e = self.edge_norm1(e + new_e * pair_mask)
        e = self.edge_norm2(e + self.edge_ffn(e)) * pair_mask

        attention = scores.reshape(batch, n, n, self.heads, self.head_dim).sum(-1)
        attention = attention.masked_fill(~pair_mask.expand(-1, -1, -1, self.heads), float("-inf"))
        attention = torch.nan_to_num(torch.softmax(attention, dim=2))
        v = self.value(x).view(batch, n, self.heads, self.head_dim)
        gathered = torch.einsum("bijh,bjhd->bihd", attention, v).reshape(batch, n, -1)

        new_x = self.node_out(gathered)
        new_x = self.y_to_node_add(y).unsqueeze(1) + (self.y_to_node_mul(y).unsqueeze(1) + 1) * new_x
        x = self.node_norm1(x + new_x * node_mask)
        x = self.node_norm2(x + self.node_ffn(x)) * node_mask

        counts = mask.sum(-1, keepdim=True).clamp(min=1).float()
        pooled_x = (x * node_mask).sum(1) / counts
        pooled_e = (e * pair_mask).sum((1, 2)) / (counts ** 2)
        y = self.y_norm1(y + self.y_to_y(y) + self.node_to_y(pooled_x) + self.edge_to_y(pooled_e))
        y = self.y_norm2(y + self.y_ffn(y))
        return x, e, y


class Denoiser(nn.Module):
    # defaults are DiGress's SBM / planar configuration
    def __init__(self, num_node_types, num_edge_types, node_dim=256, edge_dim=64, y_dim=64,
                 layers=8, heads=8):
        super().__init__()
        self.node_in = nn.Sequential(
            nn.Linear(num_node_types + NODE_FEATURE_DIM, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim))
        self.edge_in = nn.Sequential(
            nn.Linear(num_edge_types, edge_dim), nn.ReLU(), nn.Linear(edge_dim, edge_dim))
        self.y_in = nn.Sequential(  # +1 for the noise level
            nn.Linear(GRAPH_FEATURE_DIM + 1, y_dim), nn.ReLU(), nn.Linear(y_dim, y_dim))
        self.layers = nn.ModuleList(
            XEyTransformerLayer(node_dim, edge_dim, y_dim, heads) for _ in range(layers))
        self.node_head = nn.Sequential(
            nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, num_node_types))
        self.edge_head = nn.Sequential(
            nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, num_edge_types))

    def forward(self, node_onehot, edge_onehot, mask, t_normalized):
        node_mask = mask.unsqueeze(-1)
        pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        node_features, graph_features = structural_features(edge_onehot, mask)
        x = self.node_in(torch.cat([node_onehot, node_features], dim=-1)) * node_mask
        e = self.edge_in(edge_onehot) * pair_mask
        y = self.y_in(torch.cat([graph_features, t_normalized.view(node_onehot.shape[0], 1)], dim=-1))
        for layer in self.layers:
            x, e, y = layer(x, e, y, mask)
        edge_logits = self.edge_head(e)
        return self.node_head(x), 0.5 * (edge_logits + edge_logits.transpose(1, 2))


class EMA:
    """Exponential moving average of the weights; samples come from it, as in DiGress."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value, alpha=1 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self):
        return self.shadow


def posterior(current_onehot, predicted_x0, q_t, qbar_prev):
    """p(z_{t-1} | z_t) = sum_x0 q(z_{t-1} | z_t, x0) p(x0 | z_t), for node (3-d) or edge (4-d) one-hots."""
    if current_onehot.dim() == 3:
        left = torch.einsum("bnc,bdc->bnd", current_onehot, q_t)
        numerator = left.unsqueeze(-2) * qbar_prev.unsqueeze(1)
    else:
        left = torch.einsum("bnmc,bdc->bnmd", current_onehot, q_t)
        numerator = left.unsqueeze(-2) * qbar_prev.unsqueeze(1).unsqueeze(1)
    # each q(. | z_t, x0) is normalized before mixing over x0, as in D3PM / DiGress
    per_x0 = numerator / numerator.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    mixture = (predicted_x0.unsqueeze(-1) * per_x0).sum(dim=-2)
    return mixture / mixture.sum(dim=-1, keepdim=True).clamp(min=1e-12)


class DiscreteGraphDiffusion:
    def __init__(self, data: GraphTensors, timesteps=500, device="cuda",
                 layers=8, node_dim=256, edge_dim=64, y_dim=64):
        self.data = data
        self.device = torch.device(device)
        self.process = MarginalDiffusion(*data.marginals(), timesteps)
        self.model = Denoiser(data.num_node_types, data.num_edge_types, node_dim=node_dim,
                              edge_dim=edge_dim, y_dim=y_dim, layers=layers).to(self.device)
        self.ema = EMA(self.model)
        self.size_distribution = data.size_distribution().to(self.device)
        self.timesteps = timesteps

    def _onehots(self, nodes, edges):
        return (F.one_hot(nodes, self.data.num_node_types).float(),
                F.one_hot(edges, self.data.num_edge_types).float())

    def _denoise(self, nodes, edges, mask, t):
        return self.model(*self._onehots(nodes, edges), mask, t.float() / self.timesteps)

    def add_noise(self, nodes, edges, mask, t):
        node_onehot, edge_onehot = self._onehots(nodes, edges)
        node_q, edge_q = self.process.cumulative(t)
        noisy_nodes = sample_categorical(torch.einsum("bnc,bcd->bnd", node_onehot, node_q))
        noisy_edges = symmetrize(sample_categorical(torch.einsum("bnmc,bcd->bnmd", edge_onehot, edge_q)))
        return mask_graph(noisy_nodes, noisy_edges, mask)

    def loss(self, nodes, edges, mask, t, edge_weight=5.0):
        node_logits, edge_logits = self._denoise(*self.add_noise(nodes, edges, mask, t), mask, t)
        node_loss = F.cross_entropy(node_logits[mask], nodes[mask], reduction="mean")
        pair_mask = torch.triu(mask.unsqueeze(1) & mask.unsqueeze(2), diagonal=1)
        edge_loss = F.cross_entropy(edge_logits[pair_mask], edges[pair_mask], reduction="mean")
        return node_loss + edge_weight * edge_loss

    @torch.no_grad()
    def sample(self, count) -> list[nx.Graph]:
        """Samples from the EMA weights."""
        backup = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict())
        try:
            return self._sample(count)
        finally:
            self.model.load_state_dict(backup)

    def _sample(self, count):
        self.model.eval()
        n_max = self.data.max_nodes
        sizes = torch.multinomial(self.size_distribution, count, replacement=True)
        mask = torch.arange(n_max, device=self.device).unsqueeze(0) < sizes.unsqueeze(1)
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        nodes = sample_categorical(self.process.node_marginal.to(self.device).expand(count, n_max, -1))
        edges = symmetrize(sample_categorical(
            self.process.edge_marginal.to(self.device).expand(count, n_max, n_max, -1)))
        nodes, edges = mask_graph(nodes, edges, mask)

        for step in range(self.timesteps, 0, -1):
            t = torch.full((count,), step, device=self.device, dtype=torch.long)
            node_logits, edge_logits = self._denoise(nodes, edges, mask, t)
            node_x0 = torch.softmax(node_logits, dim=-1)
            edge_x0 = torch.softmax(edge_logits, dim=-1)
            if step == 1:
                # sample x_0 rather than take the argmax, which would drop most edges
                nodes = sample_categorical(node_x0)
                edges = symmetrize(sample_categorical(edge_x0))
                edges = torch.where(pair_mask, edges, torch.zeros_like(edges))
                break
            node_q, edge_q = self.process.transition(t)
            node_qbar_prev, edge_qbar_prev = self.process.cumulative(t - 1)
            node_onehot, edge_onehot = self._onehots(nodes, edges)
            nodes = sample_categorical(posterior(node_onehot, node_x0, node_q, node_qbar_prev))
            edges = symmetrize(sample_categorical(posterior(edge_onehot, edge_x0, edge_q, edge_qbar_prev)))
            nodes, edges = mask_graph(nodes, edges, mask)
        return self._to_graphs(nodes.cpu(), edges.cpu(), sizes.cpu())

    def _to_graphs(self, nodes, edges, sizes):
        graphs = []
        for i in range(nodes.shape[0]):
            n = int(sizes[i])
            graph = nx.Graph()
            for j in range(n):
                graph.add_node(j, label=self.data.node_vocab[int(nodes[i, j])])
            for a in range(n):
                for b in range(a + 1, n):
                    index = int(edges[i, a, b])
                    if index != 0:
                        graph.add_edge(a, b, label=self.data.edge_vocab[index])
            graphs.append(graph)
        return graphs


def train(diffusion, data, epochs, batch_size, learning_rate, patience, seed,
          log_every=25, checkpoint_path="", save_every=25):
    torch.manual_seed(seed)
    device = diffusion.device
    count = len(data.sizes)
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    split_at = max(1, int(0.1 * count))
    valid_idx, train_idx = indices[:split_at], indices[split_at:]

    nodes, edges, mask = data.nodes.to(device), data.edges.to(device), data.mask.to(device)
    optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=learning_rate, weight_decay=1e-12)
    best, best_state, bad = float("inf"), None, 0
    last_saved = -10**9
    start = time.time()
    for epoch in range(1, epochs + 1):
        diffusion.model.train()
        random.shuffle(train_idx)
        total, batches = 0.0, 0
        for offset in range(0, len(train_idx), batch_size):
            chunk = torch.tensor(train_idx[offset:offset + batch_size], device=device)
            t = torch.randint(1, diffusion.timesteps + 1, (len(chunk),), device=device)
            loss = diffusion.loss(nodes[chunk], edges[chunk], mask[chunk], t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
            optimizer.step()
            diffusion.ema.update(diffusion.model)
            total += loss.item()
            batches += 1

        diffusion.model.eval()
        with torch.no_grad():
            chunk = torch.tensor(valid_idx, device=device)
            valid_total = 0.0
            # a fixed grid of timesteps keeps the validation loss low-variance
            for step in torch.linspace(1, diffusion.timesteps, 10).long().tolist():
                t = torch.full((len(chunk),), step, device=device, dtype=torch.long)
                valid_total += diffusion.loss(nodes[chunk], edges[chunk], mask[chunk], t).item()
            valid_loss = valid_total / 10

        if valid_loss < best - 1e-4:
            best, bad = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in diffusion.ema.state_dict().items()}
            # one overwriting file, so a crash late in a long run keeps the best model
            if checkpoint_path and epoch - last_saved >= save_every:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best_state, "epoch": epoch, "valid_loss": valid_loss, "ema": True},
                           str(checkpoint_path) + ".best")
                last_saved = epoch
        else:
            bad += 1

        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:5d}  train {total/max(batches,1):.4f}  valid {valid_loss:.4f}  "
                  f"best {best:.4f}  [{time.time()-start:.0f}s]", flush=True)
        if patience and bad >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        diffusion.ema.shadow = best_state
    print(f"  trained in {time.time()-start:.0f}s, best valid loss {best:.4f}")


def write_timing(timing, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(timing, handle, indent=2)
    print(f"  timing -> {path}: train {timing['train_seconds']:.1f}s, "
          f"sample {timing['sample_seconds']:.1f}s ({timing['graphs_per_second']:.2f} graphs/s), "
          f"{timing['params']:,} params")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", help="EDS graph file to train on")
    parser.add_argument("--benchmark", choices=["planar", "community_small"],
                        help="train on a standard benchmark instead of --train")
    parser.add_argument("--label-level", default="pos", choices=["none", "pos", "predicate"])
    parser.add_argument("-n", "--num-samples", type=int, default=1000)
    parser.add_argument("-o", "--output", required=True, help="output .pkl of sampled graphs")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--node-dim", type=int, default=256)
    parser.add_argument("--sample-batch", type=int, default=125)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--load-checkpoint", help="skip training and sample from this checkpoint")
    parser.add_argument("--timing-json", help="write wall-clock, peak GPU memory and parameter count here")
    args = parser.parse_args()
    if not args.train and not args.benchmark:
        parser.error("one of --train or --benchmark is required")
    return args


def main() -> int:
    args = parse_args()
    graphs = load_training_graphs(args.train, args.benchmark, args.label_level, args.seed)
    node_vocab, edge_vocab = build_vocabs(graphs)
    max_nodes = max(g.number_of_nodes() for g in graphs)
    data = GraphTensors(graphs, node_vocab, edge_vocab, max_nodes)
    print(f"  {len(graphs)} graphs, max nodes {max_nodes}, "
          f"{len(node_vocab)} node types, {len(edge_vocab)} edge types (incl. none)")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    diffusion = DiscreteGraphDiffusion(data, timesteps=args.timesteps, device=device,
                                       layers=args.layers, node_dim=args.node_dim)
    parameters = sum(p.numel() for p in diffusion.model.parameters())
    print(f"  denoiser parameters: {parameters/1e6:.2f}M, device {device}, T={args.timesteps}")

    timing = {"model": "digress", "label_level": args.label_level, "n_train_graphs": len(graphs),
              "device": str(device), "params": parameters, "epochs_budget": args.epochs,
              "batch_size": args.batch_size, "timesteps": args.timesteps, "layers": args.layers,
              "node_dim": args.node_dim, "num_samples": args.num_samples}
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.time()

    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        diffusion.model.load_state_dict(state["state_dict"])
        diffusion.ema.shadow = state.get("ema", state["state_dict"])
        print(f"  loaded {args.load_checkpoint}")
    else:
        train(diffusion, data, args.epochs, args.batch_size, args.lr, args.patience,
              args.seed, checkpoint_path=args.checkpoint or "")

    timing["train_seconds"] = time.time() - started
    if cuda:
        timing["peak_gpu_mb_train"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(args.seed + 1)
    samples = []
    started = time.time()
    while len(samples) < args.num_samples:
        samples.extend(diffusion.sample(min(args.sample_batch, args.num_samples - len(samples))))
        print(f"  sampled {len(samples)}/{args.num_samples} [{time.time()-started:.0f}s]", flush=True)
    samples = samples[:args.num_samples]
    timing["sample_seconds"] = time.time() - started
    timing["graphs_per_second"] = len(samples) / max(timing["sample_seconds"], 1e-9)
    timing["seconds_per_graph"] = timing["sample_seconds"] / max(len(samples), 1)
    if cuda:
        timing["peak_gpu_mb_sample"] = torch.cuda.max_memory_allocated() / 2**20
    if args.timing_json:
        write_timing(timing, args.timing_json)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(samples, handle)
    connected = np.mean([nx.is_connected(g) for g in samples if g.number_of_nodes() > 0])
    print(f"  wrote {args.output}: mean nodes {np.mean([g.number_of_nodes() for g in samples]):.1f}, "
          f"mean edges {np.mean([g.number_of_edges() for g in samples]):.1f}, connected {connected:.1%}")
    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": diffusion.model.state_dict(), "ema": diffusion.ema.state_dict(),
                    "node_vocab": node_vocab, "edge_vocab": edge_vocab, "max_nodes": max_nodes},
                   args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
