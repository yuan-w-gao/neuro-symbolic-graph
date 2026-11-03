"""GraphRNN (You et al., ICML 2018), optionally with a node-label channel, for the EDS baseline comparison."""

import argparse
import json
import pickle
import random
import sys
import time
from collections import deque
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


def bfs_order(graph: nx.Graph, start: int) -> list[int]:
    order = [start]
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(n for n in graph.neighbors(node) if n not in seen):
            seen.add(neighbour)
            order.append(neighbour)
            queue.append(neighbour)
    # disconnected remainder, so the sequence is still defined
    for node in graph.nodes():
        if node not in seen:
            order.append(node)
            seen.add(node)
    return order


def random_bfs(graph, rng):
    """Random node permutation and a random BFS root, as in the paper's training."""
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    relabelled = nx.relabel_nodes(graph, {old: i for i, old in enumerate(nodes)})
    return relabelled, bfs_order(relabelled, rng.randrange(relabelled.number_of_nodes()))


def encode_adjacency(adjacency: np.ndarray, bandwidth: int) -> np.ndarray:
    """Row i-1 holds node i's edges to nodes i-1, i-2, ... (nearest first)."""
    n = adjacency.shape[0]
    encoded = np.zeros((n - 1, bandwidth), dtype=np.float32)
    for i in range(1, n):
        row = adjacency[i, max(0, i - bandwidth):i][::-1]
        encoded[i - 1, :len(row)] = row
    return encoded


def decode_adjacency(encoded: np.ndarray) -> np.ndarray:
    steps, bandwidth = encoded.shape
    n = steps + 1
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n):
        low = max(0, i - bandwidth)
        adjacency[i, low:i] = encoded[i - 1, :i - low][::-1]
    return adjacency + adjacency.T


def random_bfs_encoding(graph, bandwidth, rng, label_index=None):
    """Encoded adjacency rows, and with label_index the labels of the nodes those rows introduce."""
    relabelled, order = random_bfs(graph, rng)
    encoded = encode_adjacency(nx.to_numpy_array(relabelled, nodelist=order), bandwidth)
    if label_index is None:
        return encoded, None
    # row i-1 introduces order[i]; the root's label is never predicted
    labels = np.array(
        [label_index.get(relabelled.nodes[node].get("label", "*"), 0) for node in order[1:]],
        dtype=np.int64)
    return encoded, labels


def estimate_bandwidth(graphs, percentile=100.0, trials=20, seed=0) -> int:
    """Largest predecessor distance seen over random BFS orderings."""
    rng = random.Random(seed)
    widths = []
    for graph in graphs:
        for _ in range(trials):
            relabelled, order = random_bfs(graph, rng)
            position = {node: i for i, node in enumerate(order)}
            widths.extend(abs(position[u] - position[v]) for u, v in relabelled.edges())
    return int(np.percentile(widths, percentile)) if widths else 1


class GRUStack(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 output_size=None, output_embedding_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_embedding = nn.Sequential(nn.Linear(input_size, embedding_size), nn.ReLU())
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.output = None
        if output_size is not None:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, output_embedding_size), nn.ReLU(),
                nn.Linear(output_embedding_size, output_size))
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("sigmoid"))
            elif "bias" in name:
                nn.init.constant_(param, 0.25)

    def initial_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, inputs, hidden):
        states, hidden = self.rnn(self.input_embedding(inputs), hidden)
        if self.output is None:
            return states, hidden
        return self.output(states), hidden


class GraphRNN(nn.Module):
    # num_labels > 0 adds a node-label channel, so the model covers the same joint
    # distribution (topology and labels) as the grammar it is compared against.
    def __init__(self, bandwidth: int, hidden_size=128, embedding_size=64, num_layers=4,
                 edge_hidden_size=16, edge_embedding_size=8, edge_output_size=64,
                 num_labels=0):
        super().__init__()
        self.bandwidth = bandwidth
        self.edge_hidden_size = edge_hidden_size
        self.num_labels = num_labels
        self.graph_rnn = GRUStack(bandwidth + num_labels, embedding_size, hidden_size, num_layers)
        self.label_head = None
        if num_labels:
            self.label_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, num_labels))
        self.to_edge_hidden = nn.Sequential(nn.Linear(hidden_size, edge_hidden_size), nn.ReLU())
        self.edge_rnn = GRUStack(1, edge_embedding_size, edge_hidden_size, num_layers,
                                 output_size=1, output_embedding_size=edge_output_size)

    def _edge_hidden(self, node_states):
        hidden = torch.zeros(self.edge_rnn.num_layers, node_states.size(0), self.edge_hidden_size,
                             device=node_states.device)
        hidden[0] = self.to_edge_hidden(node_states)
        return hidden

    def forward(self, sequences):
        """sequences: (B, T+1, M [+ L]), SOS row first. Returns edge and label logits (teacher-forced)."""
        batch = sequences.size(0)
        hidden = self.graph_rnn.initial_hidden(batch, sequences.device)
        node_states, _ = self.graph_rnn(sequences[:, :-1], hidden)
        targets = sequences[:, 1:, :self.bandwidth]
        # the edge RNN reads the previous bit of the row, with 1 as its SOS
        edge_inputs = torch.cat([torch.ones_like(targets[:, :, :1]), targets[:, :, :-1]], dim=2)
        steps = node_states.size(1)
        logits, _ = self.edge_rnn(
            edge_inputs.reshape(batch * steps, self.bandwidth, 1),
            self._edge_hidden(node_states.reshape(batch * steps, -1)))
        edge_logits = logits.reshape(batch, steps, self.bandwidth)
        label_logits = self.label_head(node_states) if self.label_head is not None else None
        return edge_logits, label_logits

    @torch.no_grad()
    def sample(self, num_graphs, max_nodes, device, label_vocab=None,
               root_label_probs=None) -> list[nx.Graph]:
        self.eval()
        hidden = self.graph_rnn.initial_hidden(num_graphs, device)
        current = torch.zeros(num_graphs, 1, self.bandwidth + self.num_labels, device=device)
        current[:, :, :self.bandwidth] = 1.0  # SOS
        rows, labels = [], []
        for _ in range(max_nodes):
            node_state, hidden = self.graph_rnn(current, hidden)
            edge_hidden = self._edge_hidden(node_state.squeeze(1))
            bit = torch.ones(num_graphs, 1, 1, device=device)
            sampled = torch.zeros(num_graphs, self.bandwidth, device=device)
            for j in range(self.bandwidth):
                logit, edge_hidden = self.edge_rnn(bit, edge_hidden)
                edge = torch.bernoulli(torch.sigmoid(logit.squeeze(-1).squeeze(-1)))
                sampled[:, j] = edge
                bit = edge.view(num_graphs, 1, 1)
            rows.append(sampled.cpu().numpy())

            current = torch.zeros_like(current)
            current[:, 0, :self.bandwidth] = sampled
            if self.label_head is not None:
                label_probs = torch.softmax(self.label_head(node_state.squeeze(1)), dim=-1)
                drawn = torch.multinomial(label_probs, 1).squeeze(-1)
                labels.append(drawn.cpu().numpy())
                current[:, 0, self.bandwidth:] = F.one_hot(drawn, self.num_labels).float()

        encoded = np.stack(rows, axis=1)
        label_array = np.stack(labels, axis=1) if labels else None
        return [decode_sample(encoded[i], None if label_array is None else label_array[i],
                              label_vocab, root_label_probs)
                for i in range(num_graphs)]


def decode_sample(rows, labels, label_vocab, root_label_probs):
    def draw_root_label():
        return int(torch.multinomial(root_label_probs, 1).item())

    def single_node():
        # a model that emits EOS immediately still has to return a graph
        graph = nx.empty_graph(1)
        if label_vocab is not None and root_label_probs is not None:
            graph.nodes[0]["label"] = label_vocab[draw_root_label()]
        return graph

    # the first all-zero row is the EOS the model was trained to emit
    zero_rows = np.where(~rows.any(axis=1))[0]
    end = int(zero_rows[0]) if len(zero_rows) else rows.shape[0]
    if end == 0:
        return single_node()
    graph = nx.from_numpy_array((decode_adjacency(rows[:end]) > 0.5).astype(int))
    if labels is not None and label_vocab is not None:
        # row j introduces node j+1; the root has no row, so its label comes from the training marginal
        for node in graph.nodes():
            index = draw_root_label() if node == 0 else int(labels[node - 1])
            graph.nodes[node]["label"] = label_vocab[index]
    # isolated nodes only: keeping the largest component would give this model a
    # connectivity guarantee the other models in the comparison do not get
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        return single_node()
    return nx.convert_node_labels_to_integers(graph)


def make_batch(graphs, bandwidth, max_steps, rng, label_index=None):
    """(sequences, lengths, label_targets); sequences are [SOS, row_1, ..., row_k, 0, ...].

    The length includes the all-zero row after the last real row: BFS order gives every
    real row an edge, so that row is the paper's EOS, and the model must be trained to emit it.
    """
    batch = len(graphs)
    num_labels = len(label_index) if label_index else 0
    sequences = torch.zeros(batch, max_steps + 2, bandwidth + num_labels)
    lengths = torch.zeros(batch, dtype=torch.long)
    # -100 is cross_entropy's ignore_index: padding and the EOS row have no node to label
    label_targets = torch.full((batch, max_steps + 1), -100, dtype=torch.long) if num_labels else None
    for i, graph in enumerate(graphs):
        encoded, labels = random_bfs_encoding(graph, bandwidth, rng, label_index)
        steps = min(encoded.shape[0], max_steps)
        sequences[i, 0, :bandwidth] = 1.0  # SOS
        sequences[i, 1:steps + 1, :bandwidth] = torch.from_numpy(encoded[:steps])
        if num_labels:
            taken = torch.from_numpy(labels[:steps])
            sequences[i, 1:steps + 1, bandwidth:] = F.one_hot(taken, num_labels).float()
            label_targets[i, :steps] = taken
        lengths[i] = steps + 1
    return sequences, lengths, label_targets


def masked_bce(logits, targets, lengths):
    mask = torch.arange(targets.size(1), device=logits.device)[None, :] < lengths[:, None].to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=2) * mask
    return loss.sum() / mask.sum().clamp(min=1)


def batch_losses(model, graphs, bandwidth, max_steps, rng, label_index, device):
    """(edge loss, node-label loss or None)."""
    sequences, lengths, label_targets = make_batch(graphs, bandwidth, max_steps, rng, label_index)
    sequences = sequences.to(device)
    edge_logits, label_logits = model(sequences)
    edge_loss = masked_bce(edge_logits, sequences[:, 1:, :bandwidth], lengths)
    if label_logits is None:
        return edge_loss, None
    return edge_loss, F.cross_entropy(label_logits.reshape(-1, label_logits.size(-1)),
                                      label_targets.to(device).reshape(-1), ignore_index=-100)


def train(graphs, bandwidth, max_steps, device, epochs=3000, batch_size=32, learning_rate=3e-3,
          patience=0, seed=0, log_every=100, hidden_size=128, milestones=(400, 1000),
          label_index=None, checkpoint_path="", save_every=100) -> GraphRNN:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    shuffled = list(graphs)
    rng.shuffle(shuffled)
    split = max(1, int(0.1 * len(shuffled)))
    valid_graphs, train_graphs = shuffled[:split], shuffled[split:]

    model = GraphRNN(bandwidth, hidden_size=hidden_size,
                     num_labels=len(label_index) if label_index else 0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # the reference implementation decays at absolute epochs 400 and 1000 of 3000
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=0.3)

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    last_saved = -10**9
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_graphs)
        total, batches = 0.0, 0
        for offset in range(0, len(train_graphs), batch_size):
            loss, label_loss = batch_losses(model, train_graphs[offset:offset + batch_size],
                                            bandwidth, max_steps, rng, label_index, device)
            if label_loss is not None:
                loss = loss + label_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            loss, label_loss = batch_losses(model, valid_graphs, bandwidth, max_steps, rng,
                                            label_index, device)
            valid_loss = loss.item()
            if label_loss is not None:
                valid_loss += label_loss.item()

        if valid_loss < best_loss - 1e-5:
            best_loss, bad_epochs = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # one overwriting file, so a crash late in a long run keeps the best model
            if checkpoint_path and epoch - last_saved >= save_every:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best_state, "epoch": epoch, "valid_loss": valid_loss},
                           str(checkpoint_path) + ".best")
                last_saved = epoch
        else:
            bad_epochs += 1

        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:5d}  train {total/max(batches,1):.5f}  valid {valid_loss:.5f}  "
                  f"best {best_loss:.5f}  [{time.time()-start:.0f}s]", flush=True)
        # off by default: the paper trains the full schedule, and stopping early can pre-empt an LR decay
        if patience and bad_epochs >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  trained in {time.time()-start:.0f}s, best valid loss {best_loss:.5f}")
    return model


def root_label_distribution(graphs, label_index):
    counts = torch.zeros(len(label_index))
    for graph in graphs:
        for _, data in graph.nodes(data=True):
            counts[label_index[data.get("label", "*")]] += 1
    return counts / counts.sum()


def save_samples(samples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".pkl"):
        with open(path, "wb") as handle:
            pickle.dump(samples, handle)
        return
    np.savez_compressed(path, **{f"g{i}": nx.to_numpy_array(g).astype(np.int8)
                                 for i, g in enumerate(samples)})


def write_timing(timing, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(timing, handle, indent=2)
    print(f"  timing -> {path}: train {timing['train_seconds']:.1f}s, "
          f"sample {timing['sample_seconds']:.1f}s ({timing['graphs_per_second']:.1f} graphs/s), "
          f"{timing['params']:,} params")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", help="EDS graph file to train on")
    parser.add_argument("--benchmark", choices=["planar", "community_small"],
                        help="train on a standard benchmark instead of --train")
    parser.add_argument("-n", "--num-samples", type=int, default=1000)
    parser.add_argument("-o", "--output", required=True, help="sampled graphs (.npz, or .pkl to keep labels)")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=0,
                        help="0 (default) trains the full schedule, as the paper does")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="graph-level GRU width; the paper uses 64 for the small datasets")
    parser.add_argument("--bandwidth", type=int, default=0, help="0 estimates M from the data")
    parser.add_argument("--milestones", type=int, nargs="*", default=[400, 1000])
    parser.add_argument("--label-level", default="none", choices=["none", "pos", "predicate"],
                        help="'none' is topology-only GraphRNN; otherwise add a node-label channel")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", help="where to save the trained model")
    parser.add_argument("--load-checkpoint", help="skip training and sample from this checkpoint")
    parser.add_argument("--timing-json", help="write wall-clock, peak GPU memory and parameter count here")
    args = parser.parse_args()
    if not args.train and not args.benchmark:
        parser.error("one of --train or --benchmark is required")
    return args


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    graphs = load_training_graphs(args.train, args.benchmark, args.label_level, args.seed)
    print(f"  {len(graphs)} training graphs, max nodes {max(g.number_of_nodes() for g in graphs)}")

    label_vocab = []
    if args.label_level != "none":
        label_vocab = sorted({data.get("label", "*") for g in graphs for _, data in g.nodes(data=True)})
        print(f"  node-label channel on: {len(label_vocab)} labels")
    label_index = {label: i for i, label in enumerate(label_vocab)} or None

    bandwidth = args.bandwidth or estimate_bandwidth(graphs, percentile=100.0, trials=20, seed=args.seed)
    max_steps = max(g.number_of_nodes() for g in graphs) - 1
    print(f"  bandwidth M = {bandwidth}, max steps = {max_steps}, device = {device}")

    timing = {"model": "graphrnn", "label_level": args.label_level, "n_train_graphs": len(graphs),
              "device": str(device), "epochs_budget": args.epochs, "batch_size": args.batch_size,
              "hidden_size": args.hidden_size, "bandwidth": bandwidth,
              "num_samples": args.num_samples}
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    started = time.time()

    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        bandwidth = state.get("bandwidth", bandwidth)
        model = GraphRNN(bandwidth, hidden_size=state.get("hidden_size", args.hidden_size),
                         num_labels=len(label_vocab)).to(device)
        model.load_state_dict(state["state_dict"])
        print(f"  loaded {args.load_checkpoint}")
    else:
        model = train(graphs, bandwidth, max_steps, device, epochs=args.epochs,
                      batch_size=args.batch_size, patience=args.patience, seed=args.seed,
                      hidden_size=args.hidden_size, milestones=args.milestones,
                      label_index=label_index, checkpoint_path=args.checkpoint or "")

    timing["train_seconds"] = time.time() - started
    timing["params"] = sum(p.numel() for p in model.parameters())
    if cuda:
        timing["peak_gpu_mb_train"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
    started = time.time()

    torch.manual_seed(args.seed + 1)
    root_probs = root_label_distribution(graphs, label_index) if label_vocab else None
    samples = []
    while len(samples) < args.num_samples:
        chunk = min(256, args.num_samples - len(samples))
        samples.extend(model.sample(chunk, max_steps + 1, device, label_vocab=label_vocab or None,
                                    root_label_probs=root_probs))
    samples = samples[:args.num_samples]
    timing["sample_seconds"] = time.time() - started
    timing["graphs_per_second"] = len(samples) / max(timing["sample_seconds"], 1e-9)
    if cuda:
        timing["peak_gpu_mb_sample"] = torch.cuda.max_memory_allocated() / 2**20
    sizes = [g.number_of_nodes() for g in samples]
    print(f"  sampled {len(samples)} graphs, mean nodes {np.mean(sizes):.1f}, max {max(sizes)}")

    save_samples(samples, args.output)
    print(f"  wrote {args.output}")
    if args.timing_json:
        write_timing(timing, args.timing_json)
    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "bandwidth": bandwidth,
                    "hidden_size": args.hidden_size, "label_vocab": label_vocab},
                   args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
