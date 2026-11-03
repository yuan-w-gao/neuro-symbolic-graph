import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hypergraph_embeddings import CachedRuleEmbedding, GraphStructuralRuleEmbedding, node_feature_rows


def _segment_softmax(scores: Tensor, seg: Tensor, num_segments: int) -> Tensor:
    if scores.numel() == 0:
        return scores
    m = torch.full((num_segments,), -1e30, dtype=scores.dtype, device=scores.device)
    m = m.scatter_reduce(0, seg, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - m[seg])
    den = torch.zeros(num_segments, dtype=scores.dtype, device=scores.device).index_add(0, seg, ex)
    return ex / den[seg].clamp_min(1e-30)


def _segment_mean(x: Tensor, seg: Tensor, num_segments: int) -> Tensor:
    out = torch.zeros(num_segments, x.shape[-1], dtype=x.dtype, device=x.device)
    if not x.numel():
        return out
    out = out.index_add(0, seg, x)
    cnt = torch.zeros(num_segments, dtype=x.dtype, device=x.device)
    cnt = cnt.index_add(0, seg, torch.ones_like(seg, dtype=x.dtype))
    return out / cnt.clamp_min(1.0).unsqueeze(-1)


class BatchedHypergraphConv(nn.Module):
    """Shares the submodules of a HypergraphConvLayer, under the same names."""

    def __init__(self, ref_layer: nn.Module):
        super().__init__()
        self.node_to_edge_key = ref_layer.node_to_edge_key
        self.node_to_edge_value = ref_layer.node_to_edge_value
        self.node_to_edge_query = ref_layer.node_to_edge_query
        self.edge_to_node_key = ref_layer.edge_to_node_key
        self.edge_to_node_value = ref_layer.edge_to_node_value
        self.edge_to_node_query = ref_layer.edge_to_node_query
        self.edge_update = ref_layer.edge_update
        self.node_update = ref_layer.node_update
        self.edge_norm = ref_layer.edge_norm
        self.node_norm = ref_layer.node_norm
        self.dropout = ref_layer.dropout
        self.hidden_dim = ref_layer.hidden_dim
        self.scale = ref_layer.scale

    def _attend(self, queries, keys, values, target, source, num_targets):
        scores = (queries[target] * keys[source]).sum(-1) * self.scale
        attn = self.dropout(_segment_softmax(scores, target, num_targets))
        out = torch.zeros(num_targets, self.hidden_dim, dtype=values.dtype, device=values.device)
        return out.index_add(0, target, attn.unsqueeze(-1) * values[source])

    def forward(self, node_emb: Tensor, edge_emb: Tensor,
                inc_e: Tensor, inc_n: Tensor) -> tuple[Tensor, Tensor]:
        nk = self.node_to_edge_key(node_emb)
        nv = self.node_to_edge_value(node_emb)
        eq = self.node_to_edge_query(edge_emb)
        edge_agg = self._attend(eq, nk, nv, inc_e, inc_n, edge_emb.shape[0])
        new_edge = F.relu(self.edge_norm(self.edge_update(torch.cat([edge_agg, edge_emb], dim=-1))))

        ek = self.edge_to_node_key(new_edge)
        ev = self.edge_to_node_value(new_edge)
        nq = self.edge_to_node_query(node_emb)
        node_agg = self._attend(nq, ek, ev, inc_n, inc_e, node_emb.shape[0])
        new_node = F.relu(self.node_norm(self.node_update(torch.cat([node_agg, node_emb], dim=-1))))
        return new_node, new_edge


class BatchedGraphStructuralRuleEmbedding(CachedRuleEmbedding):
    """GraphStructuralRuleEmbedding with all rules laid out as one disjoint hypergraph, no Python loops."""

    NODE_FEATURE_DIM = GraphStructuralRuleEmbedding.NODE_FEATURE_DIM

    def __init__(self, grammar_file: str, embedding_dim: int = 64, hidden_dim: int = 128,
                 num_gnn_layers: int = 2, edge_embed_dim: int = 32, dropout: float = 0.0,
                 device: torch.device | None = None,
                 ref_module: GraphStructuralRuleEmbedding | None = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # built through the reference module so initialization and parameter names match it
        ref = ref_module if ref_module is not None else GraphStructuralRuleEmbedding(
            grammar_file, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers, edge_embed_dim=edge_embed_dim,
            dropout=dropout, device=self.device)
        self.extractor = ref.extractor
        self.edge_label_embedding = ref.edge_label_embedding
        self.edge_feature_dim = ref.edge_feature_dim
        self.boundary_encoder = ref.boundary_encoder
        self.output_proj = ref.output_proj
        self.node_input_proj = ref.gnn.node_input_proj
        self.edge_input_proj = ref.gnn.edge_input_proj
        self.layers = nn.ModuleList([BatchedHypergraphConv(layer) for layer in ref.gnn.layers])
        self._build_flat_index()
        self.to(self.device)

    def _build_flat_index(self):
        structures = self.extractor.structures
        max_ext = max(10, self.extractor.max_external_nodes)
        node_feats, labels, is_terminal, arity = [], [], [], []
        inc_e, inc_n, node_rule, edge_rule, ext_rows, degenerate = [], [], [], [], [], []
        n_off = e_off = 0
        for r, d in enumerate(structures):
            degenerate.append(d.num_nodes == 0 or d.num_edges == 0)
            node_feats += node_feature_rows(d)
            node_rule += [r] * d.num_nodes
            labels += d.edge_label_indices
            is_terminal += [1.0 if t else 0.0 for t in d.edge_is_terminal]
            arity += [a / 5.0 for a in d.edge_arities]
            edge_rule += [r] * d.num_edges
            for j, nodes in enumerate(d.edge_to_nodes):
                inc_e += [e_off + j] * len(nodes)
                inc_n += [n_off + n for n in nodes]
            valid = [i for i in d.external_node_indices if 0 <= i < d.num_nodes][:max_ext]
            ext_rows.append([n_off + i for i in valid])
            n_off += d.num_nodes
            e_off += d.num_edges

        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device=self.device)

        ext_idx = torch.zeros(len(structures), max_ext, dtype=torch.long, device=self.device)
        ext_pad_mask = torch.ones(len(structures), max_ext, dtype=torch.bool, device=self.device)
        for r, row in enumerate(ext_rows):
            if row:
                ext_idx[r, :len(row)] = tensor(row, torch.long)
                ext_pad_mask[r, :len(row)] = False
        buffers = {
            "node_features": tensor(node_feats, torch.float32) if node_feats
            else torch.zeros(0, self.NODE_FEATURE_DIM, device=self.device),
            "edge_label_idx": tensor(labels, torch.long),
            "edge_is_terminal": tensor(is_terminal, torch.float32).unsqueeze(-1),
            "edge_arity": tensor(arity, torch.float32).unsqueeze(-1),
            "inc_e": tensor(inc_e, torch.long),
            "inc_n": tensor(inc_n, torch.long),
            "node_rule": tensor(node_rule, torch.long),
            "edge_rule": tensor(edge_rule, torch.long),
            "degenerate": tensor(degenerate, torch.bool),
            "ext_idx": ext_idx,
            "ext_pad_mask": ext_pad_mask,
        }
        for name, value in buffers.items():
            self.register_buffer(name, value, persistent=False)

    def _boundary(self, node_emb: Tensor) -> Tensor:
        R, M = self.ext_idx.shape
        be = self.boundary_encoder
        live = ~self.ext_pad_mask
        has_ext = live.any(dim=1)
        e = node_emb[self.ext_idx.reshape(-1)].reshape(R, M, -1)
        e = e * live.unsqueeze(-1)
        e = e + be.position_embedding(torch.arange(M, device=node_emb.device)).unsqueeze(0)
        # a fully padded row would softmax to NaN: give it one live slot, zero its output below
        key_mask = self.ext_pad_mask.clone()
        key_mask[~has_ext, 0] = False
        attn, _ = be.self_attention(e, e, e, key_padding_mask=key_mask)
        z = be.layer_norm(e + attn)
        keep = live.unsqueeze(-1).to(z.dtype)
        out = be.output_proj((z * keep).sum(1) / keep.sum(1).clamp_min(1.0))
        return out * has_ext.unsqueeze(-1).to(out.dtype)

    def _compute_embeddings(self) -> Tensor:
        edge_feat = torch.cat([self.edge_label_embedding(self.edge_label_idx),
                               self.edge_is_terminal, self.edge_arity], dim=-1)
        node_emb = F.relu(self.node_input_proj(self.node_features))
        edge_emb = F.relu(self.edge_input_proj(edge_feat))
        deg_n = self.degenerate[self.node_rule].unsqueeze(-1)
        deg_e = self.degenerate[self.edge_rule].unsqueeze(-1)
        for layer in self.layers:
            node_emb, edge_emb = layer(node_emb, edge_emb, self.inc_e, self.inc_n)
            # the reference layer returns zeros for a rule with no nodes or no edges
            node_emb = node_emb.masked_fill(deg_n, 0.0)
            edge_emb = edge_emb.masked_fill(deg_e, 0.0)
        node_pool = _segment_mean(node_emb, self.node_rule, self.num_rules)
        edge_pool = _segment_mean(edge_emb, self.edge_rule, self.num_rules)
        out = self.output_proj(torch.cat([node_pool, edge_pool, self._boundary(node_emb)], dim=-1))
        return out.masked_fill(self.degenerate.unsqueeze(-1), 0.0)
