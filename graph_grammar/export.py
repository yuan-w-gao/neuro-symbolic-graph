import itertools

from .forest import PackedForest, ForestNode, HyperEdge

MAX_DERIVATIONS_PER_NODE = 100

# Part of the cache directory name: bump when export_forest output changes for the same
# chart, since the cache is otherwise keyed on the grammar hash alone.
# v2: _get_item_key gained the LHS nonterminal.
EXPORT_FORMAT_VERSION = 2


def _edge_indices(chart_item) -> tuple[int, ...]:
    return tuple(sorted(chart_item.edge_set.to_edge_indices()))


def _get_item_key(chart_item):
    # The parser keeps one subgraph derived as two nonterminals in two separate next_ptr
    # chains; keying on (edge_set, boundary) alone merged them and lost ~40% of derivations.
    grammar = chart_item.grammar
    if grammar is None:
        label_index, arity = -1, -1
    else:
        label_index, arity = grammar.label_index, len(grammar.external_nodes)
    return (hash(_edge_indices(chart_item)), tuple(chart_item.node_mapping), label_index, arity)


def _empty_forest(graph) -> PackedForest:
    return PackedForest(nodes=[], hyperedges=[], roots=[], num_rules=0, graph_id=graph.sentence_id)


def export_forest(context, graph, include_all_derivations: bool = True) -> PackedForest:
    result_item = context.result_item
    if result_item is None:
        return _empty_forest(graph)

    nodes = []
    hyperedges = []
    node_map = {}
    rule_ids = set()

    def get_or_create_node(chart_item, key):
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append(ForestNode(
                id=len(nodes), edge_set_hash=key[0], boundary_mapping=key[1],
                edge_indices=_edge_indices(chart_item)))
        return node_map[key]

    def visit(chart_item, path=frozenset()):
        key = _get_item_key(chart_item)
        if key in path:
            return node_map[key]
        node_id = get_or_create_node(chart_item, key)
        node = nodes[node_id]
        if node.incoming_edges:
            return node_id
        path = path | {key}

        items = [chart_item]
        if include_all_derivations:
            items = list(itertools.islice(chart_item.all(), MAX_DERIVATIONS_PER_NODE))

        for item in items:
            if item.grammar is None:
                continue
            rule_ids.add(item.grammar_index)
            try:
                left, right = context.split_item(item)
            except Exception:
                left, right = None, None
            # A plain loop, not a generator: deep forests are near the recursion limit.
            children = []
            for child in (left, right):
                if child is not None:
                    children.append(visit(child, path))

            edge_id = len(hyperedges)
            hyperedges.append(HyperEdge(
                id=edge_id, rule_id=item.grammar_index, parent_node=node_id,
                child_nodes=tuple(children), cfg_rule_id=item.cfg_index))
            node.incoming_edges.append(edge_id)
            if not children:
                node.is_terminal = True
        return node_id

    root_id = visit(result_item)
    nodes[root_id].is_root = True
    return PackedForest(
        nodes=nodes,
        hyperedges=hyperedges,
        roots=[root_id],
        num_rules=max(rule_ids) + 1 if rule_ids else 0,
        graph_id=graph.sentence_id,
    )


def export_single_derivation(context, graph) -> PackedForest:
    derivation = context.derivation
    if len(derivation) == 0:
        return _empty_forest(graph)

    nodes = []
    hyperedges = []
    rule_ids = set()
    for i, deriv_node in enumerate(derivation):
        item = deriv_node.item
        children = deriv_node.children
        rule_id = item.grammar_index if item else -1
        if rule_id >= 0:
            rule_ids.add(rule_id)
        edge_indices = _edge_indices(item) if item else ()

        nodes.append(ForestNode(
            id=i,
            edge_set_hash=hash(edge_indices),
            boundary_mapping=tuple(item.node_mapping) if item else (),
            incoming_edges=[i],
            is_terminal=len(children) == 0,
            is_root=i == 0,
            edge_indices=edge_indices,
        ))
        hyperedges.append(HyperEdge(id=i, rule_id=rule_id, parent_node=i, child_nodes=tuple(children)))

    return PackedForest(
        nodes=nodes,
        hyperedges=hyperedges,
        roots=[0],
        num_rules=max(rule_ids) + 1 if rule_ids else 0,
        graph_id=graph.sentence_id,
    )
