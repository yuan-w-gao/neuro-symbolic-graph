import gzip, re, sys, types, importlib.util, os

for name in ("delphin", "delphin.codecs", "delphin.codecs.mrsjson",
             "delphin.codecs.eds", "delphin.eds", "delphin.mrs"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["delphin"].codecs = sys.modules["delphin.codecs"]
sys.modules["delphin.codecs"].mrsjson = sys.modules["delphin.codecs.mrsjson"]
sys.modules["delphin.codecs"].eds = sys.modules["delphin.codecs.eds"]
sys.modules["delphin"].eds = sys.modules["delphin.eds"]
sys.modules["delphin"].mrs = sys.modules["delphin.mrs"]

spec = importlib.util.spec_from_file_location(
    "p2e", os.path.join(os.path.dirname(os.path.abspath(__file__)), "parse_to_extraction.py"))
p2e = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2e)

TOKEN = r'"token \[.*?\]"'
LEAF = re.compile(r'\("((?:[^"\\]|\\.)*)"((?:\s*\d+\s*' + TOKEN + r')+)\)', re.S)
TOKEN_RE = re.compile(TOKEN, re.S)
COREF = re.compile(r'#(\d+)=\\"([^\\]*)\\"')

def _fs_field(tok, name, coref):
    m = re.search(r'\+' + name + r' (?:#\d+=)?\\"([^\\]*)\\"', tok)
    if m: return m.group(1)
    m = re.search(r'\+' + name + r' #(\d+)(?![\d=])', tok)
    return coref.get(m.group(1)) if m else None

def _spans(tokens_blob):
    out = []
    for tok in TOKEN_RE.findall(tokens_blob):
        coref = dict(COREF.findall(tok))
        f, t = _fs_field(tok, "FROM", coref), _fs_field(tok, "TO", coref)
        if f is not None and t is not None:
            try: out.append((int(f), int(t)))
            except ValueError: pass
    return out

def _leaf_text(surface, tokens_blob):
    words = [w for w in surface.replace('\\"', '"').split(" ") if w]
    spans = _spans(tokens_blob)
    if not words:
        words = ['"']
    if len(words) == len(spans):
        return "_".join(f"{w}#__#[{s},{e}]" for w, (s, e) in zip(words, spans))
    if spans:
        s, e = min(x[0] for x in spans), max(x[1] for x in spans)
        return f"{'*'.join(words)}#__#[{s},{e}]"
    return f"{'*'.join(words)}#__#[NotExist]"

def _fix_lexical_terms(node):
    if node.type == "LEXICAL_RULE" and len(node.children) == 1 \
            and node.children[0].type == "LEXICAL":
        node.lexical_term = node.children[0].content
        node.span = (0, 0)
    for c in node.children:
        _fix_lexical_terms(c)

def deriv_to_java_tree(deriv):
    leaves, holder = [], {}
    def sub(m):
        k = f"<<L{len(leaves)}>>"
        leaves.append(_leaf_text(m.group(1), m.group(2)))
        return f'("{k}")'
    stripped = LEAF.sub(sub, deriv)
    stripped = re.sub(r'\s*\d+\s*' + TOKEN, '', stripped, flags=re.S)
    tree = p2e.parse_tree(stripped)
    if tree is None:
        return None
    p2e.update_node_type(tree)
    _fix_lexical_terms(tree)
    ser = re.sub(r" +", " ", p2e.serialize_tree(tree).strip())
    def unsub(m):
        return leaves[int(m.group(1))]
    ser = re.sub(r"<<L(\d+)>>#__#\[0,0\]", unsub, ser)
    if "<<L" in ser:
        return None
    return f"(ROOT {ser})"

def item_deriv(path):
    with gzip.open(path, "rt", errors="replace") as f:
        fields = f.read().split("\n\n")
    return fields[4] if len(fields) > 5 else None
