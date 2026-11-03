import gzip
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional

from delphin.codecs import mrsjson
from delphin import eds, mrs
from delphin.codecs import eds as eds_codecs


def extract_tokens_from_deriv(deriv_str: str) -> dict:
    form_to_spans = defaultdict(list)

    pattern = r'\+FORM \\\"([^\\]+)\\\" \+FROM \\\"(\d+)\\\" \+TO \\\"(\d+)\\\"'
    matches = re.findall(pattern, deriv_str)

    for form, from_pos, to_pos in matches:
        form_lower = form.lower()
        form_to_spans[form_lower].append((int(from_pos), int(to_pos)))

    return form_to_spans


@dataclass
class Node:
    type: str
    content: str
    span: Tuple[int, int]
    lexical_term: str = " "
    children: List['Node'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def parse_node(line):
    if not line:
        return None
    current_line = line.strip()
    if not current_line:
        return None

    tokens = re.findall(r'"[^"]*"|[^\s"]+', current_line)

    if tokens[0].startswith('"'):
        c = tokens[0].strip('"').replace('^', '')
        c = c.replace(" ", "*")
        node = Node(type="LEXICAL", content=c, span=(0, 0))
    elif len(tokens) == 1:
        cont = tokens[0].lower().replace('^', '')
        node = Node(type="ROOT", content=cont, span=(0, 0), children=[])
    else:
        assert len(tokens) == 5, f"Expected 5 tokens, got {len(tokens)}: {tokens}"
        node_label = tokens[1].lower()
        start = int(tokens[3])
        end = int(tokens[4])
        if '@' in node_label:
            node_type = 'LEXICAL_RULE'
            cont = node_label.split('@')[1].replace('^', '')
            lexical_term = node_label.split('@')[0].split('_')[0]
        else:
            node_type = 'NORMAL'
            cont = node_label
            lexical_term = ""

        node = Node(type=node_type, content=cont, span=(start, end), lexical_term=lexical_term)

    return node


def update_node_type(node):
    if node.type == "NORMAL":
        if len(node.children) == 1:
            child = node.children[0]
            if child.type == "LEXICAL":
                node.type = 'LEXICAL_RULE'
                node.lexical_term = child.content
                return

    for child in node.children:
        update_node_type(child)


def update_spans_from_deriv(node, form_to_spans, used_spans=None):
    if used_spans is None:
        used_spans = set()

    space_flag = False

    if node.type == "LEXICAL_RULE":
        assert len(node.children) == 1
        child = node.children[0]
        assert child.type == "LEXICAL"
        assert len(child.children) == 0

        content = child.content.lower().replace("'", "'").replace("*", " ")
        if " " in content:
            space_flag = True

        available_spans = form_to_spans.get(content, [])
        selected_span = None
        for span in available_spans:
            if span not in used_spans:
                selected_span = span
                break

        if selected_span:
            used_spans.add(selected_span)
            child.span = selected_span
            node.span = selected_span

            rule_lex = node.lexical_term
            child_lex = child.content
            if rule_lex != child_lex:
                node.lexical_term = child_lex

    min_span = float('inf')
    max_span = -1
    children_flag = False

    for child_node in node.children:
        child_flag, _ = update_spans_from_deriv(child_node, form_to_spans, used_spans)
        children_flag = children_flag or child_flag

        if hasattr(child_node, 'span') and child_node.span != (0, 0):
            min_span = min(min_span, child_node.span[0])
            max_span = max(max_span, child_node.span[1])

    if node.type == "NORMAL" and node.children:
        if min_span != float('inf') and max_span != -1:
            node.span = (min_span, max_span)

    max_position = max_span if max_span != -1 else 0
    return (space_flag or children_flag), max_position


def serialize_tree(node):
    if node.type in ("NORMAL", "ROOT"):
        content = node.content
        children_serialized = ' '.join(serialize_tree(child) for child in node.children)
    elif node.type == "LEXICAL_RULE" and node.children:
        start_pos, end_pos = node.children[0].span
        content = f"{node.content} {node.lexical_term}#__#[{start_pos},{end_pos}]"
        children_serialized = None
    else:
        content = ''
        children_serialized = None

    if children_serialized:
        return f" ({content.replace('^', '')} {children_serialized.replace('^', '')})"
    else:
        return f" ({content.replace('^', '')})"


def tokenize(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    tokens = []
    for line in lines:
        current = ''
        for char in line:
            if char in '()':
                if current:
                    tokens.append(current)
                    current = ''
                tokens.append(char)
            else:
                current += char
        if current:
            tokens.append(current)
    return tokens


def parse_tree(bracketed_str):
    stack = []
    root = None

    lines = tokenize(bracketed_str)
    lines = [s for s in lines if not s.isdigit()]

    for line in lines:
        if line == '(':
            continue
        elif line == ')':
            if stack:
                stack.pop()
        else:
            node = parse_node(line)
            if stack:
                stack[-1].children.append(node)
            else:
                root = node
            stack.append(node)

    return root


def transform_tree(text, sentence=None):
    form_to_spans = extract_tokens_from_deriv(text)

    res = re.sub(r'\d+\s+"token\s*\[.*?\]"', '', text, flags=re.MULTILINE).strip()
    tree = parse_tree(res)
    if tree is None:
        return False, None
    update_node_type(tree)

    flag, _ = update_spans_from_deriv(tree, form_to_spans)
    serialized = serialize_tree(tree)
    return flag, f"(ROOT (root_strict {serialized}))"


def create_export_content(sentence_id: int, sentence: str, deriv: str, eds_str: str, template_fields: list) -> str:
    fields = template_fields.copy()

    fields[1] = f"[{sentence_id}] (1 of 1) " + "{1} `" + sentence + "'"

    fields[4] = deriv

    fields[7] = eds_str

    if len(fields) > 8:
        fields[8] = eds_str

    return '\n\n'.join(fields)


def mrs_to_eds(mrs_str: str) -> Optional[str]:
    try:
        mrs_obj = mrsjson.decode(mrs_str)
        eds_obj = eds.from_mrs(mrs_obj)
        eds_str = eds_codecs.encode(eds_obj)
        return eds_str
    except Exception as e:
        return None


def load_template(template_path: str) -> list:
    with open(template_path, 'r') as f:
        content = f.read()
    return content.split('\n\n')


def process_data(input_path: str, template_path: str, output_base: str):
    print(f"Loading template from {template_path}...")
    template_fields = load_template(template_path)

    export_dir = os.path.join(output_base, 'export')
    java_out_dir = os.path.join(output_base, 'java_out')
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    Path(java_out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Reading and grouping data from {input_path}...")
    age_groups = defaultdict(list)

    total_count = 0
    success_count = 0
    skip_no_parse = 0
    skip_transform_fail = 0
    skip_eds_fail = 0

    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0 and line_num > 0:
                print(f"  Processed {line_num} lines...")

            total_count += 1
            entry = json.loads(line)

            if not entry.get('parse_successful', False):
                skip_no_parse += 1
                continue

            ages = entry.get('age', [])
            if not ages:
                skip_no_parse += 1
                continue

            unique_ages = sorted(set(int(a) for a in ages))

            parses = entry.get('parses', [])
            if not parses:
                skip_no_parse += 1
                continue

            first_parse = parses[0]
            mrs_str = first_parse.get('mrs', '')
            deriv_str = first_parse.get('deriv', '')
            sentence = entry.get('text', '').replace("'", "'")

            if not mrs_str or not deriv_str:
                skip_no_parse += 1
                continue

            try:
                space_flag, java_tree = transform_tree(deriv_str, sentence.lower())
                if java_tree is None:
                    skip_transform_fail += 1
                    continue
            except Exception as e:
                skip_transform_fail += 1
                continue

            eds_str = mrs_to_eds(mrs_str)
            if eds_str is None:
                skip_eds_fail += 1
                continue

            success_count += 1

            entry_data = {
                'sentence': sentence,
                'deriv': deriv_str,
                'mrs': mrs_str,
                'eds': eds_str,
                'java_tree': java_tree,
                'ages': ages,
                'ids': entry.get('ids', [])
            }
            for age in unique_ages:
                age_groups[age].append(entry_data)

    print(f"\nProcessing summary:")
    print(f"  Total entries: {total_count}")
    print(f"  Successful: {success_count}")
    print(f"  Skipped (no parse): {skip_no_parse}")
    print(f"  Skipped (transform fail): {skip_transform_fail}")
    print(f"  Skipped (EDS fail): {skip_eds_fail}")
    print(f"  Age groups: {len(age_groups)}")

    print(f"\nWriting output files...")
    for age in sorted(age_groups.keys()):
        entries = age_groups[age]
        age_name = f"{age:02d}"

        age_export_dir = os.path.join(export_dir, age_name)
        Path(age_export_dir).mkdir(parents=True, exist_ok=True)

        java_trees = []

        for idx, entry in enumerate(entries):
            export_content = create_export_content(
                sentence_id=idx,
                sentence=entry['sentence'],
                deriv=entry['deriv'],
                eds_str=entry['eds'],
                template_fields=template_fields
            )

            export_path = os.path.join(age_export_dir, f"{idx}.gz")
            with gzip.open(export_path, 'wt', encoding='utf-8') as f:
                f.write(export_content)

            java_trees.append(f"#{idx}\n{entry['java_tree']}")

        java_out_path = os.path.join(java_out_dir, age_name)
        with open(java_out_path, 'w') as f:
            f.write('\n'.join(java_trees))

        print(f"  {age_name}: {len(entries)} entries")

    print(f"\nDone! Output written to {output_base}")


def main():
    input_path = "data/processed/parsed/all_childes_normalized_parsed_with_metadata.jsonl"
    template_path = "deepbank1.1/export/wsj00a/20001001"
    output_base = "data/processed/for_extraction"

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        template_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_base = sys.argv[3]

    process_data(input_path, template_path, output_base)


if __name__ == '__main__':
    main()
