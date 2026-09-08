import ast
import os


def _extract_required_symbols_from_auto_map(auto_map: dict | None) -> set[str]:
    required: set[str] = set()
    if not isinstance(auto_map, dict):
        return required

    for value in auto_map.values():
        ref = value
        if isinstance(value, (list, tuple)) and len(value) > 0:
            ref = value[0]

        if not isinstance(ref, str) or "." not in ref:
            continue

        _module_name, symbol_name = ref.rsplit(".", 1)
        if symbol_name:
            required.add(symbol_name)

    return required


def _collect_def_dependency_closure(module_ast: ast.Module, required: set[str]) -> set[str]:
    defs_by_name = {
        node.name: node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    selected = set(required)

    missing = [name for name in selected if name not in defs_by_name]
    if missing:
        raise ValueError(
            "Could not find required symbol(s) in modeling module: "
            + ", ".join(sorted(missing))
        )

    changed = True
    while changed:
        changed = False
        current = list(selected)
        for name in current:
            node = defs_by_name.get(name)
            if node is None:
                continue

            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    dep = child.id
                    if dep in defs_by_name and dep not in selected:
                        selected.add(dep)
                        changed = True

    return selected


def _slice_source_by_node(source_lines: list[str], node: ast.AST) -> str:
    if getattr(node, "lineno", None) is None or getattr(node, "end_lineno", None) is None:
        return ""
    return "".join(source_lines[node.lineno - 1 : node.end_lineno])


def export_pruned_modeling_from_auto_map(
    source_modeling_path: str,
    output_modeling_path: str,
    auto_map: dict | None,
) -> None:
    required_symbols = _extract_required_symbols_from_auto_map(auto_map)

    if not required_symbols:
        raise ValueError(
            "Cannot export pruned modeling.py because config.auto_map does not contain"
            " any class references."
        )

    if not os.path.exists(source_modeling_path):
        raise ValueError(f"modeling.py not found at {source_modeling_path}")

    with open(source_modeling_path, "r", encoding="utf-8") as f:
        source_text = f.read()

    module_ast = ast.parse(source_text)
    source_lines = source_text.splitlines(keepends=True)

    selected_defs = _collect_def_dependency_closure(module_ast, required_symbols)

    out_parts: list[str] = []
    out_parts.append(
        "# Auto-generated during model export.\n"
        "# Contains only symbols required by config.auto_map for this checkpoint.\n\n"
    )

    for node in module_ast.body:
        is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if is_def and getattr(node, "name", None) not in selected_defs:
            continue

        chunk = _slice_source_by_node(source_lines, node)
        if chunk:
            out_parts.append(chunk)
            if not chunk.endswith("\n\n"):
                out_parts.append("\n")

    os.makedirs(os.path.dirname(output_modeling_path), exist_ok=True)
    with open(output_modeling_path, "w", encoding="utf-8") as f:
        f.write("".join(out_parts).rstrip() + "\n")
