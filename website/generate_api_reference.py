"""Generate API reference Markdown files from docstrings using griffe2md."""

from __future__ import annotations

import sys
from pathlib import Path

from griffe import GriffeLoader, Object, Parser
from griffe2md import ConfigDict, default_config, render_object_docs

OUTPUT_DIR = Path(__file__).parent / "src" / "content" / "docs" / "reference"

MODULES: dict[str, dict[str, str | int]] = {
    "embedding": {"description": "Time-delay embedding and parameter selection.", "order": 1},
    "simplex_projection": {"description": "Simplex projection, leave-one-out, and k-nearest neighbors.", "order": 2},
    "smap": {"description": "S-Map local linear prediction.", "order": 3},
    "ccm": {"description": "Convergent Cross Mapping for causal inference.", "order": 4},
    "metrics": {"description": "Prediction evaluation metrics.", "order": 5},
    "splits": {"description": "Time-series cross-validation strategies.", "order": 6},
    "generate": {"description": "Synthetic chaotic time series generators.", "order": 7},
    "util": {"description": "Utility functions for distance computation, padding, and more.", "order": 8},
}

CONFIG: ConfigDict = {
    **default_config,
    "docstring_style": "numpy",
    "heading_level": 2,
    "show_root_heading": True,
    "show_root_full_path": False,
    "show_root_members_full_path": False,
    "show_object_full_path": False,
    "show_if_no_docstring": False,
    "show_signature_annotations": True,
    "separate_signature": True,
    "members_order": "source",
    "docstring_section_style": "table",
}


def _resolve_member(module: Object, name: str) -> Object:
    """Resolve a member by name, following aliases and looking into submodules."""
    member = module.members.get(name)
    if member is None:
        raise KeyError(f"{name} not found in {module.path}")
    if member.is_alias:
        return member.final_target
    # When a submodule shadows the imported function (same name),
    # look for the function inside the submodule.
    if member.kind.value == "module" and name in member.members:
        return _resolve_member(member, name)
    return member


def _render_package_exports(module: Object) -> str:
    """Render __all__ exports of a package as direct members."""
    all_attr = module.members.get("__all__")
    if all_attr is None:
        return render_object_docs(module, CONFIG)

    names = [str(e).strip("'\"") for e in all_attr.value.elements]
    objects = [(name, _resolve_member(module, name)) for name in names]

    # Overview table
    rows = []
    for name, obj in objects:
        summary = ""
        if obj.docstring:
            summary = obj.docstring.parsed[0].value.split("\n")[0]
        rows.append(f"[`{name}`](#{name}) | {summary}")
    overview = "**Functions:**\n\nName | Description\n---- | -----------\n" + "\n".join(rows)

    # Individual function docs
    parts = [overview]
    for _name, obj in objects:
        parts.append(render_object_docs(obj, CONFIG))
    return "\n\n".join(parts)


def main() -> None:
    loader = GriffeLoader(
        search_paths=CONFIG.get("search_paths", []) + sys.path,
        docstring_parser=Parser("numpy"),
        docstring_options=CONFIG.get("docstring_options", {}),
    )
    package = loader.load("edmkit")
    loader.resolve_aliases(external=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for module_name, meta in MODULES.items():
        module = package.modules[module_name]

        if module.modules:
            # Package with submodules: collect exported functions from
            # submodules (referenced in __all__) and render them as if
            # they belong directly to the parent package.
            body = _render_package_exports(module)
        else:
            body = render_object_docs(module, CONFIG)

        filename = module_name.replace("_", "-") + ".md"
        frontmatter = f"""---
title: {module_name}
description: {meta["description"]}
sidebar:
  order: {meta["order"]}
---

"""
        (OUTPUT_DIR / filename).write_text(frontmatter + body)
        print(f"  {filename}")

    print(f"Generated {len(MODULES)} files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
