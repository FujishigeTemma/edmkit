"""Generate API reference Markdown files from docstrings using griffe2md."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

from griffe import Alias, Attribute, ExprList, GriffeLoader, Module, Object, Parser
from griffe import TypeAlias as GriffeTypeAlias
from griffe2md import ConfigDict, default_config, render_object_docs

OUTPUT_DIR = Path(__file__).parent / "src" / "content" / "docs" / "reference"


class ModuleMeta(NamedTuple):
    description: str
    order: int


MODULES: dict[str, ModuleMeta] = {
    "embedding": ModuleMeta("Time-delay embedding and parameter selection.", 1),
    "simplex_projection": ModuleMeta("Simplex projection, leave-one-out, and k-nearest neighbors.", 2),
    "smap": ModuleMeta("S-Map local linear prediction.", 3),
    "ccm": ModuleMeta("Convergent Cross Mapping for causal inference.", 4),
    "metrics": ModuleMeta("Prediction evaluation metrics.", 5),
    "splits": ModuleMeta("Time-series cross-validation strategies.", 6),
    "generate": ModuleMeta("Synthetic chaotic time series generators.", 7),
    "util": ModuleMeta("Utility functions for distance computation, padding, and more.", 8),
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


def resolve_member(module: Object, name: str) -> Object:
    """Resolve a member by name, following aliases and looking into submodules."""
    member = module.members.get(name)
    if member is None:
        raise KeyError(f"{name} not found in {module.path}")
    if isinstance(member, Alias):
        return member.final_target
    # When a submodule shadows the imported function (same name),
    # look for the function inside the submodule.
    if isinstance(member, Module) and name in member.members:
        return resolve_member(member, name)
    return member


def export_names(module: Object) -> list[str] | None:
    """Return the names listed in ``__all__``, or None if not declared."""
    all_attr = module.members.get("__all__")
    if all_attr is None:
        return None
    if not isinstance(all_attr, Attribute):
        raise TypeError(f"Expected __all__ to be Attribute, got {type(all_attr).__name__}")
    if not isinstance(all_attr.value, ExprList):
        raise TypeError(f"Expected __all__ value to be ExprList, got {type(all_attr.value).__name__}")
    return [str(element).strip("'\"") for element in all_attr.value.elements]


def docstring_summary(obj: Object) -> str:
    """First line of the parsed docstring, or empty string when absent."""
    if obj.docstring is None:
        return ""
    return obj.docstring.parsed[0].value.split("\n")[0]


def render_overview_table(items: list[tuple[str, Object]]) -> str:
    """Render a Markdown table listing exported objects."""
    rows = [f"[`{name}`](#{name}) | {docstring_summary(obj)}" for name, obj in items]
    return "**Functions:**\n\nName | Description\n---- | -----------\n" + "\n".join(rows)


def render_package_exports(module: Object) -> str:
    """Render ``__all__`` exports of a package as direct members."""
    names = export_names(module)
    if names is None:
        return render_object_docs(module, CONFIG)

    items = [(name, resolve_member(module, name)) for name in names]
    sections = [render_overview_table(items), *(render_object_docs(obj, CONFIG) for _, obj in items)]
    return "\n\n".join(sections)


def render_type_aliases(module: Object) -> str:
    """Render PEP 695 ``type`` statements, which griffe2md does not emit natively."""
    show_undocumented = bool(CONFIG.get("show_if_no_docstring", False))
    aliases = [
        member for member in module.members.values() if isinstance(member, GriffeTypeAlias) and (show_undocumented or member.docstring is not None)
    ]
    if not aliases:
        return ""

    rows = [f"[`{alias.name}`](#{alias.path}) | {docstring_summary(alias)}" for alias in aliases]
    table = "**Type Aliases:**\n\nName | Description\n---- | -----------\n" + "\n".join(rows)

    details = []
    for alias in aliases:
        body = alias.docstring.value if alias.docstring else ""
        details.append(f"### `{alias.name}` {{#{alias.path}}}\n\n```python\ntype {alias.name} = {alias.value}\n```\n\n{body}")
    return "\n\n".join([table, *details])


def render_module(module: Object) -> str:
    """Render a module, inserting a Type Aliases section where griffe2md would place Attributes."""
    if isinstance(module, Module) and module.modules and export_names(module) is not None:
        base = render_package_exports(module)
    else:
        base = render_object_docs(module, CONFIG)
    aliases_section = render_type_aliases(module)
    if not aliases_section:
        return base
    # griffe2md historically placed attribute listings between the module
    # heading/docstring and the first member (Functions table or `### ` heading).
    for marker in ("**Functions:**", "### "):
        index = base.find(marker)
        if index != -1:
            return base[:index] + aliases_section + "\n\n" + base[index:]
    return f"{base}\n\n{aliases_section}\n"


def write_reference(module_name: str, meta: ModuleMeta, body: str) -> Path:
    """Write a single reference file with Starlight frontmatter."""
    frontmatter = f"""---
title: {module_name}
description: {meta.description}
sidebar:
  order: {meta.order}
---

"""
    path = OUTPUT_DIR / (module_name.replace("_", "-") + ".md")
    path.write_text(frontmatter + body)
    return path


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
        path = write_reference(module_name, meta, render_module(module))
        print(f"  {path.name}")

    print(f"Generated {len(MODULES)} files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
