"""Convert a ComfyUI *UI-format* workflow into *API-format*, prune it to the
subgraph feeding a chosen output node, and optionally swap the diffusion-model
loader for the GGUF loader.

The UI format (what the editor saves) stores nodes with positional
``widgets_values`` plus link ids. The API format (what POST /prompt accepts)
needs ``{node_id: {"class_type": ..., "inputs": {name: value | [src_id, slot]}}}``.
Mapping widgets to input names requires each node's schema, which we pull from
the running ComfyUI at /object_info.

Usage:
  python comfy_workflow_tool.py --in workflow.json --out api.json \
      [--server http://127.0.0.1:8188] [--keep-output NODE_ID] [--list-outputs]
"""
import argparse
import json
import sys
import urllib.request

# Node "mode" values in the UI format.
MODE_MUTED = 2
MODE_BYPASSED = 4


def fetch_object_info(server: str) -> dict:
    with urllib.request.urlopen(f"{server.rstrip('/')}/object_info", timeout=120) as r:
        return json.load(r)


def widget_input_names(schema: dict) -> list[str]:
    """Input names that are widgets (literal values), in declaration order.

    Link-only inputs (MODEL/CLIP/VAE/LATENT/...) never consume a widgets_values
    slot; combos and primitives do. ComfyUI declares a type as a list/COMBO for
    dropdowns and a string like "INT"/"FLOAT"/"STRING"/"BOOLEAN" for primitives
    -- some newer (V3-style) node schemas instead declare a comma-joined union
    like "FLOAT,INT" for a widget that accepts either. Missing this case drops
    the name from this list entirely, which shifts the positional zip against
    widgets_values for every widget declared AFTER it (silent data corruption,
    not a crash -- confirmed on LTXVEmptyLatentAudio's "frame_rate": "FLOAT,INT"
    was skipped, so "batch_size" picked up frame_rate's value instead of its
    own). Match on any comma-separated part being a known primitive, not exact
    equality.
    """
    names = []
    for section in ("required", "optional"):
        for name, spec in (schema.get("input", {}).get(section) or {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            t = spec[0]
            is_combo = isinstance(t, list) or t == "COMBO"
            is_primitive = isinstance(t, str) and any(
                part in {"INT", "FLOAT", "STRING", "BOOLEAN"} for part in t.split(",")
            )
            if is_combo or is_primitive:
                names.append(name)
    return names


def _node_inputs(node: dict, schema: dict) -> dict:
    """Widget values (positional) + declared input slots for one node, BEFORE
    resolving links. Widget values may be overridden by a link at the same name."""
    inputs: dict = {}
    wvals = node.get("widgets_values")
    if isinstance(wvals, dict):
        inputs.update(wvals)
    elif isinstance(wvals, list) and wvals:
        for name, val in zip(widget_input_names(schema), wvals):
            inputs[name] = val
    return inputs


def convert(ui_wf: dict, object_info: dict) -> dict:
    """Flatten a UI-format workflow, INCLUDING subgraph instances, into API format.

    ComfyUI subgraphs store their internal node graph under
    ``definitions.subgraphs`` (keyed by a UUID that appears as the instance
    node's ``type`` in the outer graph). Internal links use dict form
    ``{id, origin_id, origin_slot, target_id, target_slot}`` with two
    sentinel node ids: -10 (the subgraph's own input boundary) and -20 (its
    output boundary). We resolve boundary crossings lazily via ``resolve()``,
    which walks up through nested subgraph instances as needed and memoizes
    results in ``resolved`` (keyed by (node_id, slot) in a global, prefixed
    id-space so nodes from different subgraph instances never collide).
    """
    subgraphs_by_id = {sg["id"]: sg for sg in (ui_wf.get("definitions", {}) or {}).get("subgraphs", []) or []}

    api: dict[str, dict] = {}
    skipped: list[tuple] = []
    # (prefixed_node_id, slot) -> [src_prefixed_node_id, src_slot] | literal value
    resolved: dict[tuple[str, int], object] = {}

    def outer_link_table(links: list) -> dict[int, tuple[str, int]]:
        table = {}
        for link in links or []:
            if isinstance(link, list) and len(link) >= 3:
                table[link[0]] = (str(link[1]), link[2])
        return table

    def inner_link_table(links: list) -> dict[int, dict]:
        return {l["id"]: l for l in (links or []) if isinstance(l, dict)}

    def process_graph(nodes: list, links_table: dict[int, tuple[str, int]], prefix: str) -> None:
        """Expand one graph level (top-level or one subgraph's internals).
        ``links_table``: outer-link-id -> (prefixed_src_node_id, src_slot), already
        resolved to THIS level's id-space (top-level ids or f"{prefix}{id}")."""
        for node in nodes:
            mode = node.get("mode", 0)
            nid = f"{prefix}{node['id']}"
            if mode in (MODE_MUTED, MODE_BYPASSED):
                skipped.append((nid, node.get("type"), mode))
                continue
            ctype = node.get("type")

            if ctype in subgraphs_by_id:
                sub = subgraphs_by_id[ctype]
                sub_prefix = f"{nid}:"
                inner_links = inner_link_table(sub.get("links"))

                def resolve_boundary_input(slot: int, node=node, links_table=links_table, sub=sub):
                    """Value/link feeding the subgraph instance's own input `slot`.
                    Matched by NAME against the subgraph's declared boundary input at
                    that slot -- the instance node's own `inputs` list only exposes a
                    subset (unconnected/never-wired boundary inputs are omitted, not
                    positionally aligned), so index-matching would misalign or crash.
                    Returns None when unresolvable, meaning: leave the internal node's
                    own default (from its own widgets_values) untouched."""
                    decl = (sub.get("inputs") or [])
                    if slot >= len(decl):
                        return None
                    name = decl[slot].get("name")
                    matches = [i for i in (node.get("inputs") or []) if i.get("name") == name]
                    if not matches:
                        return None
                    inp = matches[0]
                    lid = inp.get("link")
                    if lid is not None:
                        return list(links_table[lid]) if lid in links_table else None
                    wvals = node.get("widgets_values")
                    if isinstance(wvals, dict):
                        return wvals.get(name)
                    if isinstance(wvals, list):
                        widget_inputs = [i for i in (node.get("inputs") or []) if i.get("link") is None]
                        if inp in widget_inputs:
                            idx = widget_inputs.index(inp)
                            if idx < len(wvals):
                                return wvals[idx]
                    return None

                # Build this subgraph's own link table: internal link id -> resolved source,
                # where a source landing on the -10 sentinel resolves through the boundary.
                sub_links_table: dict[int, tuple[str, int] | object] = {}
                for lid, link in inner_links.items():
                    if link["origin_id"] == -10:
                        val = resolve_boundary_input(link["origin_slot"])
                        if val is None:
                            continue  # unresolvable -- leave the internal node's own default alone
                        sub_links_table[lid] = val if isinstance(val, list) else ("__literal__", val)
                    else:
                        sub_links_table[lid] = (f"{sub_prefix}{link['origin_id']}", link["origin_slot"])

                # Register this instance's OUTPUT slots -> their real internal producer,
                # so other nodes referencing (nid, slot) resolve correctly.
                for lid, link in inner_links.items():
                    if link["target_id"] == -20:
                        src = sub_links_table.get(lid) if link["origin_id"] == -10 else (
                            f"{sub_prefix}{link['origin_id']}", link["origin_slot"]
                        )
                        if src is None:
                            continue
                        resolved[(nid, link["target_slot"])] = list(src) if isinstance(src, tuple) and src[0] != "__literal__" else src

                # Turn sub_links_table (int link id -> ...) into a per-target-node table
                # analogous to top-level `links_table`, keyed the same way `link.get("link")`
                # ids are used inside the subgraph's own nodes.
                process_graph(sub.get("nodes") or [], _as_link_src(sub_links_table), sub_prefix)
                continue

            if ctype == "Reroute" and object_info.get(ctype) is None:
                # Frontend-only passthrough in newer ComfyUI (no backend schema): make it
                # transparent so downstream consumers resolve straight to its own source.
                lid = ((node.get("inputs") or [{}])[0]).get("link")
                src = links_table.get(lid) if lid is not None else None
                if src is not None:
                    resolved[(nid, 0)] = list(src) if src[0] != "__literal__" else src
                continue

            schema = object_info.get(ctype)
            if schema is None:
                skipped.append((nid, ctype, "no-schema"))
                continue

            inputs = _node_inputs(node, schema)
            for inp in node.get("inputs", []) or []:
                lid = inp.get("link")
                if lid is None:
                    continue
                src = links_table.get(lid)
                if src is None:
                    continue
                inputs[inp["name"]] = list(src) if src[0] != "__literal__" else src[1]

            api[nid] = {"class_type": ctype, "inputs": inputs}

    def _as_link_src(table: dict) -> dict:
        return table

    process_graph(ui_wf.get("nodes", []), outer_link_table(ui_wf.get("links")), "")

    # Second pass: any input still pointing at a subgraph-instance's output slot
    # (rather than a real node) gets redirected via `resolved`, transitively.
    def deref(src: list, depth: int = 0) -> list | tuple:
        if depth > 20 or not isinstance(src, list):
            return src
        key = (src[0], src[1])
        if key in resolved:
            nxt = resolved[key]
            if isinstance(nxt, tuple) and nxt and nxt[0] == "__literal__":
                return nxt
            return deref(list(nxt), depth + 1)
        return src

    for node in api.values():
        for name, val in list(node["inputs"].items()):
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                fixed = deref(val)
                if isinstance(fixed, tuple) and fixed and fixed[0] == "__literal__":
                    node["inputs"][name] = fixed[1]
                else:
                    node["inputs"][name] = fixed

    if skipped:
        print(f"[convert] skipped {len(skipped)} node(s): {skipped[:8]}", file=sys.stderr)
    return api


def prune_to_output(api: dict, keep_id: str) -> dict:
    """Keep only nodes the chosen output node transitively depends on."""
    keep: set[str] = set()
    stack = [str(keep_id)]
    while stack:
        nid = stack.pop()
        if nid in keep or nid not in api:
            continue
        keep.add(nid)
        for v in api[nid]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return {k: v for k, v in api.items() if k in keep}


def list_outputs(api: dict) -> list[tuple[str, str, str]]:
    out = []
    for nid, n in api.items():
        if n["class_type"] in ("SaveVideo", "SaveImage", "SaveAudio", "VHS_VideoCombine"):
            label = ""
            for k in ("filename_prefix", "filename"):
                if isinstance(n["inputs"].get(k), str):
                    label = n["inputs"][k]
                    break
            out.append((nid, n["class_type"], label))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--keep-output", help="node id of the Save* node to keep")
    ap.add_argument("--list-outputs", action="store_true")
    args = ap.parse_args()

    ui = json.load(open(args.inp, encoding="utf-8"))
    oi = fetch_object_info(args.server)
    api = convert(ui, oi)

    if args.list_outputs:
        for nid, ctype, label in list_outputs(api):
            print(f"output node id={nid} type={ctype} prefix={label!r}")
        return 0

    if args.keep_output:
        before = len(api)
        api = prune_to_output(api, args.keep_output)
        print(f"[prune] {before} -> {len(api)} nodes (kept subgraph of {args.keep_output})")

    if args.out:
        json.dump(api, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"[write] {args.out} ({len(api)} nodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
