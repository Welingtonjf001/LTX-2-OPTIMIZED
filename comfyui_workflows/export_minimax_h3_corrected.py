"""Export the supplied prompt into the correct API graph and official UI template."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minimax_h3_test_ui import build_workflow, COMFY


def export(report_path):
    report = Path(report_path).read_text(encoding="utf-8")
    old = json.loads(next(line for line in report.splitlines() if line.startswith('{"id"')))
    prompt = next(node for node in old["nodes"] if node["id"] == 67)["widgets_values"][0]
    output = Path(__file__).resolve().parent
    (output / "minimax_h3_prompt_original_corrigido_api.json").write_text(
        json.dumps(build_workflow(prompt), ensure_ascii=False, indent=2), encoding="utf-8")
    workflow = json.loads((COMFY.parent / "workflows" / "video_minimax_h3_r2v.json").read_text(encoding="utf-8"))
    # No reference images were included in the error report. Remove the template placeholders.
    removed = {137, 139}
    removed_links = {link[0] for link in workflow["links"] if link[1] in removed or link[3] in removed}
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] not in removed]
    workflow["links"] = [link for link in workflow["links"] if link[0] not in removed_links]
    for node in workflow["nodes"]:
        for inp in node.get("inputs", []):
            if inp.get("link") in removed_links:
                inp["link"] = None
        values = node.get("widgets_values", [])
        if node["id"] == 138:
            values[0] = prompt
        elif node["id"] == 128:
            values[:] = [values[0], "minimax", "default"]
        elif node["id"] == 132:
            values[0] = 5
        elif node["id"] == 146:
            values[0] = True
        node.pop("widgets_values_named", None)
    (output / "minimax_h3_corrigido.json").write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    export(sys.argv[1])
