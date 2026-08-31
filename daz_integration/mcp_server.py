from mcp.server.fastmcp import FastMCP
from .character_schema import CharacterSpec
from .daz_controller import create_script, run_daz
from .local_llm import interpret

mcp = FastMCP("daz-character-director")

@mcp.tool()
def generate_character(description: str, output_dir: str, daz_path: str = "") -> str:
    """Interpreta uma descrição com LLM local, gera um script DSA e opcionalmente abre o DAZ."""
    spec = interpret(description)
    script = create_script(spec, output_dir)
    result = {"spec": spec.__dict__, "script": str(script)}
    if daz_path: result["status"] = run_daz(script, daz_path)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)

if __name__ == "__main__": mcp.run()

