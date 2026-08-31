# DAZ Character Director

Integra uma descrição em linguagem natural com Genesis no DAZ Studio usando um LLM local.

## Instalação

```powershell
python -m pip install -r daz_integration/requirements.txt
```

Inicie Ollama e baixe um modelo de instrução, por exemplo `ollama pull qwen2.5:7b`. Depois execute:

```powershell
python -m daz_integration.app
```

Abra `http://127.0.0.1:7865`. A interface gera uma especificação JSON e um script `.dsa`. Marque “Abrir DAZ e executar script” para iniciar o DAZ Studio.

## Referência e assets

A imagem de referência fica registrada no job para futura etapa de Face Transfer/seleção de presets. O protótipo não cria uma malha nova a partir de pixels: ele compõe uma figura Genesis e deixa cabelos, roupas e materiais preparados para serem ligados aos presets instalados na biblioteca DAZ.

## MCP opcional

Instale `mcp` e execute `python -m daz_integration.mcp_server` usando transporte stdio. A tool `generate_character` retorna a especificação e o caminho do `.dsa` para o agente.

## Fluxo recomendado para LTX

Depois da montagem, faça uma exportação única FBX/GLTF para o Blender. BVH, câmera, passes de depth/normais e renderizações devem continuar no pipeline Blender/LTX, evitando reabrir o DAZ para cada quadro.

