# MiniMax H3: correção do workflow e teste Gradio

O relatório de 06/09/2026 aponta `audio_scale` ausente no KSampler (nó 70).
O workflow era de Z-Image, com pesos MiniMax H3 selecionados nos loaders.
O `ModelSamplingAuraFlow` (69) troca o sampler FLOW_AV do H3 por um sampler
de imagem que não fornece o contrato de áudio exigido pelo modelo.

A correção é usar o grafo próprio do H3:

- CLIPLoader com tipo `minimax` e encoder Qwen3-VL do H3.
- Sampler nativo do modelo, sem `ModelSamplingAuraFlow`.
- `MiniMaxH3ReferenceToVideo`, que cria latent conjunto de vídeo e áudio.
- VAE de vídeo no `VAEDecode` e VAE de áudio no `VAEDecodeAudio`.
- `CreateVideo` a 24 fps, recebendo ambos, seguido por `SaveVideo`.
- LoRA turbo junto com quatro passos, ou modelo base com vinte passos.

Não foi inserido um `audio_scale` artificial no código do ComfyUI: isso
ocultaria a primeira falha, mantendo o restante do grafo incompatível.

O teste real também revelou `float != c10::Half` no VAE de vídeo: com
`--cpu-vae` e carregamento dinâmico, os dados FP32 chegavam a pesos FP16.
Foi removido `--cpu-vae` de `MiniMax-H3/start_minimax_h3.ps1`, permitindo
decodificação CUDA no dtype do VAE. O modo `--lowvram` foi mantido.

## Executar

1. Mantenha o ComfyUI MiniMax H3 em `http://127.0.0.1:8189`.
   Se necessário, execute `E:\Users\home\Documents\MiniMax-H3\start_minimax_h3.bat`.
2. Execute `start_minimax_h3_test.bat` nesta pasta.
3. Abra `http://127.0.0.1:7916` e clique em **Verificar conexão, nós e modelos**.
4. Digite o prompt; até duas imagens de referência são opcionais.
5. Clique em **Gerar vídeo com áudio**. O resultado, workflow API e log são
   salvos numa pasta exclusiva em `outputs/minimax_h3_test/`.

O teste padrão usa 640 × 384, cinco segundos solicitados, seed 42 e turbo.
O modelo ajusta a duração à grade de 5 + 17k frames: cinco segundos viram
124 frames (aproximadamente 5,17 s). O ComfyUI gerencia o encoder automaticamente.
Há uma opção para forçar CPU e economizar VRAM, com maior tempo de processamento.
A UI usa o backend já iniciado e não encerra outros servidores.

O prompt original menciona 40 segundos, mas texto no prompt não configura
a duração. Os workflows corrigidos preservam esse texto e configuram um
teste de cinco segundos. A UI limita pedidos a 15 s; a implementação local
documenta a faixa treinada de aproximadamente 124–362 frames.

## Workflows

- `comfyui_workflows/minimax_h3_corrigido.json`: template visual H3 corrigido,
  com o prompt original, para abrir no ComfyUI.
- `comfyui_workflows/minimax_h3_prompt_original_corrigido_api.json`: grafo
  API com o prompt original e duração de teste.
- `comfyui_workflows/minimax_h3_corrigido_api.json`: grafo API com prompt curto.

O workflow visual deriva do template H3 existente na instalação. Os arquivos
originais do usuário e os módulos do ComfyUI foram preservados.

## Verificação

` .venv\Scripts\python.exe -m unittest tests.test_minimax_h3_test_ui -v `

Verifica contrato AV, ligação de referências/turbo, grade de frames e
rejeição de configurações inválidas. O botão de diagnóstico consulta os
nós e modelos da instalação real antes de enviar geração.
