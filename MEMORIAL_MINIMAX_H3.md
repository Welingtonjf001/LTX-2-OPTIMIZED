# Memorial — MiniMax H3 local na RTX 3090

**Estado:** plano de avaliação; nenhuma instalação, download ou mudança de
workflow H3 foi aplicada ainda.

**Objetivo:** avaliar e, se aprovado, executar o MiniMax H3 Base localmente
para vídeos curtos, preservando por completo a rota de produção LTX 2.3/2.5.

Este arquivo é um handoff para Claude/Codex. Ele descreve contexto e limites;
não autoriza alterações por si só. Antes de baixar pesos, instalar pacotes,
atualizar ComfyUI ou apagar arquivos, confirmar com o usuário.

---

## 1. O que o usuário quer

O usuário viu o vídeo local:

`C:\Users\user\Downloads\INSANO Otimizei o MiniMax H3 ao MÁXIMO - mais rápido em PC fraco e mais novidades! - Willian IA (480p, h264, youtube).mp4`

e quer reproduzir a rota otimizada para o **MiniMax H3 Base**. A intenção não
é reproduzir o H3 Max da fal nem fazer fine-tuning neste momento.

O H3 Max da fal é um pós-treino proprietário servido por API. Seus pesos e seu
motor de inferência não estão disponíveis para execução local. A instalação
local discutida aqui é o MiniMax H3 Base, de pesos abertos.

## 2. Máquina e restrições operacionais

| recurso | estado relevante |
|---|---|
| GPU de geração | RTX 3090, 24,5 GB (no `nvidia-smi`, GPU 0) |
| outra GPU | RTX 4070, 12,9 GB |
| RAM | 80 GB |
| ambiente LTX | `E:\Users\home\Documents\LTX-2-OPTIMIZED\.venv` |
| Python LTX | Torch 2.8.0 + CUDA 12.8 |
| ComfyUI existente | `ComfyUI\`, com patches e fluxos LTX 2.5 em produção |

Nesta máquina a convenção já usada pelos lançadores é
`CUDA_VISIBLE_DEVICES=1` para reservar a **3090** ao subprocesso. Não deduzir
o índice pelo `nvidia-smi`: a enumeração do Torch é diferente. Após filtrar
com `CUDA_VISIBLE_DEVICES=1`, a GPU visível dentro do processo normalmente
passa a ser `cuda:0`.

**Não iniciar LTX 2.5 e H3 ao mesmo tempo.** Ambos competem por VRAM, RAM,
cache e ComfyUI. Encerrar o outro servidor e verificar com `nvidia-smi` antes
de um benchmark.

## 3. O que o vídeo demonstra

Análise dos quadros do vídeo indica esta receita de otimização, não treino:

1. Workflow ComfyUI voltado a H3 (o vídeo chama de “MiniMax H3 Easy”).
2. Transformer H3 compacto/quantizado **W4A8** para FL2VA ou Ref2VA.
3. Encoder de texto/visão **Qwen3-VL 4B FP8** em vez do encoder grande.
4. VAE de vídeo **INT8 ConvRot**; o vídeo também menciona TAE como opção de
   decode mais rápido.
5. **Turbo LoRA** para reduzir passos.
6. **SageAttention**, se a versão instalada for compatível com Torch/CUDA.
7. `BasicScheduler`: scheduler `simple`, `steps=8`, `denoise=1.0`.

Essas mudanças negociam qualidade e fidelidade por VRAM e velocidade. Não
habilitar todas de uma vez: medir cada mudança com o mesmo prompt e seed.

## 4. Situação do ComfyUI deste projeto

O ComfyUI já contém suporte de código nativo ao MiniMax H3:

- `ComfyUI/comfy_extras/nodes_minimax_h3.py` — T2VA, FL2VA, Ref2VA, guias;
- `ComfyUI/comfy/ldm/minimax/` — transformer, VAE de vídeo e VAE de áudio;
- `ComfyUI/main.py --help` expõe `--use-sage-attention`, `--cache-none`,
  `--fast-disk`, `--reserve-vram`, `--lowvram` e opções de VRAM dinâmica.

Isso **não** significa que os pesos H3 já estejam instalados.

Não executar `git pull`, reinstalar requisitos, nem copiar nós de terceiros
sobre `ComfyUI/`: a instalação atual atende LTX 2.5 e tem alterações locais.
O caminho seguro é uma cópia/check-out isolado, por exemplo `ComfyUI-H3/`, em
outra porta. Só criar essa cópia depois de autorização explícita do usuário.

## 5. Roteiro de implementação, sujeito a aprovação

### Fase A — inventário e segurança

1. Verificar espaço livre em NVMe e RAM livre; os arquivos de H3 são grandes.
2. Registrar versões de Python, Torch, CUDA e driver.
3. Fazer snapshot da configuração H3 separada; não tocar em `ComfyUI/` LTX.
4. Definir um diretório de modelos independente ou links somente de leitura.
5. Confirmar a licença do MiniMax H3 para o uso pretendido antes de baixar.

### Fase B — instalação mínima

Baixar somente os componentes requisitados pelo workflow escolhido. Para o
primeiro teste, preferir **FL2VA** (texto + primeiro/último frame), não Ref2VA.
Usar pesos W4A8/INT8 e encoder 4B FP8 que sejam explicitamente compatíveis
com a versão do workflow. Conferir origem, hash e licença; não aceitar pesos
arbitrários de repositórios comunitários sem verificação.

Instalar SageAttention apenas após comparar sua wheel com as versões reais de
Python/Torch/CUDA. Caso falhe, deixar o ComfyUI usar PyTorch attention: é mais
lento, mas é uma linha de base válida.

### Fase C — primeiro benchmark

Subir o H3 em porta isolada, local apenas. Exemplo a adaptar à cópia separada:

```powershell
cd E:\Users\home\Documents\LTX-2-OPTIMIZED
$env:CUDA_VISIBLE_DEVICES = "1"
.\.venv\Scripts\python.exe .\ComfyUI-H3\main.py `
  --listen 127.0.0.1 `
  --port 8189 `
  --cache-none `
  --fast-disk `
  --reserve-vram 1
```

Adicionar `--use-sage-attention` somente depois de validar a dependência.
Deixar a VRAM dinâmica no padrão no primeiro teste; não copiar cegamente flags
que foram escolhidas para LTX 2.5, pois o perfil de memória do H3 é outro.

Parâmetros da primeira rodada:

- batch 1;
- uma saída curta (aprox. 4–5 s, na grade de frames do H3);
- resolução moderada;
- scheduler `simple`, 8 passos, se e somente se a Turbo LoRA correspondente
  estiver ativa;
- mesmo prompt, seed e referência para toda comparação.

Coletar: tempo de carga, tempo de geração, pico VRAM, RAM, saída, áudio e
aderência ao prompt. Salvar os resultados num diretório de benchmark separado.

### Fase D — ablação

Mudar uma variável por execução nesta ordem:

1. baseline sem Turbo LoRA/SageAttention/TAE;
2. Turbo LoRA e passos correspondentes;
3. SageAttention;
4. VAE INT8 ou TAE;
5. duração e resolução.

Não concluir “mais rápido” só pelo tempo de amostragem: incluir carregamento,
decode, áudio e qualquer offload de RAM/disco.

## 6. Limites conhecidos

- Uma 3090 pode testar variantes quantizadas/offload, mas não é a referência
  oficial de inferência multi-GPU do H3 Base.
- A qualidade/latência do H3 Max da fal não é meta reproduzível localmente.
- A variante 2K completa usa módulos hospedados; H3 Base local é a etapa de
  geração de base.
- Quantizar encoder/transformer, usar VAE rápido ou reduzir passos pode piorar
  aderência, detalhes, áudio e consistência temporal.
- `--cache-none` reduz pressão de memória, mas pode aumentar o tempo total.

## 7. Critério de decisão

Prosseguir com integração de produto apenas se o H3 local entregar, num conjunto
fixo de prompts de música/cinema:

1. não causar OOM ou swap prolongado;
2. qualidade superior ou complementar ao LTX 2.5 para um caso concreto;
3. tempo aceitável por clipe;
4. nenhum impacto na rota LTX existente;
5. licença compatível com o destino do conteúdo.

Caso contrário, usar H3/H3 Max por API somente para os casos em que a qualidade
extra justifique custo e envio de dados.

## 8. Fontes primárias

- MiniMax H3 Base e instruções de deploy: https://huggingface.co/MiniMaxAI/MiniMax-H3
- Código oficial do MiniMax H3: https://github.com/MiniMax-AI/MiniMax-H3
- H3 Max da fal (API/pós-treino proprietário): https://blog.fal.ai/introducing-h3-max-by-fal/
