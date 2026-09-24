# Continuidade de produção — Voo 702

Atualizado em 2026-09-20 após o teste de animatic integrado à WebUI.

## Contrato aplicado

O módulo `continuity_audit.py` é genérico para qualquer roteiro: locação,
inventário declarado e elenco secundário são avaliados por cena. A regra de
aeronave em voo é um contrato opcional ativado apenas quando a continuidade da
produção fornece `aircraft_contract` (como neste projeto).

Cada plano recebe um contrato persistente antes de gerar still ou vídeo:

- `location_id` único por cena, com geometria fixa da cabine, cockpit ou exterior;
- a mesma aeronave branca bimotora, com asas enflechadas e cauda azul, sempre em voo;
- objetos já apresentados permanecem no inventário da cena: painel/radar, corredor 3-3,
  bagageiros, carrinhos, galleys, jumpseats, cintos, sinais de cinto, faixas de emergência,
  bagagem da fileira 19 e passageiros;
- o elenco nomeado da cena é carregado como `scene_characters`; personagens secundários
  não podem desaparecer do contrato só porque o plano atual é um close-up;
- closes podem cortar um objeto do quadro, mas não podem removê-lo do estado da cena,
  teleportá-lo ou trocar sua posição sem uma ação explícita no roteiro.

`script_pipeline/continuity_audit.py` grava `shots/continuity_audit.json` e bloqueia a
passada de vídeo quando encontra locação divergente, aeronave sem estado de voo,
contrato de personagem ausente ou elenco secundário sem cobertura. Avisos de objeto
também ficam registrados para revisão.

## Referências de identidade

As fotos fornecidas pelo usuário foram importadas para `characters/refs/` e vinculadas
por nome no `cast.json`:

| Nome do roteiro | Arquivo | Fonte |
|---|---|---|
| `JI-HO` | `JI_HO_external.png` | foto externa |
| `SEO-YEON` | `SEO_YEON_external.png` | foto externa |
| `HA-EUN` | `HA_EUN_external.png` | foto externa |
| `MIN-JUN` | `MIN_JUN_external.png` | foto externa |
| `PILOTO` | `PILOTO_anchor.png` | anchor gerado previamente |
| `COPILOTO` | `COPILOTO_anchor.png` | anchor gerado previamente |

O importador usa `cv2.imdecode(np.fromfile(...))`, pois `cv2.imread` falha em caminhos
Windows que contêm acentos. A character sheet agora detecta que todas as seis
identidades já têm referência, grava `sheet_report.json` e não sobe o ComfyUI para
gerar candidatos redundantes. Esse foi o caminho que eliminou o travamento observado
na rodada anterior.

## Ordem de validação

1. Importar referências e atualizar `cast.json`.
2. Recalcular `shot_plan.json` e `editorial/timeline.json` com `conform_plan`.
3. Executar `character_sheet.py`; a saída esperada é “todo o elenco já tem reference_image”.
4. Gerar stills com auditoria facial e reamostragem por seed quando o score cair.
5. Executar a auditoria de continuidade estrutural dos prompts.
6. Executar o gate visual em duas passadas com `qwen3-vl:30b`: primeiro percepção
   neutra dos pixels; depois comparação da percepção com o contrato.
7. Só liberar a geração de clipes LTX quando todos os stills forem aprovados.
8. Extrair primeiro, meio e último quadro de cada clipe e repetir o gate visual.
9. Só liberar lipsync, mixagem e montagem quando todos os clipes forem aprovados.

### Smoke test da integração espacial

O caminho validado para a WebUI é:

1. gerar o spec com `python -m script_pipeline.voo702_spatial --run <RUN> --output <RUN>/world/voo702_spatial_spec.json`;
2. marcar **Ativar estado espacial persistente** na seção **Continuidade espacial 3D**;
3. informar o spec, usar `flux` nas imagens e parar em `animatic`;
4. marcar **Animatic diagnóstico** apenas se a intenção for inspecionar um gate bloqueado.

O teste de 20 segundos aprovado usa quatro planos contíguos da cabine, 384×256,
24 fps e três falas. Ele mostra HA-EUN, MIN-JUN, JI-HO e SEO-YEON com as fotos
externas fornecidas. `shots/visual_stills_audit.json` aprovou 4/4; HA-EUN exigiu
duas rodadas seletivas e passou com similaridade facial 0,385. O arquivo final é
`shots/animatic.mp4`; `shots/animatic_audit.json` registra 20,000/20,000 s, zero
still ausente e 3/3 falas sincronizadas. Esse resultado valida a passagem WebUI →
`run_decupagem` → blocking espacial → FLUX/ComfyUI → montagem do animatic; ele não
constitui aprovação do vídeo final de todos os planos.

O plano integral permanece em `parse/shot_plan.full.json`; o cockpit anterior foi
preservado em `parse/shot_plan.cockpit20.json`. O recorte ativo está em
`parse/shot_plan.json` e o spec em `world/voo702_cabin20_spatial_spec.json`.

Comandos de inspeção:

```powershell
python -m script_pipeline.continuity_audit --run-dir <RUN>
python -m script_pipeline.visual_continuity_audit --run-dir <RUN> --stage stills
python -m script_pipeline.visual_continuity_audit --run-dir <RUN> --stage video
python -m script_pipeline.character_sheet --run-dir <RUN> --apply
```

O relatório estrutural verifica se o contrato foi escrito. O gate visual verifica se
o contrato aparece de fato nos pixels. Fatos físicos simples ficam travados pela
percepção neutra: uma segunda chamada não pode aprovar uma aeronave como “em voo” se
a primeira detectou pista ou rodas apoiadas, nem ocultar texto artificial detectado.

O modelo padrão é `qwen3-vl:30b`. O `gemma4:latest` foi descartado porque, nesta
instalação do Ollama, descreveu imagens válidas e inválidas como pessoas e carros em
um estacionamento, produzindo falsos bloqueios. `qwen2.5vl:7b` permanece disponível
apenas como alternativa leve; não é o padrão de produção.

Os relatórios são gravados em `shots/visual_stills_audit.json` e
`shots/visual_video_audit.json`. A validação de integração aprovou o still 11
(HA-EUN na cabine), bloqueou o still 27 (avião na pista) e bloqueou o clipe 16
(SEO-YEON deixa de ser o sujeito principal durante a fala). A suíte focal terminou
com 16 testes aprovados.
