# Pipeline de estado espacial

Implementação opt-in do piloto proposto em `PLANO_ESTADO_CENA_3D.md`.

## Executar

Requer o ambiente `.venv` já usado pelo projeto, Blender no PATH e os modelos locais
FLUX.2 Klein, LTX 2.5 e Ollama. Dependência adicional: `requirements-spatial.txt`
(OpenEXR, para ler os passes multipart produzidos pelo Blender 5).

```powershell
.venv/Scripts/python.exe -m script_pipeline.spatial_pipeline --project outputs/spatial_pilot_20260919 --demo --qwen --denoise 0.95 --stage all
.venv/Scripts/python.exe -m script_pipeline.spatial_ui
```

A interface standalone fica em `http://127.0.0.1:7879`. A `decupagem_ui.py` principal
agora também expõe o accordion **Continuidade espacial 3D**, que envia os mesmos flags
ao `run_decupagem` quando ativado. Etapas individuais: `prepare`, `blocking`,
`stills`, `audit-stills`, `video`, `audit-video`. A etapa `video` exige aprovação
dos stills atuais; uma aprovação antiga não libera imagens modificadas.
`--stage all` para quando um gate reprova. O vídeo reprovado é mantido para diagnóstico.
Para testar a dinâmica depois de um bloqueio, use `--stage video --allow-unapproved`;
isso grava `visual_approval: bypassed_for_diagnostic` e não libera nenhuma etapa posterior.

Para outro roteiro, fornecer `--spec arquivo.json`, seguindo `demo_spec()` em
`spatial_pipeline.py`: locação em caixas simples, entidades, câmeras e planos com
eventos tipados. `--qwen` interpreta ações que já possuem um contrato explícito;
neste piloto compara a proposta à operação esperada e recusa mudanças criativas.
O compilador ainda não cria automaticamente toda a encenação de um roteiro livre.

## Implementado

- SQLite com estados imutáveis, assets versionados, eventos transacionais e bindings
  de plano. O mesmo evento não é aplicado outra vez ao mudar câmera ou retomar.
- Snapshots recuperáveis e separados dos artefatos gerados; transferência de objeto
  valida o dono anterior e o novo destino antes de efetivar a transação.
- Blender procedural para locação/manequins/objetos, câmera Z-up em metros, lente e
  sensor explícitos, attachments de mão, beauty, depth métrico, normais, máscaras
  de instância e pose COCO-18. Verificação numérica contra a projeção real do Blender.
- Cache por conteúdo de estado, câmera, renderizador e artefatos; entradas ausentes
  ou alteradas bloqueiam reutilização.
- Adaptador **RGB img2img + ReferenceLatent para Klein**, com duas estratégias por
  enquadramento. Planos abertos usam o blocking como latente inicial; closes usam o
  ambiente como referência e mantêm a referência nominal por último na cadeia de
  condicionamento. Não é um ControlNet de profundidade.
- Planejamento de câmera por tomada e sujeito: close em 72 mm, extreme close em
  85 mm, medium em 55 mm e OTS em 58 mm. A visibilidade é um estado de render por
  tomada; ocultar uma entidade não a remove do mundo persistente.
- Stills inicial/final, keyframe LTX em índice múltiplo de 8 dentro do corte editorial,
  montagem por contagem de frames e verificação com ffprobe.
- Gate visual em duas chamadas: percepção dos pixels sem contrato, seguida de decisão.
  Vídeo amostrado em cinco momentos por plano; aprovação ligada ao hash da mídia.
- Interface local e ponte `run_decupagem --spatial-spec` para bindings estáticos.
  Planos com mudança de estado usam o runner temporal `spatial_pipeline`.

## Arquivos de saída

`world/world.sqlite`, `world/snapshots/`, `world/spec.json`,
`world/shot_bindings.json`, `vfx/blocking/<hash>/`, `renders/spatial_stills/`,
`renders/spatial_video/`, `world/audit_stills.json`, `world/audit_video.json`.

O piloto de quatro planos de cinco segundos produz `entregas/spatial_20s.mp4`.
`verification.json` registra duração e frames; `media_report.json` e
`contact_sheet.jpg` mostram inspeção de movimento e início/meio/fim.
Para reconstruir o relatório: `python -m script_pipeline.spatial_report --project PASTA`.

## Uso pela WebUI de decupagem

Na `decupagem_ui.py`, abra **Continuidade espacial 3D (opcional)** e preencha:

1. **Ativar estado espacial persistente**;
2. **Spec JSON de locação/planos espaciais** com um plano para cada ID estável do
   `parse/shot_plan.json`;
3. **Força de transformação do blocking**, normalmente entre `0.65` e `0.85`;
4. **Animatic diagnóstico** somente quando for necessário inspecionar um resultado
   mesmo que o gate visual bloqueie. Esse modo preserva o relatório e não libera o
   vídeo final.

O motor de imagens precisa ser `flux`. A WebUI repassa `--spatial-spec` e
`--spatial-denoise` ao orquestrador; a execução normal continua passando pelos gates
de stills antes de qualquer vídeo. Para uso direto no terminal:

```powershell
.venv/Scripts/python.exe -m script_pipeline.run_decupagem `
  --run-dir <RUN> --ate animatic --image-engine flux --video-engine ltx `
  --spatial-spec <RUN>/world/spatial_spec.json --spatial-denoise 0.78
```

O checkbox **Animatic diagnóstico** aciona `--visual-diagnostic` (2026-09-20): roda o gate e
grava o relatório, mas não bloqueia nem regenera, e a corrida é recusada além de `--ate animatic`
(nada de vídeo, lipsync ou montagem). `--no-visual-audit` continua existindo, mas **desliga** o
gate e deve ser reservado para testes. Um spec incompleto ou com IDs duplicados é recusado antes de
gerar stills.

## Teste validado: roteiro do Voo 702

O roteiro de emergência do Voo 702 foi ligado à WebUI pelo gerador
`script_pipeline/voo702_spatial.py`. O spec atribui `LOC_COCKPIT`, `LOC_CABIN` e
`LOC_SKY` por cena, mantendo entidades nomeadas dos pilotos e passageiros. O teste
de aceitação final usa quatro planos contíguos da cabine, ajustados para exatamente
20 segundos, 384×256 a 24 fps, com FLUX/ComfyUI, TTS reaproveitado e as fotos externas
de HA-EUN, MIN-JUN, JI-HO e SEO-YEON.

Artefatos da corrida:

- `shots/animatic.mp4`: 20,000 s, quatro planos;
- `shots/animatic_audit.json`: auditoria `ok`, zero still ausente e três falas aprovadas;
- `shots/visual_stills_audit.json`: `status=ok`, quatro de quatro planos aprovados;
- `world/voo702_cabin20_spatial_spec.json`: spec usado pela ponte da WebUI.

O plano completo preservado para continuação fica em `parse/shot_plan.full.json`;
o `parse/shot_plan.json` da corrida de teste contém apenas o recorte de 20 segundos.

## Limites explícitos

Código de piloto ainda presente: `demo_spec()` (ANA/PEDRO/livro) e o roteiro fixo
"INT. UNIVERSITY CORRIDOR" com que `prepare()` cria um projeto novo; a auditoria espacial
(`spatial_audit.py`) foi generalizada (sem porta/banco/livro), mas identidade continua sem
prova além de roupa fora dos closes com foto nominal.

O img2img RGB rígido preserva composição, mas em denoise baixo mantém aparência de
blocking. Por isso closes não usam o manequim como latente inicial. O teste de cabine
produziu aparência fotográfica e identidades aprovadas, mas identidade facial em muitos
ângulos e atribuição regional de múltiplos rostos ainda exigem auditoria. Depth/masks/pose
são exportados; nesta receita os modelos recebem RGB e keyframes, não todos os passes.

Os manequins são procedurais. Importação automática de GLB/rigs, retargeting completo,
figurinos externos, biblioteca extensa de figurantes, resolução de locações por aliases,
enquadramento automático segundo o eixo de 180° e migração espacial automática de
bíblias antigas não fazem parte desta primeira versão. A pose exportada retém juntas
ocultas e recorta pelo frustum; não é segmentada por oclusão.

O gate visual é evidência amostrada, não prova de todos os frames nem de identidade
facial. Ele pode bloquear um resultado inconclusivo. Não reduzir suas exigências para
aprovar um vídeo com geometria/ação errada; corrigir a receita ou a encenação e auditar
novamente. Nenhum projeto legado é habilitado automaticamente.

## Testes

```powershell
.venv/Scripts/python.exe -m pytest tests/test_spatial_pipeline.py tests/test_production_project.py tests/test_gate_hardening.py tests/test_generate_storyboards_zimage_guard.py tests/test_motion_director.py tests/test_pose_video.py -q
```

Cobertura: replay/idempotência, rollback, evento com pai diferente, coordenadas inválidas,
projeção e orientação, grade temporal e destino antes do trim, continuidade entre câmeras,
cache alterado, backend incompatível e aprovação obsoleta. Testes numéricos não substituem
a inspeção do vídeo. O resultado real da corrida fica nos relatórios do projeto piloto.
