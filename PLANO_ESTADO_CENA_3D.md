# Plano de implementação — estado persistente de cena e composição 3D

Data: 19/09/2026. Escopo interpretado: implementar a arquitetura proposta no texto anexado, de roteiro e Qwen até estado espacial persistente, blocking 3D, stills condicionados e vídeo LTX. Este documento é planejamento; não altera a execução do pipeline.

## 1. Base efetivamente encontrada

A referência mais avançada é o working tree, incluindo arquivos ainda não rastreados e alterações posteriores ao commit `f0fe827`, de 17/09. O `README.md` é upstream; o estado local é descrito por `CLAUDE.md`, `MEMORIAL.md` §§3.89–3.95 e pelo código abaixo. A seção P0 do memorial ainda é de 18/09 e deve ser lida junto com os registros de 19/09.

| Componente | Já existe | Lacuna para a proposta |
|---|---|---|
| `production_project.py` | Projeto mestre, unidades do roteiro, IDs de cena/locação/personagem/plano, bíblia, timeline, versões e orçamento | Estado espacial transacional, catálogo de objetos/figurinos/figurantes e eventos de continuidade |
| `conform_plan()` | Liga planos a locação, inventário textual e elenco; cria `vfx/jobs.json` | Resolver versões de estado e câmera; os jobs nascem sem cena/passes atribuídos |
| `blender_production.py` | Cabine procedural, câmera configurável por job, `.blend`, beauty e EXR com Z/object index | Compositor genérico, atores, poses, attachments, passes separados e integração automática |
| `render_shots.py` / `generate_storyboards.py` | Stills, referências de personagem, segunda referência e cache por conteúdo | Entrada explícita de geometria/controles, associação identidade→instância e cache do estado 3D |
| `motion_director.py` / `pose_video.py` | InterGen→pose→LTX Union-Control; projeção ortográfica | Usar a mesma câmera em perspectiva, escala, origem e tempo do compositor |
| `ltx25_backend.py` | Keyframes com `LTXVAddGuideAdvanced`, guias IC e caminho de vídeo existente | Compilar estados t0/t1 para os índices efetivos e integrar ao plano de produção |
| Gates e regeneração | Continuidade textual, gate visual Qwen3-VL, retry seletivo, invalidação de referência reprovada | Diagnóstico espacial, vínculo de aprovação ao hash dos insumos e correção pela causa |

Limite da evidência: o memorial registra três planos reais com movimento pelo LTX. O endurecimento mais recente dos gates (§3.95) tem testes e verificações sobre dados reais, mas não uma corrida completa nova com GPU. Nesta análise não foram executados testes ou renders. O protótipo Blender foi inspecionado no código, sem validar seus artefatos em runtime.

## 2. Decisões de arquitetura

Estender `production_project.py`, sem criar outro conceito de projeto. Separar quatro coisas:

1. **Assets canônicos:** locação, corpo, rosto, roupa e objeto, com IDs e versões.
2. **Estado narrativo:** presença, transformações, pose, posse, attachments e tempo da história.
3. **Plano:** câmera e intervalo observado daquele estado; trocar enquadramento não move o mundo.
4. **Artefatos:** passes, stills e vídeos derivados, com hashes e aprovação próprios.

Fluxo proposto:

```text
roteiro preservado → eventos propostos pelo Qwen → validação → WorldStore
                                                           ↓
shot_plan + câmera + intervalo → SceneComposer / Blender → passes
                                                           ↓
                            identidade + controle compatível → still
                                                           ↓
                              gate visual → LTX → gate → montagem
```

Qwen propõe eventos estruturados; código determinístico valida e aplica. Ele não escreve diretamente no banco e não substitui falas/unidades do roteiro. Resolver posições por anchors e restrições sempre que possível; números inventados pelo LLM não equivalem a blocking válido.

Persistir a geometria da locação não significa congelar seus ocupantes. Voltar ao corredor reutiliza o asset, mas resolve o estado correspondente àquele momento da história. Flashbacks, versões editoriais e cobertura simultânea precisam de referências explícitas ao estado; ordem de geração não pode determinar continuidade.

### Fonte da verdade e armazenamento

Adotar SQLite para estado e eventos; manter roteiro/editorial nos contratos existentes. Arquivos de bíblia atuais servem como entrada de migração; depois, seus campos espaciais passam a ser exportações do banco, sem escrita dupla independente.

Estrutura aditiva sugerida:

```text
<projeto>/world/world.sqlite
<projeto>/world/snapshots/<state_hash>.json
<projeto>/assets/locacoes/<location_id>/<asset_version>/
<projeto>/assets/personagens/<character_id>/<asset_version>/
<projeto>/assets/figurinos/<wardrobe_id>/<asset_version>/
<projeto>/assets/objetos/<prop_id>/<asset_version>/
<projeto>/vfx/<shot_id>/<dependency_hash>/
```

Tabelas mínimas: `schema_migrations`, `assets`, `entities`, `wardrobes`, `scene_states`, `state_entities`, `continuity_events`, `cameras`, `shot_bindings`, `generated_assets`. Locação é asset; personagem/figurante/objeto têm instâncias persistentes. Relações e attachments têm origem, destino, socket e transformação local.

Cada transação valida a versão-pai e registra `event_id`, `source_unit_id`, tempo narrativo, precondições e delta. Reaplicar o mesmo evento é idempotente. Transferência de objeto altera dono anterior e novo na mesma transação; falha não deixa posse parcial. Snapshots imutáveis permitem replay e retomada; exportação só depois do commit. Primeiro MVP usa um escritor por projeto.

### Contratos mínimos

- `SceneState`: schema, ID/hash, versão-pai, locação e versão do asset, tempo narrativo, instâncias e iluminação.
- `EntityState`: ID, asset, roupa, posição em metros, rotação, escala, pose, presença e attachment opcional. Ausência do enquadramento é diferente de saída da cena.
- `ContinuityEvent`: ação tipada, referências, origem no roteiro, precondições e operações permitidas.
- `ShotBinding`: ID estável do plano, estado inicial/final, câmera, intervalo narrativo e keyframes.
- `CameraSpec`: transform, lente, sensor, clipping, resolução e aspecto. Convenção canônica Z-up compatível com Blender; conversão explícita dos movimentos Y-up.
- `ControlBundle`: hashes de estado/câmera/assets, arquivos, formatos, convenção de profundidade, espaço das normais, mapa máscara→entidade, fps e frames.
- `RenderRecipe`: checkpoint e revisão, workflow/nodes, controles aceitos, referências, seed, forças e parâmetros.

A regra de 180° exige eixo de ação explícito e validação da posição das câmeras. A existência de 3D sozinha não impede cruzar esse eixo. Mudanças intencionais devem ser registradas.

## 3. Etapas, dependências e critérios de conclusão

### F0 — Base e prova de compatibilidade

Inventariar versões locais de Blender, ComfyUI, nodes e pesos sem atualizar ambientes. Registrar hashes do código e preservar as alterações existentes. Executar a suíte focal atual de projeto, gates, pose e movimento como baseline.

Criar uma prova isolada: geometria simples com dois manequins, uma porta e um objeto, duas câmeras, referências canônicas. Comparar o caminho atual com um caminho condicionado por geometria, mantendo demais parâmetros comparáveis e medindo memória, tempo e aderência visual.

**Decisão obrigatória:** o padrão local é `flux-2-klein-9b-fp8.safetensors`. Não assumir que controles de FLUX.1, IP-Adapter/FaceID ou qualquer LoRA funcionam nesse modelo. Inventariar capacidades por backend. Se não houver controle estrutural compatível e demonstrado no Klein, escolher um backend específico para a rota espacial após a prova; preservar o backend atual para projetos legados. Usar beauty como referência é experimento, não prova de controle de profundidade.

**Concluída quando:** uma receita completa consegue executar no hardware local e demonstra identidade e composição juntas, ou a incompatibilidade fica documentada com alternativa concreta validada. Banco e schemas podem avançar, mas não se promete integração visual antes deste resultado.

### F1 — Estado, catálogo e migração

Criar `world_schema.py`, `world_store.py` e `world_migration.py`; integrar o vínculo em `production_project.py`. Importar IDs já existentes, nomes alternativos e referências. Campos desconhecidos ficam marcados como pendentes; não inventar coordenadas para migrar.

Manter ID estável por plano e por instância. Verificar colisões quando uma unidade do roteiro recebe várias coberturas ou subdivisões; `source_part` e índice de arquivo não devem ser a única identidade editorial.

**Concluída quando:** abrir, migrar, fechar e retomar o projeto preserva IDs; replay reproduz o hash; evento repetido não duplica ação; precondição inválida reverte a transação. Projeto legado continua no fluxo atual até habilitar a rota espacial.

### F2 — Compilador de eventos do roteiro

Criar `scene_state_planner.py` e `continuity_rules.py`. Aproveitar `parse_screenplay.py`, `restore_source()` e `shot_plan.py`. Qwen recebe catálogo, estado anterior relevante e unidade do roteiro; devolve eventos tipados como entrar, sair, mover, orientar, vestir, pegar e transferir.

Validar IDs, versões, intervalos, valores finitos, posse exclusiva e existência do receptor. Distinguir fatos do roteiro, escolhas de encenação e pendências. Coberturas do mesmo beat não reaplicam sua ação; uma transferência mostrada por dois ângulos acontece uma vez na história.

**Concluída quando:** teste de retorno à locação após outras cenas, troca de câmera e transferência de objeto funciona sem teleporte nem reaplicação; uma referência desconhecida produz diagnóstico acionável antes da GPU.

### F3 — Compositor e câmera compartilhada

Generalizar `blender_production.py` e adicionar `scene_composer.py` e `camera_geometry.py`. Carregar locação reutilizável, instanciar manequins/objetos, aplicar estados e poses e posicionar câmera. Preservar a cabine procedural como um preset.

Extrair depth e máscaras do EXR, com contrato numérico explícito e IDs por instância — o protótipo atual agrupa assentos pelo mesmo índice. MVP: beauty, depth, máscara de instâncias e pose. Normal/lineart são extensões apenas quando um consumidor precisar delas.

Substituir autoenquadramento ortográfico na rota espacial de `pose_video.py` por projeção com a câmera comum. Tratar oclusão e pontos atrás da câmera. Fixar conversões de eixos, unidades, lente/sensor e transformação local→mundo dos movimentos.

**Concluída quando:** trocar só a câmera preserva o hash do estado do mundo; landmarks e juntas projetados batem com o render Blender dentro da tolerância numérica acordada; objeto preso à mão acompanha o rig; passes compartilham enquadramento. Começar com previews leves e aspecto idêntico, ajustando dimensões à grade do backend final.

### F4 — Stills condicionados, cache e identidade

Criar `spatial_conditioning.py`, com matriz explícita de capacidades. Estender `generate_scene_storyboard()` e `_still_for_shot()` para receber `ControlBundle`, sem confundir imagem de rosto, referência de ambiente e controle geométrico.

Para múltiplos personagens, vincular cada identidade à sua região/instância quando o backend suportar. Se a receita não resolver isso, bloquear o caso ou usar composição regional validada; duas referências globais não garantem atribuição correta dos rostos.

Incluir no cache hashes de estado, câmera, assets, passes, todas as referências, recipe e versão de workflow. Separar artefato produzido de artefato aprovado; o still só vira referência aprovada após o gate. Preservar `location_master.py` e impedir que referências rejeitadas voltem a ser promovidas.

**Concluída quando:** edição de câmera/objeto/roupa invalida os derivados corretos; retomar sem alteração reutiliza resultados; dois atores mantêm identidades distintas em geral, close e contraplano. Medir aderência geométrica e identidade separadamente.

### F5 — Estado temporal e LTX

Criar `shot_conditioning.py`; ligar o compositor a `motion_director.py`, `render_shots_stage.py` e à API já existente de `ltx25_backend.py`. Gerar stills t0/t1 e, quando necessário, vídeo de pose/depth com a mesma câmera e relógio.

O backend local quantiza índices de guia em múltiplos de 8. Compilar índices efetivos, resolver duplicações após quantização e distinguir duração gerada da duração editorial. O destino precisa aparecer antes do trim final. Testar contagem final de frames, crop de guias e combinações IC/keyframes; não assumir suporte conjunto porque as modalidades funcionam separadas.

**Concluída quando:** clipe curto mantém composição e identidade e realiza a transição; entrega do objeto termina com posse visível correta. Conferir quadros antes/durante/depois do contato, além de início/meio/fim. H3 ControlNet fica fora do caminho crítico por seu bloqueio de VRAM já registrado.

### F6 — Orquestração, correção e interface

Em `run_decupagem.py`, inserir resolução do estado e render técnico após a conformação final dos planos e antes dos stills. Estender a UI com modo espacial, estado/câmera usados, preview de blocking e diagnóstico. Propostas de flags: `--spatial-mode off|strict`, `--until blocking` e seleção por ID estável; harmonizar nomes com o CLI existente na implementação.

Executar em lotes: planejamento → descarregamento Qwen → blocking CPU → stills → auditoria → vídeo → auditoria. Reaproveitar `ollama_runtime.py` e os mecanismos existentes de liberação de memória. Não manter Qwen3-VL, FLUX e LTX residentes ao mesmo tempo na 3090.

Retry por causa: geometria errada volta ao blocking; aparência errada volta ao still; trajetória errada volta ao condicionamento temporal. Seed nova sozinha não corrige câmera ou posse. Tentativas não modificam o estado aprovado do roteiro. Toda mudança semântica cria outra versão e invalida somente dependentes; aprovação visual fica vinculada ao hash dos artefatos avaliados.

**Concluída quando:** interrupção/retomada, geração parcial e reprovação não perdem manifestos nem reaproveitam aprovação antiga. O modo espacial estrito não degrada silenciosamente para prompt puro.

## 4. Primeiro marco de produção

Pilotar uma locação, dois personagens, dois figurantes persistentes, uma roupa por pessoa e um objeto transferível. Usar seis planos: geral, close A, contraplano B, transferência, outra locação e retorno ao set original. A cabine do Voo 702 pode ser a segunda validação, aproveitando o preset; começar pelo set simples facilita isolar falhas.

Entregas do marco: banco e snapshots, catálogo mínimo, seis bindings, passes, stills, clipes curtos, relatórios e comparação com o fluxo atual. A troca de câmera e o retorno precisam reutilizar a mesma geometria; a transferência deve persistir independentemente da ordem de render. Figuração fora de quadro continua no estado.

Testes necessários: transações/replay, migração, múltiplas coberturas, transformação Y-up→Z-up, projeção/oclusão, caches, quantização temporal e retomada. Validação visual deve incluir identidade, roupa, lado do eixo, porta, objeto e presença dos figurantes quando visíveis. Definir tolerâncias visuais na prova F0 antes de declarar sucesso; não transformar similaridade facial ou aprovação do VLM em prova isolada de fidelidade.

Ordem recomendada: **F0 → F1 → F2 → F3 → F4 → F5 → F6**. A prova F0 reduz o maior risco; F1–F3 entregam a memória espacial; F4 demonstra que ela chega aos pixels. Só expandir a biblioteca ou rodar um filme completo depois do piloto. Não é necessário começar com centenas de figurantes, rigs detalhados, treinamento de LoRA ou todos os passes.

## 5. Referências técnicas e limites da proposta original

- Código local: `production_project.py`, `blender_production.py`, `render_shots.py`, `generate_storyboards.py`, `pose_video.py`, `motion_director.py`, `run_decupagem.py` e `ltx25_backend.py`.
- [BFL — FLUX.1 Tools](https://bfl.ai/blog/24-11-21-tools): Depth/Canny são variantes específicas de FLUX.1; isso não comprova compatibilidade com o Klein instalado. A página também informa descontinuação desses endpoints na API BFL; a seleção local deve verificar pesos e runtime concretos.
- [LTX — nodes oficiais ComfyUI](https://docs.ltx.io/open-source-model/integration-tools/ltx-comfy-ui-nodes): referência de integração; contratos efetivamente instalados e o wrapper local prevalecem para o piloto.

Geometria condicionante reduz ambiguidade; não torna o modelo generativo um renderizador determinístico nem garante continuidade perfeita. O objetivo verificável é tornar o estado reproduzível, controlar melhor os pixels e bloquear desvios antes da montagem.
