# Registro de licenças — o que pode e o que não pode ser comercial

**Este documento é um registro, não uma trava.** A orientação do projeto é
escolher a melhor ferramenta para cada função, construir o sistema unificado e
funcional, e **só depois** decidir se o uso será pessoal, interno ou comercial.
Este arquivo existe para que essa decisão, quando chegar, seja possível — e não
uma arqueologia.

Levantado em 2026-08-20, lendo os arquivos `LICENSE` instalados nesta máquina,
não resultados de busca. Onde não li a fonte, está marcado como **não verificado**.

Revisto em **2026-08-21**: seção 4 reconferida contra os bancos SQLite reais
(não amostra), AIST++ resolvido, e as quatro bibliotecas do ChoreoEngine
inventariadas — uma delas já é 🟢 e não estava listada.

Legenda: 🟢 comercial liberado · 🟡 comercial com condição/limite · 🔴 comercial
vedado ou exige acordo à parte · ⚪ não verificado.

---

## 1. Geração de vídeo e imagem

| Componente | Licença | Comercial | Condição que importa |
|---|---|---|---|
| **LTX-2 / LTX 2.3 22b** | LTX-2 Community License (05/01/2026) | 🟡 | Livre **abaixo de US$ 10 milhões** de receita anual. Acima disso, exige *Commercial Use Agreement* pago com a Lightricks. Lido em `LICENSE`, seção linhas 87-95 |
| **IC-LoRAs 2.3** (union-control, ingredients, distilled-384) | segue a LTX-2 Community | 🟡 | Mesmo limite de receita acima |
| **Upscaler espacial x2** | segue a LTX-2 Community | 🟡 | idem |
| **Gemma 3** (text encoder) | Gemma Terms of Use (Google) | ⚪ | Permite comercial com política de uso proibido; **não verificado na fonte** |
| **SD 3.5 Medium** | Stability Community License | ⚪ | Costuma liberar comercial abaixo de um teto de receita; **não verificado na fonte** |
| **FLUX** (krea / klein / dev) | licenças distintas por variante | ⚪ | `dev` é tipicamente **não comercial**; `klein`/`krea` variam. **Verificar antes de publicar** — é o item mais provável de reprovar |

## 2. Corpo, personagem e 3D

| Componente | Licença | Comercial | Condição que importa |
|---|---|---|---|
| **SAM 3D Body** (Meta) | SAM License (19/11/2025) | 🟢 | Comercial liberado. Veda uso militar/armas/ITAR e usos sob controle de exportação. Lido em `sam-3d-body/LICENSE` |
| **Hunyuan3D-2** (Tencent) | Tencent Hunyuan 3D 2.0 Community | 🟡 | **Não vale na União Europeia, Reino Unido e Coreia do Sul.** Exige licença própria da Tencent acima de **1 milhão de usuários ativos mensais**. Lido em `Hunyuan3D-2/LICENSE`, seções 3 e 4 |
| **SMPL-X** | acadêmica; comercial via Meshcapade | 🔴 | Foi por isto que o `choreo/volume.py` optou por cápsulas próprias em vez de SMPL-X |
| **GVHMR** | uso não comercial | 🔴 | Usado hoje em `bridge/export_3d.py`. O `export_3d.py` já descarta a malha e leva só as juntas, mas **continua derivado de modelo não comercial** |
| **DAZ Genesis 9** | EULA da DAZ 3D | ⚪ | A Genesis costuma permitir render comercial; redistribuir a malha, não. **Não verificado na fonte** |
| **MHR — Momentum Human Rig** (Meta) | Apache 2.0 | 🟢 | Repo `facebookresearch/MHR`, clonado em `LTX-2-OPTIMIZED/MHR`. Traz o conversor `tools/mhr_smpl_conversion`, que o README declara servir a "SAM3D Outputs (MHR) → SMPL/SMPLX" |
| **pymomentum** (Meta, via conda-forge) | Apache 2.0 | 🟢 | Ambiente conda `mhr`, build CPU. ⚠️ Cuidado: `pyMomentum` do PyPI é **outra empresa** (Momentum Teknoloji AS), colisão de nome — não é este |
| **MoCapAnything V2** | **MIT** | 🟢 | Em `Documents/MocapAnything` (9 GB, checkpoint `video2pos2rot_epoch60.pt` 443 MB). A licença mais permissiva de toda a pilha. Ressalva do próprio README: é **reimplementação não oficial** do paper |
| **Arquivos de modelo SMPL/SMPL-X** | acadêmica MPI | 🔴 | Já presentes em `GVHMR/inputs/checkpoints/body_models/`. O conversor MHR→SMPL-X **depende deles**, então o resultado herda a restrição não comercial |

## 3. Software e runtime

| Componente | Licença | Comercial | Observação |
|---|---|---|---|
| **ComfyUI** | GPL-3.0 | 🟡 | Livre inclusive comercial, mas é **copyleft**: distribuir um produto que o incorpore obriga a abrir o código derivado. Rodar como serviço interno não dispara isso |
| **Blender** | GPL | 🟢 | O que você **produz** com ele é seu, sem contaminação |
| **Godot 4.7** | MIT | 🟢 | Sem restrição relevante |
| **three.js** | MIT | 🟢 | Vendorizado em `Interface para Editor de Danças/vendor/` |
| **Ollama** | MIT | 🟢 | O runtime; os **modelos** têm licença própria |
| **qwen3 / qwen2.5-coder** | Apache 2.0 | 🟢 | Usado pelo coreógrafo por texto |
| **onnxruntime-gpu** | MIT | 🟢 | — |
| **DWPose / detector de pose** | ⚪ | ⚪ | O extrator ONNX é o coração do `extract/`; **não verificado** |

## 4. Dados de movimento — o ponto mais sensível

| Fonte | Licença | Comercial | Situação real |
|---|---|---|---|
| **CMU MoCap** | `cmu-mocap-free` | 🟢 | "free for all uses"; pode entrar em produto vendido. **Não pode revender os dados**, nem convertidos. Registrado em `ChoreoEngine/packs/LICENSES.md` |
| **AIST++ — anotações** | `cc-by-4.0` | 🟢 | Verificado 2026-08-21. Documentado em `ChoreoEngine/packs/LICENSES.md:58-73` e declarado em `choreo/aist.py:48`. **Já importado**: 5.425 primitivas, todas com a licença gravada na proveniência |
| **AIST++ — `motions.zip` (parâmetros SMPL)** | `cc-by-4.0` | 🟢 | Baixado em 2026-08-21 do release v1.0. 411 sequências (27,2% do `keypoints2d`), formato conferido. Mesma licença das demais anotações |
| **AIST++ — vídeos e áudio** | termos próprios do AIST | 🔴 | Uso comercial exige consentimento **por escrito**; redistribuição proibida. **Não foram baixados** — só as anotações |
| **Corpus próprio — `Moving/MovExtraido/library`** | **`unknown`** | 🔴 | 6.262 primitivas. **É a biblioteca que o editor abre por padrão** (`ChoreoEngine.bat`). Ver abaixo |

### O corpus que move o editor hoje

Reconferido em **2026-08-21** direto no SQLite de `Moving/MovExtraido/library`
(é o caminho que `ChoreoEngine.bat` passa em `LIBRARY=`):

- **6.261 de 6.262** primitivas têm `origin = "user"` — vieram da sua coleção de vídeo, não de packs licenciados.
- `provenance.license`: **6.261 `unknown`** + 1 `gerado`. Não é amostra — é a contagem completa da tabela.
- Os clipes trazem `kp2d`+`kp133` (corpo inteiro 2D) e **nenhum `kp3d`**.
- Os caminhos de origem apontam para vídeos de redes sociais (nomes como `@usuario-...`), ou seja, **material de terceiros com pessoas identificáveis**.

O próprio projeto já definiu o critério, em `packs/LICENSES.md`:

> O `Provenance.license` de toda primitiva importada tem que carregar a chave da
> tabela. É o campo que o `dataset.py --no-license-filter` desliga — e o que
> separa "biblioteca de estudo" de "publicável".

Por esse critério, **a biblioteca em uso está inteiramente do lado "estudo"**.
Nada disso impede construir e testar o sistema — impede publicar o que sai dele
usando esse acervo. As saídas, quando a decisão chegar:

1. Recompor a partir do **CMU MoCap** (🟢) e de material próprio filmado, mantendo
   o corpus atual só como laboratório;
2. Registrar licença clipe a clipe e usar o filtro que já existe; ou
3. **Usar a `library_aist`, que já está no disco e já é 🟢** — ver a tabela abaixo.
   É a saída mais barata: não exige baixar, extrair nem registrar nada.

### As outras bibliotecas do ChoreoEngine (medidas em 2026-08-21)

Além do corpus acima, `ChoreoEngine/data/` guarda mais quatro índices que o
editor **não** abre hoje:

| biblioteca | primitivas | licença na proveniência | comercial | 3D (`kp3d`) |
|---|---|---|---|---|
| `data/library_aist_smpl` | **5.425** | `cc-by-4.0` | 🟢 | **4.473 (82,5%)** + **1.129 com SMPL (20,8%)** |
| `data/library_aist` | 5.425 | `cc-by-4.0` | 🟢 | 4.473 (82,5%), sem SMPL |
| `data/library` | 396 | `unknown` | 🔴 | 0 |
| `data/library_dance` | 6 | `reference-study` | 🔴 | 0 |
| `data/library_wb` | 4 | `internal-test` | ⚪ | 0 |

Duas consequências práticas, e elas puxam para o mesmo lado:

- **Licença**: a `library_aist` é o único acervo de tamanho útil com licença
  comercial gravada. Trocar o `LIBRARY=` do `.bat` já move o editor do 🔴 para
  o 🟢, sem tocar em código.
- **Profundidade**: é também o único com `z` medido. O corpus atual só tem 2D,
  e por isso a órbita da câmera no editor usa profundidade **inferida** por
  encurtamento de osso — que degrada fora do ângulo de captura original.

O que a `library_aist` **não** tem: `kp133` (mãos e rosto). O corpus atual tem.
Para a faixa de controle DWPose que o LTX consome, isso é uma perda real de
detalhe nas extremidades — não é troca sem custo.

---

## 5. Leitura rápida por cenário

- **Uso pessoal / estudo / P&D interno:** tudo o que está instalado serve. Nenhum
  item bloqueia.
- **Produto comercial abaixo de US$ 10 mi de receita e 1 mi de usuários:** LTX,
  SAM 3D Body, Hunyuan (fora de UE/UK/Coreia), CMU MoCap e as **anotações do
  AIST++** passam. Os pontos a resolver são **GVHMR/SMPL-X**, a variante de
  **FLUX** e **o corpus** — este último com saída conhecida: trocar o
  `LIBRARY=` para `data/library_aist`.
- **Distribuir software:** o **copyleft do ComfyUI** vira a pergunta central.
- **Publicar vídeo gerado com o acervo atual:** é o cenário que **não** fecha hoje,
  por causa da procedência do corpus — não por causa dos modelos. Fecha no
  momento em que o editor passar a compor a partir da `library_aist`.

## 6. Pendências de verificação

Itens ⚪ acima, em ordem de risco: variante do FLUX · DWPose ·
Stability (SD 3.5) · Gemma 3 · EULA da DAZ · `data/library_wb` (4 clipes de teste).

**Resolvido em 2026-08-21:** AIST++ saiu da lista. As anotações são CC BY 4.0
(comercial 🟢) e são a única coisa baixada; vídeos e áudio do AIST Dance DB
continuam 🔴 e **não** estão no disco — `packs/aist_plusplus/` tem apenas
`cameras`, `keypoints2d`, `keypoints3d` e `splits`.
