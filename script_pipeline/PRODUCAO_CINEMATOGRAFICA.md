# Produção cinematográfica — fluxo de aprovação

Este fluxo trata o filme como uma cadeia de decisões editoriais e visuais, não
como uma sequência de chamadas a modelos. `completed` significa que uma etapa
produziu um artefato; somente os gates aprovados liberam a próxima fase.

## Ordem recomendada

1. **Preparar roteiro e elenco.** Converter prosa em cenas e unidades visuais
   explícitas, seguindo [`GUIA_ROTEIRO.md`](GUIA_ROTEIRO.md). Cadastrar as fotos fornecidas
   como `reference_image` de cada personagem (a importação reavalia a voz pela foto),
   fixar roupa/descritores, conferir os figurantes (`"extra"`) que o elenco criou e revisar
   qualquer elenco marcado `reference_needs_review` antes de gerar. Ler
   `parse/lacunas.md`: lacuna crítica é texto a completar, não algo para o modelo adivinhar.
2. **Planejar cobertura e espaço.** Garantir que cada ação essencial tenha um
   plano próprio (causa, ação, reação e consequência), atores/objetos, duração,
   enquadramento e locação estáveis. `shots/complexity_report.json` diz quais planos
   pedem blocking revisado ou previs 3D. Desde 2026-09-27 (`previs_spec.py`,
   MEMORIAL §3.135) o previs 3D dos planos complexos é gerado sozinho, sem spec
   manual — roda como parte de `run_decupagem` (`--previs3d`, padrão `complexas`)
   e produz `shots/previs_3d.png`/`.mp4` para revisão antes dos stills. Para
   continuidade 3D com estado explícito e persistente entre cenas (spec escrito à
   mão, IDs estáveis), continue com `SPATIAL_PIPELINE.md`.
3. **Gerar stills e revisar o animatic.** Rodar até `--ate animatic`, revisar a
   sequência inteira e corrigir identidade, eixo, geografia, figurino e ritmo
   antes de gastar GPU com vídeo.
4. **Gerar movimento e revisar clipes.** Cada ação do plano deve corresponder a
   um primitivo compatível. Uma ação essencial convertida em `idle`, ou um
   plano fechado que não pode mostrar a ação, é erro de planejamento e deve ser
   corrigida antes da renderização final.
5. **Aprovar gates.** As auditorias de still/vídeo reprovam planos, tentam
   regenerá-los e bloqueiam a corrida se continuarem reprovados. A decisão exige
   tanto o veredito visual geral quanto correspondência de enquadramento. Não
   use `--visual-unresolved continue` para uma entrega.
6. **Lipsync, mix, identidade e entrega.** Após os gates visuais, executar
   lipsync/mix; auditar identidade facial antes de montar; montar e executar
   `verify_output --strict`. Qualquer erro deixa a corrida não aprovada.

## Comandos

Prévia barata para revisão editorial:

```powershell
.venv/Scripts/python.exe -m script_pipeline.run_decupagem `
  --run-dir outputs/decupagem/MINHA_CORRIDA `
  --script outputs/decupagem_input/roteiro.txt `
  --style tenso --character-sheet --motion-conditioning --ate animatic
```

Depois de revisar/corrigir o animatic, retome sem `--ate` para continuar até a
entrega. O modo final sempre roda gates visuais, gate de identidade
pré-montagem e verificação técnica estrita. Se alguma aprovação falhar, a
execução retorna erro e preserva relatórios em `shots/` e `verification.json`.

O exemplo de spec espacial e o fluxo Blender estão em
[`SPATIAL_PIPELINE.md`](SPATIAL_PIPELINE.md). Bypass de gate serve apenas para
investigação e é recusado no modo final.

## Limites conhecidos

- O gate semântico depende do modelo de percepção e ainda requer revisão humana
  do storyboard/animatic.
- O auditor facial não prova identidade quando não detecta rosto. Planos abertos
  sem rosto suficiente ficam não conclusivos; os gates visuais devem confirmar
  roupa, silhueta e posição nesses planos.
- O spec espacial ESCRITO À MÃO (`SPATIAL_PIPELINE.md`) continua explícito e
  fornecido pelo operador — é o caminho para estado persistente entre cenas e
  eventos temporais explícitos. O previs 3D AUTOMÁTICO (`previs_spec.py`) é
  outra coisa: um rascunho de blocking derivado da decupagem, só para revisão
  de posição/câmera nos planos complexos, sem estado persistente entre corridas.
  Não confunda um pelo outro nem marque a continuidade 3D manual como cumprida
  sem `world/spec.json`, bindings e relatório espacial aprovado.
