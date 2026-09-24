# Produção cinematográfica — fluxo de aprovação

Este fluxo trata o filme como uma cadeia de decisões editoriais e visuais, não
como uma sequência de chamadas a modelos. `completed` significa que uma etapa
produziu um artefato; somente os gates aprovados liberam a próxima fase.

## Ordem recomendada

1. **Preparar roteiro e elenco.** Converter prosa em cenas e unidades visuais
   explícitas. Cadastrar as fotos fornecidas como `reference_image` de cada
   personagem, fixar roupa/descritores e revisar qualquer elenco marcado
   `reference_needs_review` antes de gerar.
2. **Planejar cobertura e espaço.** Garantir que cada ação essencial tenha um
   plano próprio (causa, ação, reação e consequência), atores/objetos, duração,
   enquadramento e locação estáveis. Para continuidade 3D, fornecer um spec
   espacial com IDs estáveis e rodar `SPATIAL_PIPELINE.md`; o pipeline não cria
   automaticamente um mapa 3D confiável a partir de qualquer prosa.
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
- O spec espacial é explícito e fornecido pelo operador. A tradução automática
  de roteiro livre em blocking 3D completo ainda é futura; não marque essa etapa
  como cumprida sem `world/spec.json`, bindings e relatório espacial aprovado.
