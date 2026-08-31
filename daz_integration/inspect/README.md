# Recon: personagem Genesis animado pela dança do ChoreoEngine

2026-08-17. Confirmado rodando DENTRO do DAZ Studio (não adivinhado):

- **Genesis 9 carrega e resolve**: `App.getContentMgr().openNativeFile(...)`
  com `People/Genesis 9/Genesis 9.duf` funciona, figura vira seleção primária.
- **Esqueleto real, 316 ossos**, nomes limpos e mapeáveis 1:1 no rig do
  COCO-17 que o ChoreoEngine já usa: `hip`, `pelvis`, `spine1..spine4`,
  `neck1`, `neck2`, `head`, `l_shoulder`/`r_shoulder`,
  `l_upperarm`/`r_upperarm`, `l_forearm`/`r_forearm`, `l_thigh`/`r_thigh`,
  `l_shin`/`r_shin`, `l_hand`/`r_hand`, `l_foot`/`r_foot` (ver
  `genesis9_skeleton.json`, dump completo).
- **Exportador FBX existe**: `App.getExportMgr()` lista "Autodesk FBX" entre
  5 exportadores (BVH, Poser CR2, FBX, Mental Images, Wavefront OBJ).

## Bloqueado: `writeFile` sempre devolve 101

A API CORRETA foi identificada (obrigado ao trecho de referência que
resolveu as chamadas que eu estava adivinhando errado):

```js
var oExporter = App.getExportMgr().findExporterByClassName("DzFbxExporter");
var oSettings = new DzFileIOSettings();
oExporter.getDefaultOptions(oSettings);   // NAO getOptions() -- essa nao existe
oSettings.setBoolValue("Animation", false);
// ...
var nResult = oExporter.writeFile(FBX_OUT, oSettings);   // (caminho, settings)
```

Com isso, `findExporterByClassName`, `getDefaultOptions` e `writeFile`
rodam TODOS sem lançar exceção -- nenhum erro de API. Mas `writeFile`
devolve **código 101** de forma consistente em toda variação testada:

| variação | resultado |
|---|---|
| settings customizadas (Animation/Morphs/EmbedTextures=false, FBX201400) | 101 |
| settings 100% default, sem tocar em nada | 101 |
| com `Scene.selectAllNodes(false); fig.select(true)` explícito | 101 |
| caminho com `/` | 101 |
| caminho com `\\` (estilo Windows nativo) | 101 |

Nenhuma variação produziu o arquivo `.fbx`. O código 101 é consistente
o bastante (mesmo valor em 5 combinações diferentes de argumento) para não
ser erro de parâmetro meu -- é algum estado ou requisito que só a sessão
INTERATIVA do DAZ Studio satisfaz (suspeita, não confirmada: alguma
verificação de licença/plugin do exportador FBX, ou uma dependência de
viewport OpenGL que `-noPrompt` sem janela em foco não inicializa por
completo). Sem a lista de códigos de erro do `DzExporter` -- que não está em
nenhum lugar instalado nesta máquina -- não dá para diagnosticar mais fundo
sem abrir a GUI.

Nove ciclos de boot completo do DAZ Studio (1-2 min cada) foram gastos nisso.

### Duas hipóteses testadas e descartadas (2026-08-18)

1. **Componente do exportador ausente.** `libfbxsdk.dll` (2020.2 Release)
   confirmado presente em `DAZStudio6/`. O usuário reinstalou um componente
   do exportador FBX no meio da investigação -- **o código 101 se repetiu
   idêntico depois da reinstalação**. Não é isso.
2. **Carga assíncrona incompleta** (textura/morph da Genesis 9 ainda
   carregando quando o export roda). Testado bombeando `App.processEvents()`
   200 vezes entre o load e o `writeFile` -- **mesmo código 101**. Não é isso.

Onze tentativas ao todo, todas com código **101** idêntico: settings custom,
settings default puro, seleção explícita, caminho `/` e `\`, espera ativa
pós-load, antes e depois de reinstalar o componente. Essa uniformidade é
o próprio diagnóstico -- não é parâmetro, não é timing, não é instalação.
Sobra alguma verificação (licença? sessão interativa exigida?) que só a
GUI satisfaz, e sem a tabela de códigos de erro do `DzExporter` não dá
pra ir mais fundo por script.

**Parando aqui de vez.** O próximo passo que decide a causa de fato é rodar
o MESMO `File > Export > Autodesk FBX` pela interface gráfica uma vez -- se
funcionar manualmente, o bloqueio é específico do modo `-noPrompt`/headless
(e vale testar sem essa flag); se falhar manualmente também com o mesmo
tipo de erro, é algo na instalação que nem a GUI contorna.

## O que falta é UM passo manual, não o pipeline inteiro

Tudo depois da exportação já está pronto e testado quanto dá para testar sem
o arquivo:

1. No DAZ Studio (aberto normalmente, não por script): carregar
   `People/Genesis 9/Genesis 9.duf`, **File > Export > Autodesk FBX (.fbx)**,
   salvar em qualquer caminho. É a MESMA malha citada aqui, uma vez só —
   não precisa repetir por dança.
2. `bridge/daz_retarget.py` (já escrito, no ChoreoEngine): importa esse FBX
   no Blender, liga Damped Track nos ossos confirmados acima, anima com
   qualquer `.npz` de `kp2d` do ChoreoEngine, renderiza vídeo.

   ```
   blender -b --python bridge/daz_retarget.py -- personagem.fbx danca.npz saida.mp4
   ```
