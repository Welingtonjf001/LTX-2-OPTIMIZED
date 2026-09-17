# Avaliação do vídeo — Palace of Emerald Shadows

**Arquivo avaliado:** [movie.mp4](E:/Users/home/Documents/LTX-2-OPTIMIZED/outputs/decupagem_06_palace_emerald/final/movie.mp4).

O caminho informado continha separadores onde o arquivo existente tem sublinhados. A análise usou `outputs/decupagem_06_palace_emerald/final/movie.mp4`.

**Parecer:** o vídeo tem imagem plausível dentro de vários planos, mas ainda não está consistente como cena final. Os problemas mais importantes são continuidade das personagens, atribuição das ações e fechamento narrativo. Há também texto indesejado na imagem e indícios de problemas na fala gerada. Não é principalmente um problema de codec, resolução ou alguns frames isolados.

## O que foi verificado

- Decodificação integral dos 1.359 frames e análise temporal por fluxo óptico em CPU.
- Inspeção visual de 29 amostras distribuídas a cada 2 segundos, três amostras por plano, frames adicionais em resolução nativa e tiras de dois alertas temporais iniciais.
- Comparação com `parse/scenes_enriched.json`, `parse/shot_plan.json`, `characters/cast.json`, manifesto de stills e lista real da montagem.
- Medição de volume, silêncio, tela preta e congelamentos pelo FFmpeg.
- Duas transcrições locais Whisper base: idioma automático e português forçado. São evidência auxiliar; não equivalem a uma audição crítica nem provam cada palavra pronunciada. Sincronia fonema/lábio não foi mensurada com SyncNet.

O vídeo não foi alterado. Nenhum reparo, nova geração ou interpolação foi aplicado. O relatório editorial já existente na corrida foi consultado como contexto, mas não usado como substituto da inspeção deste arquivo final.

## Erros por trecho

Os tempos são aproximados, contados desde o início do arquivo; os limites vêm da soma das durações efetivas dos clipes normalizados. A faixa de vídeo começa cerca de 21 ms depois da faixa de áudio.

| Tempo | Gravidade | O que foi observado | Efeito e correção indicada |
|---|---|---|---|
| 00:00–00:13,17 | Alta | A chegada de Xiao-Lan é apresentada duas vezes: o primeiro plano já a mostra no interior, e aos 3,75 s ela reaparece entrando pelo arco. O enquadramento, aparência facial e detalhes da roupa mudam no corte. | Sensação de reinício da ação. Escolher uma única entrada e ligar sua conclusão ao plano de fala com posição e aparência preservadas. |
| Próximo de 00:02,50 | Alta | Há caracteres semelhantes a uma legenda/marca sobre a parte inferior da roupa. Confirmado no frame 60. | Contaminação visual que não pertence à cena. Regenerar ou remover localmente após conferir a extensão; não atribuir conteúdo ou autoria ao texto ilegível. |
| 00:13,17–00:25,83 | Alta | Mei-Li muda de rosto aparente, penteado e construção do vestido entre os planos. Aos 22,08 s o azul passa a uma túnica de gola alta, bem diferente do figurino bordado anterior. Xiao-Lan também aparece com outro cabelo e gola. | As personagens parecem recastadas entre tomadas. Fixar retratos e figurino aprovados antes de refazer estes planos. |
| 00:22,08–00:25,83 | Alta | A ação esperada é Mei-Li alisar as sobrancelhas diante do espelho; na amostra, Xiao-Lan toca a testa/cabelo de Mei-Li. O plano está cadastrado com subject=XIAO-LAN, embora o beat comece por Mei-Li. | A direção da ação já está ambígua no plano. Corrigir quem executa a ação, quem a recebe e o objeto de interação antes de gerar. |
| 00:25,83–00:35,25 | Muito alta | Ao fim do plano, por volta de 34–35 s, as duas mulheres usam praticamente o mesmo vestido azul com faixa vermelha e penteados muito semelhantes. A distinção visual entre princesa e criada desaparece. | Contaminação de identidade/figurino em plano com duas pessoas. Refazer com duas referências fixas e descrições individuais, posições claras e ação mais simples. Não é reparável por interpolação de poucos frames. |
| 00:35,25–00:49,83 | Alta | Xiao-Lan ajusta o próprio adorno de cabelo. O roteiro pede que ajuste o adorno de Mei-Li. O gesto reaparece nos dois planos consecutivos; no segundo ela também começa sentada, mudando a disposição da cena. | A relação entre as personagens é perdida. Mostrar as duas em um plano de interação; usar depois um contraplano de fala sem repetir o gesto. |
| 00:39,00–00:49,83 | Média/alta | O plano de fala dura 10,83 s e sustenta um gesto de cabelo já apresentado. A duração não é defeito isoladamente, pois há diálogo, mas a ação duplicada deixa a cobertura pouco variada. | Manter tempo para a fala e substituir o gesto por olhar/reação, sem cortar palavras para encurtar. |
| 00:49,83–00:56,63 | Alta | A dança é retomada no corte de 53,58 s, com mudança da faixa/detalhes do vestido. O vídeo termina ainda na dança. | Falta a virada final do roteiro: passos/anúncio do guarda, susto, corrida ao divã e pose de serenidade. Acrescentar esses beats à decupagem; não constam dos dez planos atuais. |

### Texto indesejado aos 2,50 segundos

![Caracteres sobre a roupa no frame 60](E:/Users/home/Documents/LTX-2-OPTIMIZED/auditoria/video_palace_emerald/frame_60.png)

### Perda de distinção entre as personagens aos 35 segundos

![Duas personagens com vestido e penteado semelhantes no frame 840](E:/Users/home/Documents/LTX-2-OPTIMIZED/auditoria/video_palace_emerald/frame_840.png)

### Ação incorreta mantida no plano de 39–49,83 segundos

![Xiao-Lan mexendo no próprio adorno ao longo do plano](E:/Users/home/Documents/LTX-2-OPTIMIZED/auditoria/video_palace_emerald/plano_07.jpg)

## Áudio e fala

**Confirmado por medição:** há áudio ao longo do filme; volume médio de -14,9 dBFS e pico de amostras de -0,5 dBFS. Esses números não são LUFS e não demonstram ausência de distorção perceptiva. Foi detectado um intervalo abaixo de -45 dBFS entre **43,94 e 44,98 s**. Pode ser pausa de fala; não deve ser preenchido automaticamente sem ouvir.

**Indício relevante:** o Whisper em detecção automática escolheu italiano; forçado para português, continuou reconhecendo diversas frases distantes do roteiro. Exemplos:

- Aos 3,85–13 s, a fala esperada sobre o príncipe atravessar o pátio foi transcrita com trechos como “O princípio de um ano do norte” e “seu sapo sento”.
- Aos 26,79–33,83 s, perguntas sobre cabelo e pó de arroz ficaram parcialmente ininteligíveis para o reconhecedor, embora “Estou bonita?” tenha sido recuperado.
- Aos 34,87–35,39 s, apareceu “Buiau!”, próximo do fim de um prompt que contém o termo de figurino **buyao**. Isso sugere possível vazamento de descrição para a fala, mas não prova a palavra efetivamente pronunciada.
- Aos 54,43–55,11 s, “Eu sabia!” foi reconhecido. Perto de 56 s, ambas as passagens do reconhecedor produziram outra expressão curta não prevista; pode ser vocalização gerada ou erro do ASR.

Esses resultados justificam revisão auditiva e validação das falas, não a afirmação de que o áudio esteja literalmente em italiano. O modelo ASR base pode errar com sotaque, música e voz sintética. Arquivos: [transcrição automática](transcricao.json) e [português forçado](transcricao_pt.json).

Os dez registros de `mixed_clips.json` têm `audio_path=null` e `lipsync_applied=false`: esta montagem preserva o áudio nativo dos clipes, sem aplicação da voz TTS/lip-sync nesses registros. Para garantir texto exato em português, considerar TTS controlado e sincronização posterior, ou validar cada geração nativa antes da montagem. Não há evidência suficiente aqui para quantificar atraso labial.

## Diagnóstico técnico

| Item | Resultado |
|---|---|
| Imagem | H.264, 864 × 480, 24 fps constantes, 1.359 frames |
| Duração da imagem | 56,625 s |
| Áudio | AAC estéreo, 48 kHz, 56,646 s |
| Deslocamento inicial | Vídeo começa ~0,021 s após áudio; finais praticamente alinhados |
| Decodificação completa | Sem erro reportado pelo FFmpeg |
| Tela preta | Nenhum trecho detectado com limiar de 0,1 s e pix_th=0,10 |
| Congelamento | Nenhum trecho detectado com limiar de 0,5 s e n=-50 dB |
| Flicker global | Não detectado pelo video_doctor |
| Cortes | 9 cortes detectados, compatíveis com a montagem de 10 planos |
| Alertas temporais restantes | 42 segmentos candidatos; não são 42 defeitos visualmente confirmados |

As tiras dos alertas em 2,08–2,21 s e 2,38–2,54 s mostram movimento e aparecimento de texto; não estabelecem “rosto derretendo”. Os alertas em torno de mãos e mangas podem corresponder a movimento/oclusão legítimos. Portanto, não recomendo executar os reparos automáticos sugeridos pelo detector em lote.

Há desvios entre duração pedida e produzida: o plano atual soma 1.266 frames, ou **52,75 s a 24 fps**, enquanto o vídeo soma **56,625 s** — 93 frames, cerca de **3,88 s**, a mais. A soma do campo `seconds` é 52,52 s, ligeiramente diferente da própria grade de frames planejada. A diferença entre frames planejados e produzidos é compatível com arredondamentos por motor; não demonstra descompasso labial e deve ser tratada pelo contrato de duração.

## Causas prováveis, separadas das observações

1. **Sem referência canônica no elenco:** `cast.json` registra `reference_image=null` para as duas personagens. A continuidade depende dos stills da própria corrida, que já variam.
2. **Referência de locação usada também em avaliação facial:** os stills iniciais de Mei-Li apontam `shot000_wide.png`, um plano de Xiao-Lan. Os scores desses pares não validam a identidade de Mei-Li; no plano 3 o score registrado é aproximadamente -0,032. A referência pode servir ao ambiente, mas não deve representar o rosto da outra personagem.
3. **Descrição incompleta no plano de duas pessoas:** o prompt de vídeo do plano 5 descreve a roupa de Mei-Li, mas não fixa separadamente a aparência de Xiao-Lan. Isso é uma hipótese plausível para as duas acabarem vestidas como a princesa.
4. **Sujeito/ação desalinhados:** no plano 4, subject e beat apontam protagonistas diferentes. Nos planos 6/7, a ação relacional é exigida, mas o resultado acaba isolando Xiao-Lan.
5. **Correção editorial insuficiente:** o relatório editorial existente marcou repetição entre 6/7. O plano 7 recebeu uma instrução genérica para fazer ação diferente, mas ainda carrega o mesmo beat de ajustar o adorno. O resultado final continua repetindo a ação errada.
6. **O fechamento já desapareceu antes da geração:** o texto enriquecido contém a reação ao guarda e o divã; `shot_list`/`shot_plan` terminam na dança. Não basta aumentar a duração do último clipe para recuperar esse conteúdo.

## Prioridade para uma nova versão

1. Aprovar duas referências de personagem e figurino claramente distintos; separar referência de rosto e referência de cenário.
2. Corrigir os beats e recuperar o desfecho ausente antes de gastar com novas gerações.
3. Refazer primeiro os planos 4–7 (22,08–49,83 s), onde a identidade e a interação mais se perdem. Os planos 0–3 também precisam revisão de continuidade/texto.
4. Validar texto e inteligibilidade de cada fala; só depois avaliar e corrigir sincronização labial.
5. Montar novamente e verificar continuidade nos cortes. Aplicar reparo localizado apenas a artefatos realmente confirmados visualmente.

## Evidências complementares

- [Visão geral 0–18 s](panorama_1.jpg), [20–38 s](panorama_2.jpg), [40–56 s](panorama_3.jpg).
- [Timeline dos dez clipes](timeline.json).
- [Medições FFmpeg](ffmpeg.txt).
- [Diagnóstico temporal](doctor.json) e [log](doctor.log).
- Imagens `plano_00.jpg` a `plano_09.jpg`: início/meio/fim de cada plano, extraídos do vídeo final.

Os quadros têm timestamps da amostragem; não são imagens geradas ou reconstruídas para ilustrar defeitos.
