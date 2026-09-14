"""Primeiro teste da tese em VÍDEO REAL: um propagador compacto sobre latentes do
VAE do LTX 2.5 (decoder congelado) prevê a evolução da cena melhor com estrutura
hamiltoniana do que sem?

Dados: latentes de `extract_latents.py`, clipes do LTX 2.5 deste repositório (os
"vídeos finais do professor" da reanálise). Formato medido: 128 canais, grade
17x30, um passo latente = 8 quadros (1/3 s). O 1º quadro latente (causal) já foi
descartado na extração.

Estado: q = latente atual, v = latente atual - latente anterior. Dois passos
reais de contexto; o modelo prevê os seguintes, autorregressivo.

Modelos (convolucionais sobre a grade, parâmetros pareados, SiLU por ser C²):
  baseline          conv residual sobre (q, p): p' = p + dp, q' = q + p' + dq
  hamiltoniano      V(q) = soma espacial de um campo escalar convolucional; força
                    -∇V por autograd (operador local, como uma EDP aprendida);
                    massa por canal; Verlet.
  port_hamiltoniano o anterior + amortecimento por canal (splitting de Strang).

GANHO DE INÉRCIA APRENDIDO, igual nos três: p0 = α·v (α·v·m no hamiltoniano),
α = sigmoid(·) iniciado em 0,1. MEDIDO no smoke de 2026-09-14: nestes latentes o
piso CONGELADO (MSE 1,08) é 4x melhor que a extrapolação LINEAR (4,04, que vai
de 0,98 a 7,87 em 4 passos). A 1/3 s por passo, a diferença entre latentes não
se comporta como movimento persistente. A versão anterior dava a todos o atalho
inercial completo (α = 1), o que fazia cada modelo começar no PIOR piso. Com α
aprendido, todos começam perto do melhor piso e aprendem quanto do movimento
observado persiste.

Pisos: congelado (repete o último latente) e linear (q + k·v).
Divisão treino/teste POR CORRIDA de origem, escolhendo corridas inteiras até ~20%
das sequências (a regra anterior, "1 corrida em cada N", deu 4 seqs de treino
contra 36 de teste no smoke, porque as corridas têm tamanhos muito desiguais).
Métrica: MSE do latente normalizado por passo de rollout; escalar principal =
média nos passos 1..K. Comparação só para quem bate o melhor piso, com teste
pareado por semente (troca de sinal, validado nos experimentos de brinquedo).

O que isto NÃO mede: qualidade do vídeo decodificado. Os latentes previstos de 2
sequências de teste por modelo são salvos para `decode_predictions.py` (régua:
31,2 dB na ida e volta do VAE).
"""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_dissipative import p_troca_sinal

ALFA0 = 0.1


# ----------------------------------------------------------------- dados ----

def carregar(pasta: Path, min_passos: int, limite: int | None):
    regs = [json.loads(l) for l in (pasta / "manifesto.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    seqs = []
    for r in regs:
        if r["passos_dinamica"] < min_passos:
            continue
        d = torch.load(pasta / r["arquivo"], map_location="cpu")
        seqs.append({"grupo": r["grupo"], "rel": r["rel"], "z": d["sequencia"].float()})  # [C,T,H,W]
        if limite and len(seqs) >= limite:
            break
    return seqs


LIMIAR_FAMILIA = 0.8


def familias(seqs, limiar: float = LIMIAR_FAMILIA):
    """Une em FAMÍLIAS as corridas com conteúdo quase igual (união-busca sobre a
    similaridade de cosseno do 1º latente de dinâmica, reduzido a 4x8).

    Dividir só por corrida vazava. MEDIDO 2026-09-14: cinco corridas
    20260913_* (msr_g03, msr_g05, distilled_close, close_v2, webui_loras) são o
    MESMO plano de 6 tomadas, com similaridade 1,000 entre corridas; e
    palacio_esmeralda_v6_ltx x teste_decupagem_w4a8 dão 0,989 (stills
    compartilhados). Referência: mediana 0,37 dentro da mesma corrida, 0,19
    entre corridas. O limiar 0,8 é conservador: une também as duas
    20260907_ltx_* (0,856, mesmo roteiro, não cópia) e _lora_ic_test_palacio
    (0,861)."""
    grupos = sorted({s["grupo"] for s in seqs})
    pai = {g: g for g in grupos}

    def acha(g):
        while pai[g] != g:
            pai[g] = pai[pai[g]]
            g = pai[g]
        return g

    desc = []
    for s in seqs:
        d = F.adaptive_avg_pool2d(s["z"][:, 0].unsqueeze(0), (4, 8)).flatten()
        desc.append(d / d.norm().clamp_min(1e-12))
    sim = torch.stack(desc) @ torch.stack(desc).T
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            if seqs[i]["grupo"] != seqs[j]["grupo"] and sim[i, j] > limiar:
                pai[acha(seqs[i]["grupo"])] = acha(seqs[j]["grupo"])
    membros = {}
    for g in grupos:
        membros.setdefault(acha(g), []).append(g)
    nome = {raiz: "+".join(sorted(ms)) for raiz, ms in membros.items()}
    for s in seqs:
        s["familia"] = nome[acha(s["grupo"])]


def dividir(seqs, frac_teste: float, fold: int | None = None, n_folds: int = 0):
    """FAMÍLIAS inteiras (ver `familias`) para teste.

    Modo fração (n_folds=0): famílias até ~frac_teste das sequências, sem
    estourar 1,5x a fração, em ordem determinística. Modo validação cruzada
    (n_folds>0): famílias distribuídas em n_folds dobras balanceadas por número
    de sequências (maior primeiro, sempre na dobra mais vazia); teste = dobra
    `fold`. Cada família passa pelo teste exatamente uma vez.

    BUGFIX 2026-09-14: a versão anterior montava treino/teste comparando
    s["grupo"] (nome da CORRIDA) com o nome da FAMÍLIA. Para família de uma
    corrida só os nomes coincidem; para família unida ("a+b+c") nunca, e as
    sequências dela iam para o TREINO mesmo com a família escolhida para teste.
    Na corrida dinamica_v1 o teste ficou com 13 sequências de 3 famílias
    (beatriz_lucas, lyra, teste_qualidade_20260912), não as 4 impressas. Sem
    vazamento: todas as famílias unidas ficaram no treino."""
    familias(seqs)
    por_fam = {}
    for s in seqs:
        por_fam.setdefault(s["familia"], []).append(s)
    fams = sorted(por_fam)
    if n_folds and n_folds > 0:
        dobras = [[] for _ in range(n_folds)]
        carga = [0] * n_folds
        for f in sorted(fams, key=lambda f: (-len(por_fam[f]), f)):
            k = min(range(n_folds), key=lambda i: (carga[i], i))
            dobras[k].append(f)
            carga[k] += len(por_fam[f])
        teste_f = set(dobras[fold])
    else:
        random.Random(1234).shuffle(fams)
        alvo = frac_teste * len(seqs)
        escolhidas, n = [], 0
        for f in fams:
            if n >= alvo:
                break
            if n + len(por_fam[f]) <= 1.5 * alvo:
                escolhidas.append(f)
                n += len(por_fam[f])
        if not escolhidas:   # toda família é grande demais: fica a menor
            escolhidas = [min(fams, key=lambda f: len(por_fam[f]))]
        teste_f = set(escolhidas)
    treino = [s for s in seqs if s["familia"] not in teste_f]
    teste = [s for s in seqs if s["familia"] in teste_f]
    return treino, teste, sorted(teste_f), sorted(set(por_fam) - teste_f)


def estatisticas(seqs):
    soma = soma2 = None
    n = 0
    for s in seqs:
        z = s["z"].double()
        zz = z.reshape(z.shape[0], -1)
        soma = zz.sum(1) if soma is None else soma + zz.sum(1)
        soma2 = (zz ** 2).sum(1) if soma2 is None else soma2 + (zz ** 2).sum(1)
        n += zz.shape[1]
    media = soma / n
    desvio = (soma2 / n - media ** 2).clamp_min(1e-12).sqrt()
    return media.float().view(-1, 1, 1, 1), desvio.float().view(-1, 1, 1, 1)


def amostrar_janelas(seqs, lote: int, tam: int, rng: random.Random, dev):
    pesos = [max(0, s["z"].shape[1] - tam + 1) for s in seqs]
    escolhidas = rng.choices(seqs, weights=pesos, k=lote)
    jan = []
    for s in escolhidas:
        ini = rng.randrange(0, s["z"].shape[1] - tam + 1)
        jan.append(s["z"][:, ini:ini + tam])
    return torch.stack(jan).to(dev)  # [B,C,tam,H,W]


# --------------------------------------------------------------- modelos ----

def conv_mlp(cin: int, h: int, cout: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.Conv2d(cin, h, 3, padding=1), nn.SiLU(),
        nn.Conv2d(h, h, 3, padding=1), nn.SiLU(),
        nn.Conv2d(h, cout, 3, padding=1))
    with torch.no_grad():   # começa perto de "sem correção"
        net[-1].weight.mul_(0.1)
        net[-1].bias.zero_()
    return net


class GanhoInercia(nn.Module):
    """α = sigmoid(raw), iniciado em ALFA0: quanto da velocidade observada persiste."""

    def __init__(self):
        super().__init__()
        self.raw_alfa = nn.Parameter(torch.tensor(math.log(ALFA0 / (1 - ALFA0))))

    @property
    def alfa(self):
        return torch.sigmoid(self.raw_alfa)


class Baseline(GanhoInercia):
    def __init__(self, c: int, h: int):
        super().__init__()
        self.f = conv_mlp(2 * c, h, 2 * c)

    def momento_inicial(self, v):
        return self.alfa * v

    def step(self, q, p):
        dq, dp = self.f(torch.cat([q, p], 1)).chunk(2, 1)
        p = p + dp
        return q + p + dq, p


class Hamiltoniano(GanhoInercia):
    def __init__(self, c: int, h: int):
        super().__init__()
        self.g = conv_mlp(c, h, 1)
        self.raw_m = nn.Parameter(torch.full((1, c, 1, 1), 0.5413))  # softplus = 1,0

    @property
    def m(self):
        return F.softplus(self.raw_m) + 1e-3

    def momento_inicial(self, v):
        return self.alfa * v * self.m

    def forca(self, q):
        treino = torch.is_grad_enabled()   # ver BUGFIX 2026-09-14 em hamiltonian.py
        with torch.enable_grad():
            qg = q if q.requires_grad else q.detach().requires_grad_(True)
            (dv,) = torch.autograd.grad(self.g(qg).sum(), qg, create_graph=treino)
        return dv

    def step(self, q, p):
        p = p - 0.5 * self.forca(q)
        q = q + p / self.m
        p = p - 0.5 * self.forca(q)
        return q, p


class PortHamiltoniano(Hamiltoniano):
    def __init__(self, c: int, h: int):
        super().__init__(c, h)
        self.raw_d = nn.Parameter(torch.full((1, c, 1, 1), -3.0))

    def step(self, q, p):
        fator = torch.exp(-F.softplus(self.raw_d) / self.m * 0.5)
        q, p = super().step(q, p * fator)
        return q, p * fator


def n_params(m):
    return sum(t.numel() for t in m.parameters())


def hidden_pareado(fabrica, alvo: int, c: int, faixa=range(16, 512)):
    return min(faixa, key=lambda h: abs(n_params(fabrica(c, h)) - alvo))


def rollout(modelo, ctx, k: int):
    q = ctx[:, :, -1]
    p = modelo.momento_inicial(ctx[:, :, -1] - ctx[:, :, -2])
    saidas = []
    for _ in range(k):
        q, p = modelo.step(q, p)
        saidas.append(q)
    return torch.stack(saidas, 2)


def pisos(ctx, k: int):
    q, v = ctx[:, :, -1], ctx[:, :, -1] - ctx[:, :, -2]
    congelado = q.unsqueeze(2).expand(-1, -1, k, -1, -1)
    passos = torch.arange(1, k + 1, device=ctx.device, dtype=ctx.dtype).view(1, 1, k, 1, 1)
    linear = q.unsqueeze(2) + passos * v.unsqueeze(2)
    return {"congelado": congelado, "linear": linear}


# --------------------------------------------------------- avaliação ----

def curva_mse(prever, teste, k_eval: int, ctx_len: int, dev):
    """MSE por passo de rollout nas sequências de teste (média ponderada)."""
    soma = torch.zeros(k_eval, dtype=torch.float64)
    cont = torch.zeros(k_eval, dtype=torch.float64)
    for s in teste:
        z = s["zn"].to(dev).unsqueeze(0)          # [1,C,T,H,W], normalizado
        k = min(k_eval, z.shape[2] - ctx_len)
        if k < 1:
            continue
        pred = prever(z[:, :, :ctx_len], k)
        alvo = z[:, :, ctx_len:ctx_len + k]
        erro = ((pred - alvo) ** 2).mean(dim=(0, 1, 3, 4)).double().cpu()
        soma[:k] += erro
        cont[:k] += 1
    return (soma / cont.clamp_min(1)).tolist(), cont.tolist()


def media_ponderada(curva, cont):
    tot = sum(cont)
    return sum(c * n for c, n in zip(curva, cont)) / tot if tot else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latentes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ctx", type=int, default=2)
    ap.add_argument("--k-treino", type=int, default=3)
    ap.add_argument("--k-eval", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--h-baseline", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--frac-teste", type=float, default=0.2, help="fração das sequências em famílias de teste")
    ap.add_argument("--n-folds", type=int, default=0,
                    help=">0 liga validação cruzada por famílias (cada família testada uma vez)")
    ap.add_argument("--fold", type=int, default=0, help="dobra de teste, de 0 a n-folds-1")
    ap.add_argument("--limite-seqs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default="baseline,hamiltoniano,port_hamiltoniano")
    args = ap.parse_args()
    if args.n_folds and not 0 <= args.fold < args.n_folds:
        raise SystemExit(f"--fold {args.fold} fora de 0..{args.n_folds - 1}")

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seqs = carregar(Path(args.latentes), args.ctx + args.k_treino, args.limite_seqs)
    treino, teste, g_teste, g_treino = dividir(seqs, args.frac_teste, args.fold, args.n_folds)
    if not treino or not teste:
        raise SystemExit(f"divisão vazia: {len(treino)} treino, {len(teste)} teste")
    media, desvio = estatisticas(treino)
    for s in seqs:
        s["zn"] = (s["z"] - media) / desvio
    c = seqs[0]["z"].shape[0]
    modo = f"dobra {args.fold} de {args.n_folds}" if args.n_folds else f"fração {args.frac_teste}"
    print(f"dispositivo {dev} | {len(seqs)} sequências ({sum(s['z'].shape[1] for s in seqs)} passos) | "
          f"divisão por família ({modo}) | treino {len(treino)} seqs de {len(g_treino)} famílias | "
          f"teste {len(teste)} seqs de {len(g_teste)} famílias: {g_teste}")

    alvo = n_params(Baseline(c, args.h_baseline))
    fabricas = {
        "baseline": lambda: Baseline(c, args.h_baseline),
        "hamiltoniano": (lambda h=hidden_pareado(Hamiltoniano, alvo, c): Hamiltoniano(c, h)),
        "port_hamiltoniano": (lambda h=hidden_pareado(PortHamiltoniano, alvo, c): PortHamiltoniano(c, h)),
    }
    nomes = [n for n in args.only.split(",") if n in fabricas]

    with torch.no_grad():
        curvas_piso = {nome: curva_mse(lambda ctx, k, nome=nome: pisos(ctx, k)[nome], teste,
                                       args.k_eval, args.ctx, dev)
                       for nome in ("congelado", "linear")}
    escalar_piso = {n: media_ponderada(*cv) for n, cv in curvas_piso.items()}
    melhor_piso_passo = [min(curvas_piso["congelado"][0][i], curvas_piso["linear"][0][i])
                         for i in range(args.k_eval)]
    cont_ref = curvas_piso["congelado"][1]
    escalar_melhor_piso = media_ponderada(melhor_piso_passo, cont_ref)
    for n, v in escalar_piso.items():
        print(f"piso {n:>9}: MSE médio {v:.4f} | por passo {[round(x, 3) for x in curvas_piso[n][0]]}")
    print(f"melhor piso por passo: MSE médio {escalar_melhor_piso:.4f}\n")

    res = {n: [] for n in nomes}
    npar = {}
    for seed in range(args.seeds):
        for nome in nomes:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            m = fabricas[nome]().to(dev)
            npar[nome] = n_params(m)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.0)
            t0 = time.time()
            perda = float("nan")
            for it in range(args.iters):
                jan = amostrar_janelas(treino, args.lote, args.ctx + args.k_treino, rng, dev)
                jan = (jan - media.to(dev).unsqueeze(0)) / desvio.to(dev).unsqueeze(0)
                pred = rollout(m, jan[:, :, :args.ctx], args.k_treino)
                loss = F.mse_loss(pred, jan[:, :, args.ctx:])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                perda = loss.item()
                if not math.isfinite(perda):
                    break
            custo = time.time() - t0
            with torch.no_grad():
                curva, cont = curva_mse(lambda ctx, k: rollout(m, ctx, k), teste, args.k_eval, args.ctx, dev)
            esc = media_ponderada(curva, cont)
            r = {"seed": seed, "perda_treino": perda, "mse_teste": esc, "curva": curva,
                 "alfa": float(m.alfa.detach()), "treino_s": custo}
            res[nome].append(r)
            print(f"  semente {seed} {nome:>17}: perda_treino={perda:.4f} mse_teste={esc:.4f} "
                  f"alfa={r['alfa']:.3f} curva={[round(x, 3) for x in curva]} ({custo:.0f}s)", flush=True)
            if seed == 0:   # latentes previstos de 2 sequências de teste, para decodificar depois
                with torch.no_grad():
                    for j, s in enumerate(teste[:2]):
                        z = s["zn"].to(dev).unsqueeze(0)
                        k = min(args.k_eval, z.shape[2] - args.ctx)
                        pred = rollout(m, z[:, :, :args.ctx], k)[0].cpu() * desvio + media
                        torch.save({"rel": s["rel"], "ctx": args.ctx, "previsto": pred.half(),
                                    "real": s["z"][:, :args.ctx + k].half()},
                                   out / f"previsto_{nome}_seq{j}.pt")

    (out / "resultados.json").write_text(json.dumps(
        {"args": vars(args), "grupos_teste": g_teste,
         "pisos": {n: {"curva": cv[0], "escalar": escalar_piso[n]} for n, cv in curvas_piso.items()},
         "melhor_piso_escalar": escalar_melhor_piso, "params": npar, "modelos": res},
        indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 104)
    print(f"{'modelo':>18} {'params':>9} {'MSE teste (passos 1..K)':>26} {'vs melhor piso':>15} "
          f"{'alfa':>6} {'treino':>8}")
    print("-" * 104)
    resumo = {}
    for nome in nomes:
        vals = [r["mse_teste"] for r in res[nome]]
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        resumo[nome] = mu
        t = statistics.mean(r["treino_s"] for r in res[nome])
        a = statistics.mean(r["alfa"] for r in res[nome])
        print(f"{nome:>18} {npar[nome]:>9,} {mu:>17.4f} ±{sd:<7.4f} {mu / escalar_melhor_piso:>14.2f}x "
              f"{a:>6.3f} {t:>7.0f}s")
    print("-" * 104)
    for n, v in escalar_piso.items():
        print(f"{'piso ' + n:>18} {'—':>9} {v:>17.4f}")
    print("=" * 104)

    print("\nLeitura:")
    reprovados = {n for n in nomes if resumo[n] >= 0.9 * escalar_melhor_piso}
    for n in sorted(reprovados):
        print(f"  {n}: não bate o melhor piso com folga ({resumo[n]:.4f} vs {escalar_melhor_piso:.4f}) "
              f"— não aprendeu dinâmica útil além de repetir ou extrapolar.")

    def compara(a, b):
        if a not in res or b not in res:
            return
        if a in reprovados or b in reprovados:
            print(f"  {a} vs {b}: comparação suprimida (ao menos um não bate o piso).")
            return
        d = [rb["mse_teste"] - ra["mse_teste"] for ra, rb in zip(res[a], res[b])]  # >0: a melhor
        n = len(d)
        mu = sum(d) / n
        p = p_troca_sinal(d)
        venc = a if mu > 0 else b
        vit = sum(1 for x in d if x != 0 and (x > 0) == (mu > 0))
        if n < 6:
            print(f"  {a} vs {b}: {n} sementes não permitem p<0,05; diferença média {mu:+.4f}, p = {p:.3f}.")
        elif p < 0.05:
            print(f"  {a} vs {b}: {venc} tem MSE menor, {abs(mu):.4f} em média, "
                  f"{vit}/{n} sementes (troca de sinal p = {p:.4f}).")
        else:
            print(f"  {a} vs {b}: sem efeito detectável (diferença {mu:+.4f}, {vit}/{n}, p = {p:.3f}).")

    compara("hamiltoniano", "baseline")
    compara("port_hamiltoniano", "baseline")
    compara("port_hamiltoniano", "hamiltoniano")


if __name__ == "__main__":
    main()
