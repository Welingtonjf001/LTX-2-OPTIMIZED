"""Agrega a validação cruzada por famílias de `train_latent_dynamics.py`.

Lê <base>/cv_f*/resultados.json e compara dois modelos pareando por (dobra,
semente). A diferença é MSE(B) - MSE(A), então positivo quer dizer que A foi
melhor. Dois níveis de teste:
  - por par (dobra, semente): n = dobras x sementes, troca de sinal exata;
  - por dobra (média das sementes): n = dobras. Com 5 dobras o menor p possível
    é 0,0625, então esse nível só indica consistência entre conteúdos.

Regra de exclusão: uma dobra só sai da conta se NENHUM dos dois modelos bate o
melhor piso dela com folga (<0,9x). Se só um falha, a dobra entra, e a falha
conta como derrota dele.

BUGFIX 2026-09-14: a versão anterior excluía a dobra quando QUALQUER um dos dois
falhava. Isso é viés de seleção a favor de quem falhou. Na validação cruzada
dinamica, a dobra 2 saiu da conta porque o port_hamiltoniano não bateu o piso,
enquanto o baseline bateu: o que era derrota do port virou dado descartado. A
guarda contra falso positivo serve para não comparar dois modelos que não
aprenderam nada, e não para apagar a derrota de um deles.

Sem torch: roda em qualquer Python.
"""
import argparse
import json
import statistics
from pathlib import Path


def p_troca_sinal(d):
    n = len(d)
    alvo = abs(sum(d) / n) - 1e-12
    ext = 0
    for mask in range(1 << n):
        s = sum(-x if (mask >> i) & 1 else x for i, x in enumerate(d))
        ext += abs(s / n) >= alvo
    return ext / (1 << n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="pasta que contém cv_f0, cv_f1, ...")
    ap.add_argument("--a", default="port_hamiltoniano")
    ap.add_argument("--b", default="baseline")
    args = ap.parse_args()

    dobras = sorted(Path(args.base).glob("cv_f*/resultados.json"),
                    key=lambda p: int(p.parent.name.removeprefix("cv_f")))
    if not dobras:
        raise SystemExit(f"nenhum cv_f*/resultados.json em {args.base}")

    pares, por_dobra = [], []
    print(f"{'dobra':>5} {'piso':>7} {args.a:>18} {args.b:>10} {'melhora':>8} {'vitórias':>9}  famílias de teste")
    for arq in dobras:
        r = json.loads(arq.read_text(encoding="utf-8"))
        f = r["args"]["fold"]
        piso = r["melhor_piso_escalar"]
        ma = {x["seed"]: x["mse_teste"] for x in r["modelos"].get(args.a, [])}
        mb = {x["seed"]: x["mse_teste"] for x in r["modelos"].get(args.b, [])}
        sementes = sorted(set(ma) & set(mb))
        if not sementes:
            print(f"{f:>5}  (sem sementes em comum)")
            continue
        mua = statistics.mean(ma[s] for s in sementes)
        mub = statistics.mean(mb[s] for s in sementes)
        a_ok, b_ok = mua < 0.9 * piso, mub < 0.9 * piso
        entra = a_ok or b_ok
        d = [mb[s] - ma[s] for s in sementes]
        vit = sum(1 for x in d if x > 0)
        fams = ", ".join(x if len(x) < 40 else x[:37] + "..." for x in r.get("grupos_teste", []))
        if a_ok and b_ok:
            marca = ""
        elif entra:
            marca = f"  [{args.a if not a_ok else args.b} não bate o piso: conta como derrota dele]"
        else:
            marca = "  [EXCLUÍDA: nenhum dos dois bate o piso]"
        print(f"{f:>5} {piso:>7.3f} {mua:>18.4f} {mub:>10.4f} {(mub - mua) / mub:>+7.1%} "
              f"{vit:>4}/{len(d):<4}  {fams}{marca}")
        if entra:
            pares += d
            por_dobra.append(statistics.mean(d))

    print()
    if not pares:
        raise SystemExit("nenhuma dobra em que ao menos um modelo bata o piso")
    n = len(pares)
    vit = sum(1 for x in pares if x > 0)
    print(f"pares (dobra, semente): {n} | {args.a} melhor em {vit}/{n} | diferença média "
          f"{statistics.mean(pares):+.4f} | troca de sinal exata p = {p_troca_sinal(pares):.4f}")
    k = len(por_dobra)
    vit_d = sum(1 for x in por_dobra if x > 0)
    print(f"dobras (média das sementes): {k} | {args.a} melhor em {vit_d}/{k} | p = {p_troca_sinal(por_dobra):.4f} "
          f"(com {k} dobras o menor p possível é {2 / 2 ** k:.4f})")


if __name__ == "__main__":
    main()
