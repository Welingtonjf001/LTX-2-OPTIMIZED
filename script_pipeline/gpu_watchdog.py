"""Observador compartilhado para os servidores ComfyUI (LTX 2.3/2.5, MiniMax
H3) e para qualquer launcher (.bat/UI) que precise liberar a porta e a VRAM
antes de subir de novo.

POR QUE ISTO EXISTE

Toda trava real medida nesta maquina ate agora (MEMORIAL 3.24, 3.30, 3.33,
3.41, 3.58) tem a MESMA assinatura: um servidor ComfyUI fica com um job preso
na fila, GPU em ~100%/24 GB, e todo cliente que tenta gerar depois so descobre
isso depois de esperar o PROPRIO timeout (600-3600s) esgotar -- um caso chegou
a 7 HORAS porque 43 planos esperaram, cada um, o timeout inteiro atras de um
job que nunca ia terminar sozinho. O denominador comum: ninguem olhava a fila
ENQUANTO ela travava, so depois que o cliente desistia.

Este modulo poe alguem olhando a fila o tempo todo, num thread separado, para
detectar a trava em minutos (nao em horas) e reagir -- reiniciar o servidor
sozinho ate um limite, e depois disso PARAR de tentar e avisar com clareza, em
vez de reiniciar para sempre escondendo um problema mais profundo.

O QUE NAO FAZ

Nao mata processo nenhum sem ter certeza de que a porta e do tipo esperado
(ComfyUI/Gradio que ESTE modulo mesmo comecou a vigiar) -- `free_port` mata
por PORTA, entao só chame com a porta que voce mesmo controla. Nao tenta
recuperar-se de kill forcado (`taskkill /F`) vindo de FORA: se alguem mata o
processo que RODA este watchdog, o thread morre junto -- o unico jeito de
sobreviver a isso seria um processo supervisor separado, que nao existe aqui.
O que este modulo evita e precisar do kill forcado em primeiro lugar."""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


# --------------------------------------------------------------------------
# porta e GPU
# --------------------------------------------------------------------------
def free_port(port: int, *, log=print) -> bool:
    """Derruba quem estiver ESCUTANDO em `port` -- exceto o PROPRIO processo
    chamador (nunca se auto-mata; um servidor que por algum motivo ja se ve
    na porta que esta prestes a abrir nao deve tentar se derrubar). Devolve
    se derrubou algo.

    Mesmo idioma netstat+taskkill usado em `generate_storyboards.stop_comfyui`
    e em `decupagem_ui._free_port` -- consolidado aqui para as duas paradas de
    terem UMA implementacao, nao duas que podem divergir."""
    if sys.platform != "win32":
        log(f"[watchdog] free_port({port}): so implementado no Windows.")
        return False
    try:
        saida = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout or ""
    except Exception as e:
        log(f"[watchdog] free_port({port}): netstat falhou ({type(e).__name__}).")
        return False
    import os
    meu_pid = str(os.getpid())
    pids = set()
    for linha in saida.splitlines():
        if f":{port} " in linha and "LISTENING" in linha.upper():
            m = re.search(r"(\d+)\s*$", linha.strip())
            if m:
                pids.add(m.group(1))
    pids -= {meu_pid}
    if not pids:
        return False
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True, text=True)
    log(f"[watchdog] porta {port}: encerrado PID {', '.join(sorted(pids))}.")
    time.sleep(2)
    return True


def gpu_status(*, gpu_index: int | None = None) -> list[dict]:
    """[{index, name, used_mib, total_mib, util_pct}, ...] via nvidia-smi.
    Lista vazia se nvidia-smi nao existir/falhar -- nunca levanta excecao,
    porque isto e usado so para LOG, e um log que quebra o programa e pior
    que um log que falta."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return []
    except Exception:
        return []
    out = []
    for linha in r.stdout.strip().splitlines():
        partes = [p.strip() for p in linha.split(",")]
        if len(partes) != 5:
            continue
        idx, nome, usado, total, util = partes
        try:
            idx_i = int(idx)
        except ValueError:
            continue
        if gpu_index is not None and idx_i != gpu_index:
            continue
        out.append({"index": idx_i, "name": nome, "used_mib": int(usado),
                    "total_mib": int(total), "util_pct": int(util)})
    return out


def gpu_status_line(*, gpu_index: int | None = None) -> str:
    linhas = gpu_status(gpu_index=gpu_index)
    if not linhas:
        return "(nvidia-smi indisponivel)"
    return " | ".join(f"GPU{g['index']} {g['used_mib']}/{g['total_mib']} MiB ({g['util_pct']}%)"
                      for g in linhas)


# --------------------------------------------------------------------------
# lixo / produtos parciais
# --------------------------------------------------------------------------
# Sufixos/padroes conhecidos de arquivo que uma geracao interrompida deixa pra
# tras -- cada um documentado no ponto do codigo que o cria. Nunca apaga nada
# mais novo que `max_age_seconds`: um arquivo sendo escrito AGORA por uma
# corrida em andamento nao pode ser confundido com lixo de uma corrida morta.
GARBAGE_GLOBS = (
    "*.freeze_tmp.mp4",       # render_shots._apply_freeze, sobra se o ffmpeg falhar no meio
    "*.tentativa[0-9].png",   # render_shots._still_for_shot, tentativas da auditoria de consistencia
    "*.tentativa[0-9][0-9].png",
)


def sweep_garbage(run_dir: str | Path, *, max_age_seconds: int = 1800, log=print) -> int:
    """Apaga produtos parciais conhecidos dentro de `run_dir`, mais velhos que
    `max_age_seconds`. Devolve quantos arquivos apagou.

    Roda-se ANTES de retomar uma corrida (run_decupagem faz isto no inicio de
    cada estagio de imagem/video) -- nao durante, para nao correr atras de
    arquivo que outra parte do mesmo processo ainda esta escrevendo."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return 0
    agora = time.time()
    apagados = 0
    for padrao in GARBAGE_GLOBS:
        for p in run_dir.rglob(padrao):
            try:
                if agora - p.stat().st_mtime < max_age_seconds:
                    continue
                p.unlink()
                apagados += 1
            except OSError:
                continue
    # PNG/MP4 de tamanho zero: geracao que morreu no meio do write. Verificado
    # em qualquer lugar da corrida, nao so nos padroes acima -- e o sintoma
    # mais barato de detectar (tamanho, nao nome) e o mais enganoso de deixar
    # (o manifesto pode achar que "existe" e pular a regeneracao).
    for ext in ("*.png", "*.mp4", "*.wav"):
        for p in run_dir.rglob(ext):
            try:
                st = p.stat()
                if st.st_size == 0 and (agora - st.st_mtime) > max_age_seconds:
                    p.unlink()
                    apagados += 1
            except OSError:
                continue
    if apagados:
        log(f"[watchdog] sweep_garbage: {apagados} arquivo(s) parcial(is)/vazio(s) removido(s) em {run_dir}.")
    return apagados


# --------------------------------------------------------------------------
# vigia de trava na fila do ComfyUI
# --------------------------------------------------------------------------
class StallWatch:
    """Thread em segundo plano que confere `{server}/queue` a cada
    `poll_seconds` e reage se o MESMO job ficar "rodando" por mais de
    `stall_seconds` sem trocar.

    `auto_recover=True`: ao detectar trava, chama `free_port(port)` sozinho
    (mesma recuperacao que ja era feita NA MAO nesta sessao -- taskkill no
    processo do ComfyUI) e loga a acao. Depois de `max_auto_recoveries`
    trocas seguidas SEM progresso real (a fila trava nao so uma vez, mas
    repetidamente), PARA de tentar sozinho e so avisa -- trava que sobrevive
    a reinicio nao e trava de fila, e outro problema, e reiniciar para sempre
    so esconderia isso. `auto_recover=False`: nunca mata nada, so avisa (via
    `log` e `on_stall`) -- para quem prefere decidir na hora, por conta do
    pedido do usuario de poder aceitar ou nao a acao em casos mais graves."""

    def __init__(self, server: str, port: int, *, log=print,
                stall_seconds: int = 300, poll_seconds: int = 15,
                auto_recover: bool = False, max_auto_recoveries: int = 2,
                on_stall=None, on_exhausted=None):
        self.server = server
        self.port = port
        self.log = log
        self.stall_seconds = stall_seconds
        self.poll_seconds = poll_seconds
        self.auto_recover = auto_recover
        self.max_auto_recoveries = max_auto_recoveries
        self.on_stall = on_stall
        self.on_exhausted = on_exhausted
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recoveries = 0

    def start(self) -> "StallWatch":
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"stallwatch-{self.port}")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _queue_running_id(self):
        try:
            with urllib.request.urlopen(f"{self.server}/queue", timeout=10) as r:
                import json
                d = json.load(r)
        except Exception:
            return None, None
        running = d.get("queue_running") or []
        pending = len(d.get("queue_pending") or [])
        if not running:
            return None, pending
        return running[0][1], pending  # prompt_id do job em execucao

    def _run(self) -> None:
        job_atual = None
        desde = time.time()
        while not self._stop.wait(self.poll_seconds):
            pid, pendentes = self._queue_running_id()
            if pid is None:
                job_atual, desde = None, time.time()
                continue
            if pid != job_atual:
                job_atual, desde = pid, time.time()
                continue
            parado_ha = time.time() - desde
            if parado_ha < self.stall_seconds:
                continue

            gpu = gpu_status_line()
            self.log(f"[watchdog] ATENCAO: mesmo job ({pid}) rodando ha "
                     f"{parado_ha:.0f}s sem trocar, {pendentes} pendente(s) na fila. {gpu}")
            if callable(self.on_stall):
                try:
                    self.on_stall({"prompt_id": pid, "seconds": parado_ha,
                                   "pending": pendentes, "gpu": gpu})
                except Exception:
                    pass

            if not self.auto_recover:
                self.log("[watchdog] auto_recover desligado -- nao vou reiniciar sozinho. "
                         "Confirme com o usuario antes de agir.")
                desde = time.time()  # nao repete o aviso a cada poll, so a cada stall_seconds de novo
                continue

            if self._recoveries >= self.max_auto_recoveries:
                self.log(f"[watchdog] ja tentei recuperar {self._recoveries}x sem sucesso -- "
                         "desistindo de reiniciar sozinho. Provavelmente nao e trava de fila "
                         "(pode ser VRAM insuficiente pro job, ou bug no workflow); "
                         "intervencao manual necessaria.")
                if callable(self.on_exhausted):
                    try:
                        self.on_exhausted({"prompt_id": pid, "recoveries": self._recoveries})
                    except Exception:
                        pass
                self._stop.set()
                break

            self._recoveries += 1
            self.log(f"[watchdog] recuperando automaticamente (tentativa {self._recoveries}/"
                     f"{self.max_auto_recoveries}): encerrando o servidor na porta {self.port}.")
            free_port(self.port, log=self.log)
            job_atual, desde = None, time.time()


def start_stall_watch(server: str, port: int, **kwargs) -> StallWatch:
    """Atalho: cria e ja inicia um StallWatch."""
    return StallWatch(server, port, **kwargs).start()
