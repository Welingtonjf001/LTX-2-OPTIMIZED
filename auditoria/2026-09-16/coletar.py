"""Coleta estática reproduzível; não importa nem executa scripts auditados."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXT = {'.py', '.ps1', '.bat', '.cmd', '.sh', '.js', '.ts', '.dsa'}
paths = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True, encoding='utf-8').splitlines())
paths.update(subprocess.check_output(['rg', '--files'], cwd=ROOT, text=True, encoding='utf-8').splitlines())
rows = []
for rel in sorted({p.replace('\\', '/') for p in paths}):
    p = ROOT / rel
    if p.suffix.lower() not in EXT or rel.startswith('auditoria/') or not p.is_file():
        continue
    source = p.read_text(encoding='utf-8-sig', errors='replace')
    row = dict(path=rel, lines=len(source.splitlines()), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), signals={}, functions=[], purpose='')
    patterns = {'caminhos_fixos': r'[A-Za-z]:[\\/]', 'shell_true': r'shell\s*=\s*True',
                'rede_ampla': r'0\.0\.0\.0|share\s*=\s*True',
                'remocao': r'shutil\.rmtree|Remove-Item|\bos\.remove\(|\.unlink\(',
                'processos': r'subprocess\.|taskkill|Stop-Process',
                'excecao_ampla': r'except\s*(?:Exception(?:\s+as\s+\w+)?|BaseException)?\s*:'}
    for name, pattern in patterns.items():
        hits = [i for i, line in enumerate(source.splitlines(), 1) if re.search(pattern, line) and not line.lstrip().startswith(('#', 'REM ', '::', '//'))]
        if hits:
            row['signals'][name] = hits
    if p.suffix == '.py':
        try:
            tree = ast.parse(source, filename=rel)
            compile(tree, rel, 'exec')
            row['syntax'] = 'OK (Python 3.12)'
            doc = ast.get_docstring(tree) or ''
            row['purpose'] = ' '.join(doc.split())[:240]
            row['functions'] = [{'name': n.name, 'line': n.lineno, 'size': n.end_lineno-n.lineno+1} for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            row['subprocess_without_timeout'] = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'subprocess' and n.func.attr in ('run','check_output','check_call','call') and not any(k.arg == 'timeout' for k in n.keywords)]
        except (SyntaxError, ValueError) as exc:
            row['syntax'] = str(exc)
    else:
        row['syntax'] = 'Inspeção textual; não executado'
    rows.append(row)
(OUT / 'inventario.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'scripts':len(rows), 'linhas':sum(r['lines'] for r in rows), 'por_extensao':{e:sum(Path(r['path']).suffix==e for r in rows) for e in sorted(EXT)}, 'erros_python':[{'path':r['path'],'syntax':r['syntax']} for r in rows if r['path'].endswith('.py') and not r['syntax'].startswith('OK')], 'maiores':sorted([(r['lines'],r['path']) for r in rows], reverse=True)[:15]}, ensure_ascii=False, indent=2))
