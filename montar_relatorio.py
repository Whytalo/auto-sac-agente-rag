"""Monta relatorio_entrega.md: README + código + saídas reais dos testes (pasta saidas/teste1.txt ... teste7.txt)."""
from pathlib import Path

TITULOS = {1: "Consulta RAG simples", 2: "Pedido + RAG + estorno aprovado", 3: "Estorno rejeitado",
           4: "CPF não encontrado", 5: "Fora do escopo", 6: "RAG cruzando documentos", 7: "Garantia (FAQ)"}
partes = ["# Relatório de Entrega — Auto-SAC\n", Path("README.md").read_text(encoding="utf-8")]
for arq in ("documentos.py", "pedidos.py", "auto_sac.py"):
    partes.append(f"\n## {arq}\n\n```python\n{Path(arq).read_text(encoding='utf-8')}\n```\n")
partes.append("\n## .gitignore\n\n```\n" + Path(".gitignore").read_text() + "```\n")
partes.append("\n## Testes obrigatórios (output real do terminal)\n")
for n, t in TITULOS.items():
    f = Path(f"saidas/teste{n}.txt")
    corpo = f.read_text(encoding="utf-8") if f.exists() else "(rode o teste e salve a saída aqui)"
    partes.append(f"\n### Teste {n} — {t}\n\n```text\n{corpo}\n```\n")
Path("relatorio_entrega.md").write_text("\n".join(partes), encoding="utf-8")
print("relatorio_entrega.md gerado")
