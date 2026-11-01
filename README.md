# 🤖 Auto-SAC — Agente de Operações com RAG e Human-in-the-Loop

Agente autônomo de terminal para o suporte de um e-commerce fictício (**ECommerce Fictício Ltda.**). Ele conversa com o cliente, consulta as regras da empresa em uma base vetorial (RAG), verifica pedidos e resolve problemas — mas **nunca move dinheiro sem aprovação humana**.

Projeto Capstone da Trilha Tech, integrando em um único script Python os conceitos de APIs resilientes, RAG, engenharia de contexto e agentes com ferramentas.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991) ![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-orange)

---

## 📌 Sumário
1. [O que o agente faz](#-o-que-o-agente-faz)
2. [Arquitetura](#-arquitetura)
3. [Como cada curso entrou](#-como-cada-curso-entrou)
4. [As 3 ferramentas](#-as-3-ferramentas)
5. [Setup e execução](#-setup-e-execução)
6. [Estrutura do projeto](#-estrutura-do-projeto)
7. [Exemplos de output](#-exemplos-de-output)
8. [Testes obrigatórios](#-testes-obrigatórios)
9. [Decisões de projeto](#-decisões-de-projeto)
10. [Segurança](#-segurança)

---

## 🎯 O que o agente faz
- Responde dúvidas sobre **trocas, devoluções, garantia e frete** consultando uma base de conhecimento com 3 documentos.
- Consulta o **status de pedidos** pelo CPF (dados mockados).
- Processa **estornos**, mas só depois que um supervisor humano digitar `s` no terminal.
- Recusa educadamente assuntos **fora do escopo** (ex.: passagens aéreas), sem acionar ferramentas.
- Ao final do atendimento, gera um **Ticket de Resumo em JSON** com ação tomada, estorno, documentos consultados e resumo.

## 🏗 Arquitetura

```
                    ┌──────────────────────────────┐
  Cliente ────────▶ │  Ciclo ReAct (MAX_ITERACOES) │ ◀──── System prompt (tags XML)
                    └──────────────┬───────────────┘
                                   │ tool_calls
          ┌────────────────────────┼─────────────────────────┐
          ▼                        ▼                         ▼
 consultar_pedido     consultar_base_conhecimento      processar_estorno 🔴
 (mock, CPF)          (busca vetorial, n_results=3)             │
                                   │                   FUNCOES_SENSIVEIS
                                   ▼                            ▼
                      ChromaDB + text-embedding-3-small   ⚠ Human-in-the-loop
                      3 documentos · chunks 300/50         input() s/n
                      metadado {"fonte": ...}                   │
                                                                ▼
                                         resultado volta ao modelo como mensagem `tool`
                                                                │
                                                                ▼
                                         Ticket de Resumo (JSON Mode) ao final
```

**Fluxo do ciclo ReAct:** o modelo recebe o histórico → decide se chama ferramentas → o código executa cada `tool_call` (pedindo aprovação quando sensível) → devolve o resultado com o `tool_call_id` correspondente → repete até o modelo responder em texto, no máximo 15 iterações.

## 📚 Como cada curso entrou

| Curso | Técnica | Onde está no código (`auto_sac.py`) |
|---|---|---|
| **1 — APIs e Integração** | `@retry` do tenacity (`wait_random_exponential` + `stop_after_attempt`) em todas as portas de saída para a API; ticket em **JSON Mode** | `chamar_modelo`, `ingerir`, `consultar`, `gerar_ticket` |
| **2 — RAG** | `RecursiveCharacterTextSplitter(300, 50)`, ChromaDB com `text-embedding-3-small`, metadado `fonte`, busca com `n_results=3` | `preparar_base`, `consultar_base_conhecimento` |
| **3 — Contexto e Prompt** | System prompt com tags XML `<instrucoes>`, `<base_conhecimento>` e `<seguranca>` | `SYSTEM_PROMPT` |
| **4 — Agentes e Ferramentas** | Ciclo ReAct com `for i in range(MAX_ITERACOES)`, 3 ferramentas em JSON Schema, `FUNCOES_SENSIVEIS` e human-in-the-loop | `atender`, `TOOLS`, `FUNCOES_SENSIVEIS`, `pedir_aprovacao` |

## 🛠 As 3 ferramentas

| Ferramenta | Tipo | O que faz |
|---|---|---|
| `consultar_pedido(cpf)` | Inofensiva | Consulta status do pedido em um dicionário mockado. Se o CPF não existe, devolve `"Nenhum pedido encontrado para este CPF."` |
| `consultar_base_conhecimento(pergunta)` | Inofensiva (RAG) | Busca semântica nos 3 documentos e devolve os trechos com a fonte (`[politica_trocas] ...`). Registra as fontes para o ticket. |
| `processar_estorno(id_pedido)` | 🔴 **Sensível** | Estorna um pedido. **Só executa após aprovação humana.** |

As `description` de cada ferramenta dizem **quando usar e quando não usar**, o que evita consultas desnecessárias (ex.: "vocês vendem passagens?") e respostas de memória.

### Base de conhecimento (fictícia)
| Documento (`fonte`) | Conteúdo |
|---|---|
| `politica_trocas` | Arrependimento (7 dias), defeito (30 dias), estornos, pedido atrasado, itens não elegíveis, cancelamentos |
| `faq_produtos` | Garantia legal (90 dias), como acionar, mau uso, garantia estendida, produto divergente, vendas PJ |
| `condicoes_frete` | Modalidades, frete grátis, rastreamento, tentativas de entrega, frete especial, danos no transporte |

### Pedidos mockados
| CPF | Pedido | Status |
|---|---|---|
| `111.111.111-11` | PED-1001 · Fone Bluetooth ProMax | Entregue |
| `222.222.222-22` | PED-1002 · Smart TV 50" UltraHD | Atrasado (8 dias) |
| `333.333.333-33` | PED-1003 · Cadeira Gamer ThroneX | Em trânsito |

## ⚙️ Setup e execução

Requisitos: Python 3.10+ e uma chave da OpenAI.

```bash
git clone https://github.com/Whytalo/auto-sac-agente-rag.git
cd auto-sac-agente-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # edite e coloque sua OPENAI_API_KEY
```

Execução:

```bash
# Modo interativo (digite 'sair' para encerrar e gerar o ticket)
python auto_sac.py

# Modo de um turno: uma mensagem + ticket (usado nos testes)
python auto_sac.py "Quantos dias eu tenho para devolver um produto?"
```

Na primeira execução, os 3 documentos são fatiados e embedados em `chroma_data/`. Nas seguintes, a coleção é reaproveitada.

## 🗂 Estrutura do projeto

```
auto-sac-agente-rag/
├── auto_sac.py          # Agente: retry, RAG, ferramentas, ReAct, HITL, ticket
├── documentos.py        # Fixtures: politica_trocas, faq_produtos, condicoes_frete
├── pedidos.py           # PEDIDOS mockado e buscar_pedido
├── montar_relatorio.py  # Gera o relatório de entrega a partir do código e das saídas
├── saidas/              # Output real do terminal dos 7 testes
├── requirements.txt
├── .env.example         # Modelo do .env (o .env real nunca é commitado)
├── .gitignore           # .env, chroma_data/, __pycache__/, .venv/
└── chroma_data/         # Persistência do ChromaDB (gerada; ignorada pelo git)
```

## 💻 Exemplos de output

**Consulta RAG (Teste 1)**
```text
Cliente: Oi, comprei um produto e quero devolver. Quantos dias eu tenho?
  [ferramenta] consultar_base_conhecimento({"pergunta": "prazo de devolução por arrependimento"})

Agente: Conforme a Política de Trocas e Devoluções, caso seja por direito de arrependimento, você pode
desistir da compra em até 7 dias corridos após o recebimento do produto, sem precisar justificar...
```

**Estorno com Human-in-the-loop (Teste 2)**
```text
Cliente: Meu pedido atrasou demais. Meu CPF é 222.222.222-22. Quero meu dinheiro de volta.
  [ferramenta] consultar_pedido({"cpf": "222.222.222-22"})
  [ferramenta] consultar_base_conhecimento({"pergunta": "pedido atrasado cancelamento estorno"})
  [ferramenta] processar_estorno({"id_pedido": "PED-1002"})

⚠ ALERTA DE SEGURANÇA: A IA deseja processar estorno do pedido PED-1002.
Aprovar operação? (s/n): s
```

**Ticket de Resumo (JSON Mode)**
```json
{
  "id_pedido": "PED-1002",
  "cpf_cliente": "222.222.222-22",
  "status_pedido": "Atrasado",
  "acao_tomada": "Estorno processado com sucesso",
  "estorno_realizado": true,
  "valor_estorno": "R$ 2.499,00",
  "documentos_consultados": ["politica_trocas"],
  "resumo": "Cliente reclamou de atraso na entrega e solicitou devolução do dinheiro. Estorno aprovado pelo supervisor e processado."
}
```

## ✅ Testes obrigatórios

Os outputs completos estão em [`saidas/`](saidas/).

| # | Cenário | Esperado | Arquivo |
|---|---|---|---|
| 1 | Prazo de devolução | RAG na Política de Trocas; sem ferramenta sensível | `teste1.txt` |
| 2 | Pedido atrasado + estorno aprovado (`s`) | pedido → RAG → estorno com aprovação → ticket | `teste2.txt` |
| 3 | Mesmo cenário, estorno rejeitado (`n`) | Agente informa a negativa; ticket registra | `teste3.txt` |
| 4 | CPF inexistente | "Nenhum pedido encontrado"; pede conferência | `teste4.txt` |
| 5 | Fora do escopo (passagens aéreas) | Sem ferramentas | `teste5.txt` |
| 6 | Caixa amassada + defeito | RAG cruza `condicoes_frete` e `politica_trocas` | `teste6.txt` |
| 7 | Garantia e queda | FAQ: mau uso não coberto; reparo avulso | `teste7.txt` |

## 🧠 Decisões de projeto

- **Ingestão só se a coleção estiver vazia, com ids `f"{fonte}-{i}"`.** Evita pagar embeddings a cada execução e chunks duplicados ocupando os 3 resultados da busca (o que impediria o Teste 6 de cruzar documentos).
- **A aprovação humana fica no loop, antes do despacho.** Qualquer nova ferramenta adicionada a `FUNCOES_SENSIVEIS` já nasce protegida, e a rejeição volta ao modelo como resultado da tool, não como exceção.
- **O ticket lê o estado do sistema (`ESTADO`, `FONTES_CONSULTADAS`), não a memória do modelo.** `documentos_consultados` é sobrescrito com as fontes realmente consultadas.
- **Retry também na ingestão e na busca.** O ChromaDB chama a API de embeddings por dentro de `add` e `query`, então essas chamadas ficam em funções com `@retry`.
- **Encerramento da conversa:** o cliente digita `sair`, ou o atendimento termina após `MAX_TURNOS` (10).
- **Modelo `gpt-4o-mini`:** barato, rápido, com tool calling e JSON mode, suficiente para este fluxo. `temperature=0` para respostas consistentes.
- **Documentos fictícios** escritos a partir da descrição do enunciado, apenas para fins educacionais.

## 🔒 Segurança

- A `OPENAI_API_KEY` vive só no `.env`, que está no `.gitignore`. Nunca commite a chave.
- Estorno exige aprovação explícita (`s`); qualquer outra entrada conta como rejeição.
- Loop limitado a 15 iterações, para evitar chamadas infinitas e custo descontrolado.
- Todos os dados de clientes e pedidos são mockados.

## 👤 Autor

**Whytalo Rodrigues do Nascimento** — Web Designer e Desenvolvedor Full-Stack · [@Whytalo](https://github.com/Whytalo)
