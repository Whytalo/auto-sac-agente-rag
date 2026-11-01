"""Auto-SAC: agente de suporte com RAG, ciclo ReAct e human-in-the-loop para estornos."""
import json
import os
import sys

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from documentos import condicoes_frete, faq_produtos, politica_trocas
from pedidos import NAO_ENCONTRADO, PEDIDOS, buscar_pedido

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

MODELO_CHAT = "gpt-4o-mini"  # barato, rápido e suporta tool calling + JSON mode
MAX_ITERACOES = 15
MAX_TURNOS = 10

client = OpenAI(api_key=API_KEY)

# Política de retry única: espalha as tentativas e impede travar o terminal para sempre.
RETRY = dict(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))

# ---------------------------------------------------------------- Curso 1: chamadas protegidas


@retry(**RETRY)  # chat completions passam pela API: protegido contra falhas transitórias
def chamar_modelo(messages, tools=None, response_format=None):
    kwargs = {"model": MODELO_CHAT, "messages": messages, "temperature": 0}
    if tools:
        kwargs["tools"] = tools
    if response_format:
        kwargs["response_format"] = response_format
    return client.chat.completions.create(**kwargs)


@retry(**RETRY)  # collection.add chama a API de embeddings por dentro
def ingerir(collection, ids, documents, metadatas):
    collection.add(ids=ids, documents=documents, metadatas=metadatas)


@retry(**RETRY)  # collection.query embeda a pergunta via API
def consultar(collection, pergunta, n_results=3):
    return collection.query(query_texts=[pergunta], n_results=n_results)


# ---------------------------------------------------------------- Curso 2: RAG

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
embed = OpenAIEmbeddingFunction(api_key=API_KEY, model_name="text-embedding-3-small")
chroma = chromadb.PersistentClient(path="chroma_data")
collection = chroma.get_or_create_collection(name="base_conhecimento", embedding_function=embed)


def preparar_base():
    """Ingere os 3 documentos só se a coleção estiver vazia (ids determinísticos f'{fonte}-{i}')."""
    if collection.count() > 0:
        return
    documentos = {"politica_trocas": politica_trocas, "faq_produtos": faq_produtos,
                  "condicoes_frete": condicoes_frete}
    for fonte, texto in documentos.items():
        chunks = splitter.split_text(texto)
        ingerir(collection, ids=[f"{fonte}-{i}" for i in range(len(chunks))],
                documents=chunks, metadatas=[{"fonte": fonte}] * len(chunks))


# Estado do atendimento: o ticket lê daqui, não da memória do modelo.
FONTES_CONSULTADAS: set[str] = set()
ESTADO = {"id_pedido": None, "cpf_cliente": None, "status_pedido": None, "valor": None,
          "estorno_realizado": False, "estorno_negado": False}


def consultar_base_conhecimento(pergunta: str) -> str:
    r = consultar(collection, pergunta, n_results=3)
    trechos = []
    for doc, meta in zip(r["documents"][0], r["metadatas"][0]):
        FONTES_CONSULTADAS.add(meta["fonte"])
        trechos.append(f"[{meta['fonte']}] {doc}")
    return "\n\n".join(trechos)


# ---------------------------------------------------------------- Curso 4: ferramentas


def consultar_pedido(cpf: str) -> str:
    pedido = buscar_pedido(cpf)
    if pedido == NAO_ENCONTRADO:
        return NAO_ENCONTRADO
    ESTADO.update(id_pedido=pedido["id"], cpf_cliente=cpf.strip(),
                  status_pedido=pedido["status"], valor=pedido["valor"])
    return json.dumps(pedido, ensure_ascii=False)


def processar_estorno(id_pedido: str) -> str:
    for pedido in PEDIDOS.values():
        if pedido["id"] == id_pedido:
            ESTADO["estorno_realizado"] = True
            return f"Estorno do pedido {id_pedido} processado com sucesso. Valor: {pedido['valor']}."
    return f"Pedido {id_pedido} não encontrado. Nenhum estorno realizado."


TOOLS = [
    {"type": "function", "function": {
        "name": "consultar_pedido",
        "description": ("Use apenas para consultar status de pedido a partir de um CPF informado "
                        "pelo cliente. Não use para dúvidas sobre política, prazo ou garantia, "
                        "e nunca invente um CPF."),
        "parameters": {"type": "object",
                       "properties": {"cpf": {"type": "string",
                                              "description": "CPF no formato 000.000.000-00, com pontos e traço."}},
                       "required": ["cpf"]}}},
    {"type": "function", "function": {
        "name": "consultar_base_conhecimento",
        "description": ("Use para buscar regras da empresa sobre trocas, devoluções, arrependimento, "
                        "estorno, cancelamento, garantia, defeito, frete, prazo de entrega e "
                        "rastreamento. Não use para saudações, para status de pedido nem para "
                        "assuntos fora do e-commerce."),
        "parameters": {"type": "object",
                       "properties": {"pergunta": {"type": "string",
                                                   "description": "Pergunta em linguagem natural sobre a regra procurada."}},
                       "required": ["pergunta"]}}},
    {"type": "function", "function": {
        "name": "processar_estorno",
        "description": ("OPERAÇÃO FINANCEIRA SENSÍVEL. Use somente depois de consultar o pedido e "
                        "confirmar na base de conhecimento que o cliente tem direito ao estorno. "
                        "Exige aprovação humana. Não use para simples dúvidas."),
        "parameters": {"type": "object",
                       "properties": {"id_pedido": {"type": "string",
                                                    "description": "ID do pedido, ex.: PED-1002, obtido em consultar_pedido."}},
                       "required": ["id_pedido"]}}},
]

FUNCOES_SENSIVEIS = {"processar_estorno"}
DESPACHO = {"consultar_pedido": consultar_pedido,
            "consultar_base_conhecimento": consultar_base_conhecimento,
            "processar_estorno": processar_estorno}

# ---------------------------------------------------------------- Curso 3: system prompt

SYSTEM_PROMPT = """
<instrucoes>
Você é o agente de suporte da ECommerce Fictício Ltda.
Responda ao cliente com empatia e objetividade.
Use as ferramentas disponíveis quando necessário.
Se a base de conhecimento não cobrir a dúvida, diga que vai encaminhar ao time de operações.
Nunca invente regras ou valores.
Ao citar informações da política, mencione a fonte (ex: "Conforme a Política de Trocas...").
</instrucoes>

<base_conhecimento>
Você tem acesso à ferramenta consultar_base_conhecimento para buscar regras em 3 documentos: a Política de Trocas e Devoluções, o FAQ de Produtos e Garantia, e as Condições de Frete e Entrega.
Use-a sempre que o cliente perguntar sobre prazos, trocas, garantia, frete ou devoluções.
Não responda de memória — sempre consulte a ferramenta para garantir precisão.
</base_conhecimento>

<seguranca>
Estorno é operação financeira e só acontece com aprovação humana do supervisor.
Se o resultado de processar_estorno disser que a operação não foi aprovada, informe o cliente com clareza e ofereça alternativas (por exemplo, encaminhar ao time de operações).
Perguntas fora do escopo de e-commerce (ex.: passagens aéreas): diga que está fora do escopo, sem usar ferramentas.
</seguranca>
"""

# ---------------------------------------------------------------- Human-in-the-loop


def pedir_aprovacao(nome, args):
    print(f"\n⚠ ALERTA DE SEGURANÇA: A IA deseja processar estorno do pedido {args.get('id_pedido')}.")
    return input("Aprovar operação? (s/n): ").strip().lower() == "s"


# ---------------------------------------------------------------- Curso 4: ciclo ReAct


def atender(messages):
    """Roda o ciclo ReAct até o modelo responder sem tool_calls (ou estourar MAX_ITERACOES)."""
    for i in range(MAX_ITERACOES):
        resp = chamar_modelo(messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content
        # assistant com tool_calls entra no histórico ANTES das mensagens de tool
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            nome = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"  [ferramenta] {nome}({json.dumps(args, ensure_ascii=False)})")
            if nome in FUNCOES_SENSIVEIS and not pedir_aprovacao(nome, args):
                ESTADO["estorno_negado"] = True
                resultado = ("Operação não aprovada pelo supervisor. "
                             "Informe o cliente e ofereça alternativas.")
            elif nome in DESPACHO:
                resultado = DESPACHO[nome](**args)
            else:
                resultado = f"Ferramenta desconhecida: {nome}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(resultado)})
    return "Não consegui concluir o atendimento; vou encaminhar ao time de operações."


# ---------------------------------------------------------------- Curso 1: ticket em JSON Mode


def gerar_ticket(messages):
    conversa = "\n".join(f"{m['role']}: {m['content']}" for m in messages
                         if m["role"] in ("user", "assistant") and m.get("content"))
    fatos = {**ESTADO, "documentos_consultados": sorted(FONTES_CONSULTADAS)}
    pedido = (
        "Gere o Ticket de Resumo do atendimento em JSON com exatamente as chaves: id_pedido, "
        "cpf_cliente, status_pedido, acao_tomada, estorno_realizado (bool), valor_estorno "
        "(string ou null), documentos_consultados (lista), resumo. Use estes fatos do sistema como "
        "verdade (null se não houver): " + json.dumps(fatos, ensure_ascii=False) +
        ". Se estorno_negado for true, acao_tomada deve registrar que o estorno foi negado e "
        "valor_estorno deve ser null. Só preencha valor_estorno se estorno_realizado for true.\n\n"
        "Conversa:\n" + conversa)
    resp = chamar_modelo([{"role": "system", "content": "Você gera tickets de suporte em JSON."},
                          {"role": "user", "content": pedido}],
                         response_format={"type": "json_object"})
    ticket = json.loads(resp.choices[0].message.content)
    ticket["documentos_consultados"] = sorted(FONTES_CONSULTADAS)  # não depende do modelo
    return ticket


def main():
    preparar_base()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Modo teste: python auto_sac.py "mensagem do cliente" -> um turno e ticket
    entradas = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else None
    for turno in range(MAX_TURNOS):
        if entradas is not None:
            if turno >= len(entradas):
                break
            texto = entradas[turno]
            print(f"\nCliente: {texto}")
        else:
            texto = input("\nCliente (digite 'sair' para encerrar): ").strip()
            if texto.lower() in ("sair", "exit", ""):
                break
        messages.append({"role": "user", "content": texto})
        print(f"\nAgente: {atender(messages)}")
    print("\n=== TICKET DE RESUMO ===")
    print(json.dumps(gerar_ticket(messages), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
