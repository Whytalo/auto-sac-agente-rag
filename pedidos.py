"""Pedidos fictícios (mock) usados pela ferramenta consultar_pedido. Não faz chamadas de API."""

PEDIDOS = {
    "111.111.111-11": {"id": "PED-1001", "produto": "Fone Bluetooth ProMax",
                       "valor": "R$ 349,90", "status": "Entregue", "dias_atraso": 0},
    "222.222.222-22": {"id": "PED-1002", "produto": "Smart TV 50\" UltraHD",
                       "valor": "R$ 2.499,00", "status": "Atrasado", "dias_atraso": 8},
    "333.333.333-33": {"id": "PED-1003", "produto": "Cadeira Gamer ThroneX",
                       "valor": "R$ 1.299,00", "status": "Em trânsito", "dias_atraso": 0},
}

NAO_ENCONTRADO = "Nenhum pedido encontrado para este CPF."


def buscar_pedido(cpf: str):
    """Devolve o dicionário do pedido ou a frase literal de não encontrado."""
    return PEDIDOS.get(cpf.strip(), NAO_ENCONTRADO)
