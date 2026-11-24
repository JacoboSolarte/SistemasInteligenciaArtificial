"""Pruebas de forma de salida de los 3 modelos y de la pérdida del detector."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelos.cnn_espacios import CNNEspacios, crear_cnn  # noqa: E402
from modelos.mini_detector import MiniDetector, PerdidaDetector, construir_objetivo, decodificar  # noqa: E402
from modelos.transformer_asistente import AtencionMultiCabeza, TransformerAsistente  # noqa: E402


def test_cnn_forma_salida():
    m = CNNEspacios()
    y = m(torch.rand(5, 3, 64, 64))
    assert y.shape == (5, 2)
    clase, conf, p = m.predecir(torch.rand(3, 3, 64, 64))
    assert clase.shape == (3,) and torch.all((conf >= 0.5) & (conf <= 1)) and torch.all((p >= 0) & (p <= 1))


def test_cnn_configurable():
    m = crear_cnn({"filtros": [8, 16, 32], "neuronas_densa": 64, "dropout": 0.3})
    assert m(torch.rand(2, 3, 64, 64)).shape == (2, 2)


def test_detector_forma_salida():
    m = MiniDetector(cajas_por_celda=2, tam_imagen=416)
    s = m(torch.rand(2, 3, 416, 416))
    assert s.shape == (2, 13, 13, 2, 5)
    punt, cajas = decodificar(s)
    assert punt.shape == (2, 13 * 13 * 2) and cajas.shape == (2, 13 * 13 * 2, 4)
    assert torch.all(cajas >= 0) and torch.all(cajas <= 1)


def test_objetivo_y_perdida_detector():
    obj, desc = construir_objetivo([[[0.5, 0.5, 0.1, 0.2]], []], S=13, B=2)
    assert obj.shape == (2, 13, 13, 2, 5)
    assert obj[0, 6, 6, 0, 0] == 1 and obj[..., 0].sum() == 1 and desc == 0
    salida = torch.zeros(2, 13, 13, 2, 5, requires_grad=True)
    loss, partes = PerdidaDetector()(salida, obj)
    loss.backward()
    assert loss.item() > 0 and set(partes) == {"coord", "obj", "noobj"}


def test_transformer_forma_salida_y_mascara():
    m = TransformerAsistente(tam_vocab=50, num_intenciones=9, d_modelo=32, cabezas=4, capas=2, d_ff=64, max_tokens=10)
    ids = torch.randint(3, 50, (4, 11))
    ids[:, 0] = 2
    ids[:, 6:] = 0   # relleno
    assert m(ids).shape == (4, 9)
    att = m.capas[0].atencion.ultimos_pesos
    assert att.shape == (4, 4, 11, 11)
    assert torch.allclose(att.sum(-1), torch.ones(4, 4, 11), atol=1e-5)   # softmax suma 1
    assert torch.all(att[..., 6:] == 0)                                    # no se atiende al relleno


def test_atencion_a_mano_equivale_a_formula():
    torch.manual_seed(0)
    a = AtencionMultiCabeza(8, 1, dropout=0.0).eval()
    x = torch.rand(1, 3, 8)
    Q, K, V = a.W_q(x), a.W_k(x), a.W_v(x)
    esperado = a.W_o(torch.softmax(Q @ K.transpose(-2, -1) / 8 ** 0.5, -1) @ V)
    assert torch.allclose(a(x), esperado, atol=1e-6)
