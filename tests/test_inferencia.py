"""Pruebas de IoU, NMS, fusión, recorte, JSON de espacios, porcentaje y suavizado temporal."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluacion.metricas_detector import average_precision  # noqa: E402
from inferencia.fusion import fusionar, iou_poligono_caja  # noqa: E402
from inferencia.nms import iou_cajas, nms  # noqa: E402
from inferencia.pipeline import SuavizadorTemporal, construir_estado  # noqa: E402
from inferencia.recorte import cargar_espacios, cuatro_vertices, recortar_espacio  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


def test_iou_casos_basicos():
    assert iou_cajas([0, 0, 10, 10], [[0, 0, 10, 10]])[0] == pytest.approx(1.0)
    assert iou_cajas([0, 0, 10, 10], [[20, 20, 30, 30]])[0] == 0.0
    assert iou_cajas([0, 0, 10, 10], [[5, 0, 15, 10]])[0] == pytest.approx(50 / 150)


def test_nms_elimina_solapadas():
    cajas = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 9, 10]])
    puntajes = np.array([0.9, 0.8, 0.7, 0.95])
    conservar = nms(cajas, puntajes, 0.5)
    assert conservar == [3, 2]


def test_nms_vacio():
    assert nms(np.zeros((0, 4)), np.zeros(0)) == []


def test_average_precision_perfecta_y_nula():
    gt = [np.array([[0, 0, 10, 10], [20, 20, 30, 30]])]
    ap, _, _ = average_precision([(gt[0], np.array([0.9, 0.8]))], gt)
    assert ap == pytest.approx(1.0)
    ap0, _, _ = average_precision([(np.array([[100, 100, 110, 110]]), np.array([0.9]))], gt)
    assert ap0 == 0.0


def test_iou_poligono_caja():
    pol = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert iou_poligono_caja(pol, [0, 0, 10, 10]) == pytest.approx(1.0, abs=1e-4)
    assert iou_poligono_caja(pol, [20, 20, 30, 30]) == 0.0


def test_fusion_regla():
    esp = [{"id": "A1", "poligono": [[0, 0], [10, 0], [10, 10], [0, 10]]},
           {"id": "A2", "poligono": [[20, 0], [30, 0], [30, 10], [20, 10]]}]
    # A1: la CNN está segura (0.97 ocupado). A2: la CNN duda (0.55) y el detector ve un carro encima.
    r = fusionar(esp, [0.97, 0.55], cajas_detector=np.array([[20, 0, 30, 10]]), umbral_cnn=0.8, umbral_iou=0.3)
    assert r[0]["estado"] == "ocupado" and r[0]["fuente"] == "cnn" and r[0]["discrepancia"]
    assert r[1]["estado"] == "ocupado" and r[1]["fuente"] == "detector"
    r2 = fusionar(esp, [0.05, 0.4], cajas_detector=None)
    assert [x["estado"] for x in r2] == ["libre", "libre"] and r2[1]["fuente"] == "cnn_dudosa"


def test_calculo_porcentaje():
    por = [{"id": f"E{i}", "estado": "ocupado" if i < 3 else "libre", "confianza": 0.9 if i else 0.6} for i in range(4)]
    e = construir_estado(por, umbral_baja=0.75)
    assert (e["total"], e["ocupados"], e["libres"]) == (4, 3, 1)
    assert e["porcentaje_ocupacion"] == 75.0
    assert e["espacios_libres"] == ["E3"] and e["espacios_baja_confianza"] == ["E0"]
    json.dumps(e)   # serializable


def test_suavizado_temporal():
    s = SuavizadorTemporal(3)
    estados = []
    for crudo in ["libre", "ocupado", "ocupado", "libre", "ocupado", "ocupado", "ocupado"]:
        estados.append(s.actualizar([{"id": "A1", "estado": crudo}])[0]["estado"])
    assert estados == ["libre", "libre", "libre", "libre", "libre", "libre", "ocupado"]


def test_recorte_perspectiva():
    img = np.zeros((100, 100, 3), np.uint8)
    img[20:40, 30:60] = 255
    r = recortar_espacio(img, [[30, 20], [59, 20], [59, 39], [30, 39]], 64)
    assert r.shape == (64, 64, 3) and r.mean() > 240
    v = cuatro_vertices([[10, 0], [0, 10], [10, 20], [20, 10]])   # rombo
    assert len({tuple(p) for p in v.tolist()}) == 4


def test_carga_json_espacios(tmp_path):
    datos = [{"id": "A1", "poligono": [[0, 0], [10, 0], [10, 10], [0, 10]]}]
    p = tmp_path / "e.json"
    p.write_text(json.dumps(datos))
    assert cargar_espacios(p)[0]["id"] == "A1"
    p.write_text(json.dumps([{"id": "A1", "poligono": [[0, 0]]}]))
    with pytest.raises(ValueError):
        cargar_espacios(p)
    p.write_text(json.dumps(datos * 2))
    with pytest.raises(ValueError):
        cargar_espacios(p)


def test_json_espacios_del_proyecto_validos():
    for archivo in (RAIZ / "config").glob("espacios_*.json"):
        assert len(cargar_espacios(archivo)) > 0


def test_marcador_modo_fila_divide_y_deshace():
    import cv2
    import numpy as np
    from herramientas.marcar_espacios import Marcador, dividir_fila
    polis = dividir_fila([[0, 0], [100, 0], [100, 20], [0, 20]], 4)
    assert len(polis) == 4 and polis[0] == [[0, 0], [25, 0], [25, 20], [0, 20]] and polis[3][1] == [100, 0]
    m = Marcador(np.zeros((50, 120, 3), np.uint8))
    m.cambiar_modo()
    for x, y in [(0, 0), (100, 0), (100, 20), (0, 20)]:
        m.clic(cv2.EVENT_LBUTTONDOWN, x, y)
    assert m.espacios == []  # en modo FILA no se cierra con 4 clics
    for t in (ord("1"), ord("0"), 13):
        m.tecla_fila(t)
    assert len(m.espacios) == 10
    m.deshacer()  # deshace la fila completa
    assert m.espacios == []
