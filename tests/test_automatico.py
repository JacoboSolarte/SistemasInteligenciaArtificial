"""Pruebas del modo automático: modelo universal, objetivos/decodificación, reglas de postproceso y seguidor."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferencia.automatico import (SeguidorEspacios, construir_estado_automatico, asignar_ids,  # noqa: E402
                                   postprocesar_automatico, preparar_entrada)
from modelos.detector_universal import (DetectorUniversal, PerdidaDetectorUniversal, construir_objetivo,  # noqa: E402
                                        decodificar)


def test_forma_salida_y_tamano_libre():
    m = DetectorUniversal().eval()
    s = m(torch.rand(1, 3, 256, 384))  # cualquier tamaño múltiplo de 32
    assert s["calor"].shape == (1, 3, 64, 96)
    assert s["tam"].shape == (1, 6, 64, 96) and s["desfase"].shape == (1, 2, 64, 96)


def test_objetivo_y_decodificacion_ida_y_vuelta():
    """Si la salida del modelo es exactamente el objetivo, decodificar debe devolver las cajas originales."""
    cajas = np.array([[20, 30, 80, 70, 0], [120, 100, 160, 180, 1], [200, 40, 240, 60, 2]], np.float32)
    calor, ind, cls, tam, des, msk = construir_objetivo(cajas, 256, 256)
    h = w = 64
    logit = torch.logit(torch.from_numpy(calor).clamp(1e-4, 1 - 1e-4))[None]
    tam_mapa = torch.zeros(1, 6, h * w)
    des_mapa = torch.zeros(1, 2, h * w)
    for k in range(int(msk.sum())):
        tam_mapa[0, 2 * cls[k]:2 * cls[k] + 2, ind[k]] = torch.from_numpy(tam[k])
        des_mapa[0, :, ind[k]] = torch.from_numpy(des[k])
    salida = {"calor": logit, "tam": tam_mapa.view(1, 6, h, w), "desfase": des_mapa.view(1, 2, h, w)}
    c, p, k = decodificar(salida, umbral=0.9)[0]
    assert len(c) == 3
    for caja in cajas:
        j = int(np.argmin(np.abs(c - caja[:4]).sum(1)))
        assert np.allclose(c[j], caja[:4], atol=1.0) and k[j] == caja[4]


def test_perdida_ignora_clases_no_etiquetadas():
    m = DetectorUniversal()
    x = torch.rand(2, 3, 128, 128)
    cajas = np.array([[10, 10, 40, 40, 0]], np.float32)
    obj = [torch.from_numpy(np.stack([o, o])) for o in construir_objetivo(cajas, 128, 128)]
    crit = PerdidaDetectorUniversal()
    s = m(x)
    l_todas, _ = crit(s, *obj, torch.tensor([[1., 1., 1.], [1., 1., 1.]]))
    l_solo_veh, _ = crit(s, *obj, torch.tensor([[0., 0., 1.], [0., 0., 1.]]))
    assert torch.isfinite(l_todas) and torch.isfinite(l_solo_veh) and l_solo_veh.item() != l_todas.item()


def test_preparar_entrada_multiplo_de_32():
    img, esc = preparar_entrada(np.zeros((637, 1600, 3), np.uint8), 800)
    assert img.shape[0] % 32 == 0 and img.shape[1] % 32 == 0 and abs(esc - 0.5) < 1e-6


def test_reglas_postproceso():
    cajas = np.array([[0, 0, 50, 100], [2, 2, 50, 100],     # libre y ocupado sobre el mismo espacio
                      [100, 0, 150, 100],                  # libre con un vehículo dentro
                      [105, 10, 145, 90],                  # vehículo dentro del espacio anterior
                      [300, 300, 340, 380]], np.float32)   # vehículo fuera de cualquier espacio
    punt = np.array([0.6, 0.9, 0.5, 0.8, 0.7], np.float32)
    clases = np.array([0, 1, 0, 2, 2])
    espacios, vehiculos = postprocesar_automatico(cajas, punt, clases, 0.3, 0.5)
    assert len(espacios) == 2                             # NMS cruzado eliminó el libre duplicado
    estados = sorted(e["estado"] for e in espacios)
    assert estados == ["ocupado", "ocupado"]               # el libre con vehículo se corrigió
    assert sum(e["corregido"] for e in espacios) == 1
    assert sum(not v["en_espacio"] for v in vehiculos) == 1
    est = construir_estado_automatico(asignar_ids(espacios), vehiculos, 0.75)
    assert est["total"] == 2 and est["ocupados"] == 2 and est["vehiculos_fuera_de_espacio"] == 1


def test_seguidor_suaviza_cambios():
    seg = SeguidorEspacios(n_frames=2)
    caja = [0, 0, 50, 100]
    secuencia = ["libre", "ocupado", "libre", "ocupado", "ocupado"]
    salida = []
    for est in secuencia:
        e = [{"caja": list(caja), "estado": est, "confianza": 0.9, "corregido": False}]
        seg.actualizar(e)
        salida.append(e[0]["estado"])
    assert salida == ["libre", "libre", "libre", "libre", "ocupado"]
    assert e[0]["id_pista"] == "P1"


def test_regla_de_filas():
    from inferencia.filas import inferir_por_filas
    # fila horizontal de carros verticales (20x40) con paso 30; falta el carro de x=90 y los de x=180,210
    xs = [0, 30, 60, 120, 150, 240]
    vehiculos = [{"caja": [x, 0, x + 20, 40], "confianza": 0.9, "en_espacio": False} for x in xs]
    vehiculos.append({"caja": [500, 300, 520, 340], "confianza": 0.9, "en_espacio": False})  # carro suelto
    esp = inferir_por_filas([], vehiculos)
    ocupados = [e for e in esp if e["estado"] == "ocupado"]
    libres = [e for e in esp if e["estado"] == "libre"]
    assert len(ocupados) == 6                       # los 6 carros de la fila, no el suelto
    assert len(libres) == 3                         # 1 hueco en x=90 + 2 huecos en x=180 y 210
    centros = sorted(round((e["caja"][0] + e["caja"][2]) / 2) for e in libres)
    assert centros == [100, 190, 220]
    assert vehiculos[-1]["en_espacio"] is False
