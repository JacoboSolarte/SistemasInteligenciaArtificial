"""Pruebas del tokenizador, del generador de dataset y de las respuestas del asistente."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asistente.generar_dataset import INTENCIONES, generar  # noqa: E402
from asistente.respuestas import responder  # noqa: E402
from asistente.tokenizador import CLS, PAD, Tokenizador, normalizar  # noqa: E402


def test_normalizar():
    assert normalizar("¿Hay PUESTOS disponibles, señor?") == "hay puestos disponibles senor"
    assert normalizar("  ¿Está   fúll?? ") == "esta full"


def test_tokenizador_subpalabras_y_relleno():
    tok = Tokenizador(max_tokens=8).construir(["hay puestos disponibles", "disponibilidad de puestos"] * 3)
    ids = tok.codificar("¿Hay disponibilidda?")
    assert len(ids) == 9 and ids[0] == CLS and ids[-1] == PAD
    piezas = tok.tokenizar("disponibilidda")         # palabra con error de tipeo -> subpalabras conocidas
    assert "[UNK]" not in piezas and len(piezas) > 1
    assert tok.tokenizar("hay puestos") == ["hay", "puestos"]


def test_tokenizador_guardar_cargar(tmp_path):
    tok = Tokenizador(max_tokens=6).construir(["hola mundo", "hola parce"] * 2)
    tok.guardar(tmp_path / "t.json")
    t2 = Tokenizador.cargar(tmp_path / "t.json")
    assert t2.codificar("hola parce") == tok.codificar("hola parce")


def test_dataset_minimo_150_por_intencion():
    df = generar(160, semilla=1)
    conteo = df.intencion.value_counts()
    assert set(conteo.index) == set(INTENCIONES)
    assert conteo.min() >= 150


def test_respuestas_usan_estado_real():
    estado = {"total": 40, "libres": 12, "ocupados": 28, "porcentaje_ocupacion": 70.0, "confianza_promedio": 0.93,
              "espacios_libres": ["A3", "A7", "B2"], "espacios_baja_confianza": ["C4", "C5"], "discrepancias": [],
              "usar_detector": False, "vehiculos_detectados": None,
              "espacios": [{"id": "A3", "estado": "libre", "dudoso": False}]}
    r = responder("disponibilidad", estado, random.Random(0))
    assert "12 de 40" in r and "70" in r
    assert "C4" in responder("confianza", estado) and "0.93" in responder("confianza", estado)
    assert "A3" in responder("ubicacion", estado)
    assert "28" in responder("conteo_ocupados", estado)
    assert "no he analizado" in responder("porcentaje", None)
