"""Criterio de aceptación: ningún archivo de código usa la librería prohibida, modelos de detección ya hechos,
ni pesos preentrenados. Las palabras se arman por partes para que este archivo no se detecte a sí mismo."""
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
YO = Path(__file__).resolve()

PROHIBIDOS = [
    "ultra" + "lytics", "yo" + "lo", "from" + "_pretrained", "pre" + "trained", "torch" + "vision",
    "hugging" + "face", "transfor" + "mers import", "faster" + "rcnn", "faster_" + "rcnn", "ret" + "inanet",
    "de" + "tr", "ssd" + "lite", "be" + "rt", "be" + "to", "g" + "pt", "torch" + ".hub", "ti" + "mm",
    "weights" + "=",
]


def archivos_codigo():
    for p in RAIZ.rglob("*"):
        if any(x in p.parts for x in (".venv", "datos", "__pycache__", ".pytest_cache", "pesos", "resultados")):
            continue
        if p.suffix in {".py", ".ipynb", ".txt", ".yaml", ".toml", ".cfg"} and p.resolve() != YO:
            yield p


def test_sin_librerias_ni_pesos_prohibidos():
    hallazgos = []
    for p in archivos_codigo():
        texto = p.read_text(encoding="utf-8", errors="ignore").lower()
        for palabra in PROHIBIDOS:
            # coincidencia como palabra completa para evitar falsos positivos (p. ej. 'alberto')
            if re.search(r"(?<![a-z])" + re.escape(palabra) + r"(?![a-z])", texto):
                hallazgos.append(f"{p.relative_to(RAIZ)}: '{palabra}'")
    assert not hallazgos, "Se encontraron referencias prohibidas:\n" + "\n".join(hallazgos)


def test_libreria_prohibida_no_importada():
    assert ("ultra" + "lytics") not in sys.modules
