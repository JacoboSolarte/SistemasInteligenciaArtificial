"""
Punto de entrada para Streamlit Community Cloud.

Streamlit Cloud instala las dependencias del requirements.txt que está junto al archivo de entrada,
así que este archivo vive en despliegue/ (con un requirements.txt de PyTorch solo CPU) y ejecuta
la app principal (app.py, en la raíz del repositorio) sin duplicar código.
"""
import runpy
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

runpy.run_path(str(RAIZ / "app.py"), run_name="__main__")
