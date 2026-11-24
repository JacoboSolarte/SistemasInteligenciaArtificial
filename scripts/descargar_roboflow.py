"""
Descarga DATASETS (solo imágenes + anotaciones en formato COCO) desde Roboflow Universe.
No se descarga ningún modelo ni peso: todo el entrenamiento del proyecto es desde cero.

La API key se lee de config/secretos.yaml (roboflow_api_key: "...") o de la variable de entorno
ROBOFLOW_API_KEY. Ese archivo está en .gitignore.

    python scripts/descargar_roboflow.py            # descarga todos los de la lista
    python scripts/descargar_roboflow.py --solo carpk
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilidades.comun import cargar_config, ruta  # noqa: E402

# nombre local -> (workspace/proyecto, versión, descripción de la vista). Todos con licencia CC BY 4.0.
DATASETS = {
    "carpk": ("elpida-eleftheriadi/carpk-xk8e1", 1, "dron, vista cenital; solo vehículos (copia de CARPK, Hsieh et al. 2017)"),
    "dron_parkinglot": ("drone-software-meetup-k1lqk/parkinglot-g5ppr", 1, "dron, vista cenital; espacios Empty/Car"),
    "lateral_park2": ("my-project-ppguz/park-2", 1, "cámara lateral en poste; espacios empty/occupied"),
    "suelo_mingouws": ("mingouws/parking-space-detection-uum5d", 10, "cámara a nivel de suelo / cercana; empty/occupied"),
    "garaje_gvjqh": ("parking-gvjqh/parking-lot-9sjil", 2, "garaje y cámaras laterales; empty/occupied"),
    "noche_drone": ("drone-wrcaq/parking-space-ahhy7", 1, "a nivel de suelo, de noche; empty/occupied"),
}


def leer_api_key():
    clave = os.environ.get("ROBOFLOW_API_KEY")
    sec = Path(__file__).resolve().parents[1] / "config" / "secretos.yaml"
    if not clave and sec.exists():
        clave = (yaml.safe_load(sec.read_text(encoding="utf-8")) or {}).get("roboflow_api_key")
    if not clave:
        raise SystemExit("Falta la API key de Roboflow (config/secretos.yaml o ROBOFLOW_API_KEY)")
    return clave


def obtener(url, intentos=20):
    for i in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception as e:  # red inestable: reintentar
            print(f"  intento {i + 1} falló: {e}")
            time.sleep(min(5 * (i + 1), 30))
    raise IOError(f"No se pudo descargar {url}")


def descargar(nombre, proyecto, version, clave, destino):
    salida = destino / nombre
    if (salida / "train" / "_annotations.coco.json").exists():
        print(f"{nombre}: ya existe, se omite")
        return
    # Si Roboflow aún no tiene generada la exportación COCO, responde {"progress": x}: se espera y se reintenta
    for _ in range(120):
        info = json.loads(obtener(f"https://api.roboflow.com/{proyecto}/{version}/coco?api_key={clave}"))
        if "export" in info:
            break
        print(f"  generando exportación en Roboflow... {info.get('progress', '?')}")
        time.sleep(5)
    else:
        raise IOError(f"Roboflow no generó la exportación de {proyecto}")
    enlace = info["export"]["link"]
    print(f"{nombre}: descargando {proyecto} v{version} ...")
    datos = obtener(enlace)
    salida.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(datos)) as z:
        z.extractall(salida)
    print(f"{nombre}: listo ({len(datos) / 1e6:.1f} MB) -> {salida}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", nargs="*", default=None)
    a = ap.parse_args()
    cfg = cargar_config()
    destino = ruta(cfg, "datos") / "roboflow"
    clave = leer_api_key()
    for nombre, (proyecto, version, desc) in DATASETS.items():
        if a.solo and nombre not in a.solo:
            continue
        print(f"[{nombre}] {desc}")
        descargar(nombre, proyecto, version, clave, destino)


if __name__ == "__main__":
    main()
