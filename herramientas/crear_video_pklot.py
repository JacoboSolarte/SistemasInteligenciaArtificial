"""
Construye un video de demostración (time-lapse) con las imágenes de un día de PKLot que NO se usó
para entrenar, y el archivo de anotaciones de referencia para evaluar el sistema en video.

    python herramientas/crear_video_pklot.py --parqueadero UFPR04

Salidas:
    ejemplos/video_<parq>.mp4                  (1 imagen de PKLot = 1 frame; PKLot toma una foto cada ~5 min)
    evaluacion/anotaciones_manuales.json       estado real de cada espacio en algunos frames
    ejemplos/<parq>_*.jpg                      imágenes sueltas de ejemplo para la app
Las anotaciones vienen de los XML de PKLot, que fueron etiquetados a mano por los autores del dataset.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from herramientas.pklot_a_json import leer_xml_pklot, listar_pklot  # noqa: E402
from inferencia.pipeline import EscritorVideo  # noqa: E402
from inferencia.recorte import cargar_espacios  # noqa: E402
from utilidades.comun import RAIZ, cargar_config, guardar_json, leer_json, ruta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parqueadero", default="UFPR04")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--cada", type=int, default=3, help="anotar 1 de cada N frames")
    a = ap.parse_args()
    cfg = cargar_config()
    reservados = leer_json(ruta(cfg, "datos") / "procesados" / "dias_reservados.json")
    dia = reservados[a.parqueadero]
    imgs = [f for f in listar_pklot(ruta(cfg, "datos") / "crudos" / "pklot")
            if f["parqueadero"] == a.parqueadero and f["fecha"] == dia]
    imgs.sort(key=lambda f: Path(f["imagen"]).name)
    print(f"{len(imgs)} imágenes del día reservado {dia}")
    espacios_json = RAIZ / "config" / f"espacios_{a.parqueadero}.json"
    espacios = cargar_espacios(espacios_json)
    por_pklot = {e["pklot_id"]: e["id"] for e in espacios}

    ejemplos = RAIZ / "ejemplos"
    ejemplos.mkdir(exist_ok=True)
    salida = ejemplos / f"video_{a.parqueadero}.mp4"
    esc = EscritorVideo(salida, a.fps)
    anot = {}
    tam = None
    for k, f in enumerate(imgs):
        img = cv2.imread(f["imagen"])
        tam = tam or (img.shape[1], img.shape[0])
        esc.escribir(cv2.resize(img, tam))
        if k % a.cada == 0:
            estado = {por_pklot[e["pklot_id"]]: ("ocupado" if e["ocupado"] else "libre")
                      for e in leer_xml_pklot(f["xml"]) if e["ocupado"] is not None and e["pklot_id"] in por_pklot}
            anot[str(k)] = {"imagen": Path(f["imagen"]).name, "estado": estado}
    esc.cerrar()
    guardar_json({"video": f"ejemplos/{salida.name}", "espacios": f"config/{espacios_json.name}", "fps": a.fps,
                  "fuente": "XML de PKLot (etiquetado manual de los autores del dataset)", "frames": anot},
                 RAIZ / "evaluacion" / "anotaciones_manuales.json")
    # imágenes de ejemplo: 3 momentos del día reservado
    for nombre, f in (("manana", imgs[len(imgs) // 8]), ("mediodia", imgs[len(imgs) // 2]), ("tarde", imgs[-len(imgs) // 8])):
        shutil.copy(f["imagen"], ejemplos / f"{a.parqueadero}_{nombre}.jpg")
    print(f"Video: {salida} ({len(imgs)} frames a {a.fps} fps) | {len(anot)} frames anotados")


if __name__ == "__main__":
    main()
