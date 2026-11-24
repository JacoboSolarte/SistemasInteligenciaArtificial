"""
Utilidades para PKLot + conversor XML -> JSON de espacios.

Formato XML de PKLot (una por imagen):
    <parking id="ufpr04">
      <space id="1" occupied="0">
        <rotatedRect> ... </rotatedRect>
        <contour> <point x="..." y="..."/> x4 </contour>
      </space> ...

Uso del conversor (genera config/espacios_<nombre>.json sin marcar a mano):
    python herramientas/pklot_a_json.py --xml ruta/a/imagen.xml --nombre UFPR04
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferencia.recorte import guardar_espacios  # noqa: E402

CLIMAS = {"sunny": "soleado", "cloudy": "nublado", "rainy": "lluvioso"}
PARQUEADEROS = ("UFPR04", "UFPR05", "PUCPR")


def leer_xml_pklot(ruta_xml):
    """Devuelve [{'pklot_id': int, 'ocupado': 0/1/None, 'poligono': [[x,y]x4]}]."""
    try:
        raiz = ET.parse(ruta_xml).getroot()
    except ET.ParseError:
        return []
    espacios = []
    for sp in raiz.iter():
        if sp.tag.lower() != "space":
            continue
        puntos = [[int(float(p.get("x"))), int(float(p.get("y")))]
                  for p in sp.iter() if p.tag.lower() == "point" and p.get("x") is not None]
        if len(puntos) < 3:
            continue
        occ = sp.get("occupied")
        espacios.append({"pklot_id": int(sp.get("id")), "ocupado": None if occ is None else int(occ), "poligono": puntos})
    return espacios


def asignar_ids_por_filas(espacios):
    """Agrupa los espacios en filas según la coordenada y del centro: fila A, B, C...; número según x.
    Así el asistente puede decir 'Libres: A3, A7, B2' como en un parqueadero real."""
    centros = np.array([np.mean(e["poligono"], axis=0) for e in espacios])
    altos = np.array([np.ptp(np.asarray(e["poligono"])[:, 1]) for e in espacios])
    umbral = 0.6 * float(np.median(altos))
    orden = np.argsort(centros[:, 1])
    filas, actual, ref = [], [orden[0]], centros[orden[0], 1]
    for i in orden[1:]:
        if centros[i, 1] - ref > umbral:
            filas.append(actual)
            actual = []
        actual.append(i)
        ref = np.mean(centros[actual, 1])
    filas.append(actual)
    letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    salida = []
    for f, fila in enumerate(filas):
        for n, i in enumerate(sorted(fila, key=lambda j: centros[j, 0]), start=1):
            pref = letras[f] if f < 26 else f"F{f}-"
            salida.append({"id": f"{pref}{n}", "poligono": espacios[i]["poligono"], "pklot_id": espacios[i]["pklot_id"]})
    return salida


def listar_pklot(raiz):
    """Recorre PKLot y devuelve una lista de dicts con ruta de imagen/XML, parqueadero, clima y fecha."""
    raiz = Path(raiz)
    filas = []
    for img in raiz.rglob("*.jpg"):
        partes = [p for p in img.parts]
        parq = next((p for p in partes if p.upper() in PARQUEADEROS), None)
        clima = next((CLIMAS[p.lower()] for p in partes if p.lower() in CLIMAS), None)
        xml = img.with_suffix(".xml")
        if parq is None or clima is None or not xml.exists() or "segmented" in str(img).lower():
            continue
        filas.append({"imagen": str(img), "xml": str(xml), "parqueadero": parq.upper(), "clima": clima,
                      "fecha": img.parent.name})
    filas.sort(key=lambda f: f["imagen"])
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    esp = asignar_ids_por_filas(leer_xml_pklot(a.xml))
    salida = a.salida or Path(__file__).resolve().parents[1] / "config" / f"espacios_{a.nombre}.json"
    guardar_espacios(esp, salida)
    print(f"{len(esp)} espacios guardados en {salida}")


if __name__ == "__main__":
    main()
