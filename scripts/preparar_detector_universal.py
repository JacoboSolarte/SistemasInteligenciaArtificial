"""
Prepara el dataset del DETECTOR UNIVERSAL (modo automático, sin JSON de espacios).

Une varias fuentes con vistas distintas en un solo formato con 3 clases:
    0 = espacio libre     1 = espacio ocupado     2 = vehículo

Fuente                   Vista                              Clases etiquetadas
-----------------------  ---------------------------------  -------------------------------
pklot                    cámara en edificio (inclinada)     libre, ocupado, vehículo(*)
carpk                    dron, cenital                      vehículo
dron_parkinglot          dron, cenital                      libre, ocupado, vehículo(*)
lateral_park2            cámara lateral en poste            libre, ocupado, vehículo(*)
suelo_mingouws           cámara cercana / nivel de suelo    libre, ocupado, vehículo(*)
garaje_gvjqh             garaje y cámaras laterales         libre, ocupado, vehículo(*)
noche_drone              nivel de suelo, de noche           libre, ocupado, vehículo(*)
(*) la caja del vehículo se aproxima con la caja del espacio ocupado.

Cuando una fuente NO etiqueta una clase (p. ej. CARPK no marca espacios), esa clase queda
"no etiquetada" para sus imágenes y la pérdida la ignora (no se castiga al modelo por predecirla).

Salida (en <datos>/procesados/universal/):
    imagenes/<fuente>/<n>.jpg   imagen reescalada (lado mayor <= 800 px)
    anotaciones.json            [{archivo, fuente, particion, ancho, alto, cajas: [[x1,y1,x2,y2,clase],...],
                                  etiquetadas: [libre, ocupado, vehiculo]}]
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from herramientas.pklot_a_json import leer_xml_pklot, listar_pklot  # noqa: E402
from utilidades.comun import cargar_config, guardar_json, leer_json, ruta  # noqa: E402

LADO_MAX = 800
CLASES = ("libre", "ocupado", "vehiculo")


def guardar_imagen(img, destino):
    """Reescala para que el lado mayor sea <= LADO_MAX y guarda en JPEG. Devuelve la escala aplicada."""
    h, w = img.shape[:2]
    esc = min(1.0, LADO_MAX / max(h, w))
    if esc < 1.0:
        img = cv2.resize(img, (round(w * esc), round(h * esc)), interpolation=cv2.INTER_AREA)
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return esc, img.shape[1], img.shape[0]


def particion_por_grupo(grupos, semilla, prop=(0.70, 0.15, 0.15)):
    """Asigna train/val/test por GRUPO (p. ej. imagen original y sus copias aumentadas van juntas)."""
    unicos = sorted(set(grupos))
    random.Random(semilla).shuffle(unicos)
    n = len(unicos)
    n_tr, n_va = round(n * prop[0]), round(n * prop[1])
    asign = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, g in enumerate(unicos)}
    return [asign[g] for g in grupos]


# ---------------------------------------------------------------- PKLot
def preparar_pklot(cfg, salida, por_grupo, semilla):
    crudos = ruta(cfg, "datos") / "crudos" / "pklot"
    archivo_res = ruta(cfg, "datos") / "procesados" / "dias_reservados.json"
    reservados = leer_json(archivo_res) if archivo_res.exists() else {}
    imagenes = listar_pklot(crudos)
    rng = random.Random(semilla)
    elegidas = []
    for parq in ("UFPR04", "UFPR05", "PUCPR"):
        for clima in ("soleado", "nublado", "lluvioso"):
            g = [f for f in imagenes if f["parqueadero"] == parq and f["clima"] == clima and reservados.get(parq) != f["fecha"]]
            rng.shuffle(g)
            elegidas += g[:por_grupo]
    # partición por DÍA: frames del mismo día (casi idénticos) nunca quedan en train y test a la vez
    parts = particion_por_grupo([f"{f['parqueadero']}_{f['fecha']}" for f in elegidas], semilla)
    filas = []
    for k, (info, part) in enumerate(zip(elegidas, parts)):
        img = cv2.imread(info["imagen"])
        if img is None:
            continue
        esc, w, h = guardar_imagen(img, salida / "imagenes" / "pklot" / f"{k}.jpg")
        cajas = []
        for e in leer_xml_pklot(info["xml"]):
            if e["ocupado"] not in (0, 1):
                continue
            x, y, bw, bh = cv2.boundingRect(np.asarray(e["poligono"], dtype=np.int32))
            caja = [max(x, 0) * esc, max(y, 0) * esc, min(x + bw, img.shape[1]) * esc, min(y + bh, img.shape[0]) * esc]
            if caja[2] - caja[0] < 2 or caja[3] - caja[1] < 2:
                continue
            cajas.append(caja + [int(e["ocupado"])])
            if e["ocupado"] == 1:
                cajas.append(caja + [2])
        filas.append({"archivo": f"imagenes/pklot/{k}.jpg", "fuente": "pklot", "particion": part, "ancho": w, "alto": h,
                      "cajas": cajas, "etiquetadas": [1, 1, 1], "detalle": info["parqueadero"]})
    return filas


# ---------------------------------------------------------------- CARPK (original o copia de Roboflow)
def preparar_carpk_original(raiz, salida):
    base = raiz / "datasets" / "CARPK_devkit" / "data"
    filas = []
    test = set((base / "ImageSets" / "test.txt").read_text().split())
    nombres = sorted(set((base / "ImageSets" / "train.txt").read_text().split()) | test)
    # validación: un 15 % del train oficial, por imagen
    tr = [n for n in nombres if n not in test]
    val = set(random.Random(0).sample(tr, round(0.15 * len(tr))))
    for k, n in enumerate(nombres):
        img = cv2.imread(str(base / "Images" / f"{n}.png"))
        if img is None:
            continue
        esc, w, h = guardar_imagen(img, salida / "imagenes" / "carpk" / f"{k}.jpg")
        cajas = []
        for linea in (base / "Annotations" / f"{n}.txt").read_text().splitlines():
            v = linea.split()
            if len(v) >= 4:
                x1, y1, x2, y2 = (float(t) * esc for t in v[:4])
                cajas.append([x1, y1, x2, y2, 2])
        part = "test" if n in test else ("val" if n in val else "train")
        filas.append({"archivo": f"imagenes/carpk/{k}.jpg", "fuente": "carpk", "particion": part, "ancho": w, "alto": h,
                      "cajas": cajas, "etiquetadas": [0, 0, 1], "detalle": n.split("_")[0]})
    return filas


def clase_roboflow(nombre, fuente):
    n = nombre.lower()
    if fuente == "carpk":
        return 2
    if "empty" in n or "free" in n:
        return 0
    if "occup" in n or n == "car":
        return 1
    return None


def tiene_mascara_negra(img, bloque=16):
    """Algunas imágenes del dataset de dron traen los objetos tapados con rectángulos casi negros
    (anonimización): se descartan. Se cuentan bloques de 16x16 completamente oscuros; un carro negro
    real puede llenar a lo sumo uno, una máscara llena muchos."""
    m = img.max(axis=2)
    h, w = (m.shape[0] // bloque) * bloque, (m.shape[1] // bloque) * bloque
    b = m[:h, :w].reshape(h // bloque, bloque, w // bloque, bloque).max(axis=(1, 3))
    return int((b < 35).sum()) > 1


def preparar_roboflow(raiz, fuente, salida, semilla):
    filas = []
    k = 0
    carpetas = {"train": "train", "valid": "val", "test": "test"}
    hay_test = (raiz / "test" / "_annotations.coco.json").exists()
    for carpeta, part in carpetas.items():
        f = raiz / carpeta / "_annotations.coco.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        por_img = defaultdict(list)
        for a in d["annotations"]:
            por_img[a["image_id"]].append(a)
        for im in d["images"]:
            img = cv2.imread(str(raiz / carpeta / im["file_name"]))
            if img is None or (fuente == "dron_parkinglot" and tiene_mascara_negra(img)):
                continue
            fx = 1.0
            if fuente == "carpk":
                # la copia de Roboflow viene estirada de 1280x720 a 416x416: se devuelve la proporción 16:9
                fx = (img.shape[0] * 1280 / 720) / img.shape[1]
                img = cv2.resize(img, (round(img.shape[1] * fx), img.shape[0]), interpolation=cv2.INTER_CUBIC)
            cajas = []
            for a in por_img[im["id"]]:
                c = clase_roboflow(cats[a["category_id"]], fuente)
                if c is None:
                    continue
                x, y, w, h = a["bbox"]
                x, w = x * fx, w * fx
                cajas.append([x, y, x + w, y + h, c])
                if c == 1:
                    cajas.append([x, y, x + w, y + h, 2])
            if not cajas:
                continue
            esc, w, h = guardar_imagen(img, salida / "imagenes" / fuente / f"{k}.jpg")
            cajas = [[v * esc for v in c[:4]] + [c[4]] for c in cajas]
            grupo = re.sub(r"\.rf\..*$", "", im["file_name"])  # imagen original (sin el sufijo de aumento)
            num = re.search(r"MP4-(\d+)", grupo)
            if fuente == "dron_parkinglot" and num:  # frames de un mismo video: bloques de 10 frames juntos
                grupo = f"bloque{int(num.group(1)) // 10}"
            filas.append({"archivo": f"imagenes/{fuente}/{k}.jpg", "fuente": fuente, "particion": part, "ancho": w, "alto": h,
                          "cajas": cajas, "etiquetadas": [0, 0, 1] if fuente == "carpk" else [1, 1, 1], "grupo": grupo})
            k += 1
    # Fuentes sin test (o solo con train): se re-particiona por imagen original para que no haya fugas
    # (el de dron también: sus pocas imágenes limpias quedaron casi todas en valid/test de Roboflow)
    if not hay_test or fuente == "dron_parkinglot":
        parts = particion_por_grupo([f["grupo"] for f in filas], semilla)
        for f, p in zip(filas, parts):
            f["particion"] = p
    return filas


def main():
    cfg = cargar_config()
    semilla = cfg["semilla"]
    datos = ruta(cfg, "datos")
    salida = datos / "procesados" / "universal"
    salida.mkdir(parents=True, exist_ok=True)
    todas = []

    print("PKLot ...")
    todas += preparar_pklot(cfg, salida, cfg["detector_universal"]["pklot_por_grupo"], semilla)

    # El CARPK original (lafi.github.io/LPN) viene en un zip con contraseña que se pide a los autores por
    # formulario; si se consigue y se descomprime en <datos>/carpk_original se usa; si no, la copia de Roboflow.
    carpk_orig = datos / "carpk_original"
    if (carpk_orig / "datasets" / "CARPK_devkit" / "data" / "Images").exists():
        print("CARPK (original) ...")
        todas += preparar_carpk_original(carpk_orig, salida)
    else:
        print("CARPK (copia de Roboflow) ...")
        todas += preparar_roboflow(datos / "roboflow" / "carpk", "carpk", salida, semilla)

    for fuente in ("dron_parkinglot", "lateral_park2", "suelo_mingouws", "garaje_gvjqh", "noche_drone"):
        print(f"{fuente} ...")
        todas += preparar_roboflow(datos / "roboflow" / fuente, fuente, salida, semilla)

    for f in todas:
        f["cajas"] = [[round(v, 1) for v in c[:4]] + [int(c[4])] for c in f["cajas"]]
    guardar_json(todas, salida / "anotaciones.json")

    print("\nImágenes por fuente y partición:")
    cuenta = Counter((f["fuente"], f["particion"]) for f in todas)
    for fuente in sorted({f["fuente"] for f in todas}):
        cajas = Counter(c[4] for f in todas if f["fuente"] == fuente for c in f["cajas"])
        print(f"  {fuente:16s} train={cuenta[(fuente, 'train')]:5d} val={cuenta[(fuente, 'val')]:4d} "
              f"test={cuenta[(fuente, 'test')]:4d} | libres={cajas[0]} ocupados={cajas[1]} vehiculos={cajas[2]}")


if __name__ == "__main__":
    main()
