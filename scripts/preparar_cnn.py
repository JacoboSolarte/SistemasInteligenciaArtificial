"""
Prepara el dataset del Modelo 1 (CNN libre/ocupado) a partir de CNRPark-EXT y PKLot.

Salida (en <datos>/procesados/cnn/):
    recortes.npy     uint8 [N, 64, 64, 3] (RGB)
    metadatos.csv    idx, etiqueta (0 libre / 1 ocupado), fuente, parqueadero, clima, particion (train/val/test)
    resumen.json     conteos por fuente, clima, clase y partición

Decisiones:
  * CNRPark-EXT ya trae recortes 150x150 de cada espacio: solo se redimensionan a 64x64.
  * Para PKLot NO se usan los recortes de PKLotSegmented: se recortan los espacios de las imágenes
    completas con la MISMA transformación de perspectiva que usa el pipeline de inferencia
    (inferencia/recorte.py), para que entrenamiento y producción vean el mismo tipo de recorte.
  * Se reserva un día de UFPR04 (no se usa para entrenar) para construir el video de demostración.
  * La división 70/15/15 es estratificada por (fuente, parqueadero, clima, etiqueta).
"""
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from herramientas.pklot_a_json import asignar_ids_por_filas, leer_xml_pklot, listar_pklot  # noqa: E402
from inferencia.recorte import guardar_espacios, recortar_espacio  # noqa: E402
from utilidades.comun import RAIZ, cargar_config, fijar_semillas, guardar_json, ruta  # noqa: E402

CLIMAS_CNR = {"SUNNY": "soleado", "OVERCAST": "nublado", "RAINY": "lluvioso"}


# ---------------------------------------------------------------------------
# CNRPark-EXT
# ---------------------------------------------------------------------------
def listar_cnrext(raiz: Path) -> pd.DataFrame:
    """Lee las etiquetas de CNRPark-EXT (LABELS/all.txt si existe; si no, el CSV)."""
    etiquetas = next(iter(raiz.rglob("LABELS/all.txt")), None)
    filas = []
    if etiquetas:
        base = etiquetas.parent.parent / "PATCHES"
        for linea in etiquetas.read_text().splitlines():
            if not linea.strip():
                continue
            rel, lab = linea.split()
            partes = rel.split("/")
            filas.append({"ruta": str(base / rel), "etiqueta": int(lab), "clima": CLIMAS_CNR.get(partes[0], "nublado"),
                          "parqueadero": "CNR-" + partes[2]})
    else:
        csv = next(iter(raiz.rglob("CNRPark+EXT.csv")))
        df = pd.read_csv(csv)
        indice = {p.name: p for p in raiz.rglob("*.jpg")}
        for _, f in df.iterrows():
            nombre = Path(str(f["image_url"])).name
            if nombre not in indice:
                continue
            partes = Path(str(f["image_url"])).parts
            clima = next((CLIMAS_CNR[p] for p in partes if p in CLIMAS_CNR), "nublado")
            filas.append({"ruta": str(indice[nombre]), "etiqueta": int(f["occupancy"]), "clima": clima,
                          "parqueadero": f"CNR-camera{f['camera']}"})
    df = pd.DataFrame(filas)
    df["fuente"] = "cnrext"
    return df


def muestrear_estratificado(df, n, columnas, semilla):
    if len(df) <= n:
        return df
    return df.groupby(columnas).sample(frac=n / len(df), random_state=semilla)


def leer_recorte_cnr(ruta_img, tam):
    img = cv2.imread(ruta_img)
    if img is None:
        return None
    return cv2.cvtColor(cv2.resize(img, (tam, tam), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# PKLot
# ---------------------------------------------------------------------------
def elegir_dia_reservado(imagenes, parq, valor):
    if valor and valor != "auto":
        return valor
    dias = Counter(f["fecha"] for f in imagenes if f["parqueadero"] == parq and f["clima"] == "soleado")
    return dias.most_common(1)[0][0]


def muestrear_pklot(imagenes, por_grupo, reservados, semilla):
    rng = random.Random(semilla)
    elegidas = []
    for parq in ("UFPR04", "UFPR05", "PUCPR"):
        for clima in ("soleado", "nublado", "lluvioso"):
            grupo = [f for f in imagenes if f["parqueadero"] == parq and f["clima"] == clima
                     and reservados.get(parq) != f["fecha"]]
            rng.shuffle(grupo)
            elegidas += grupo[:por_grupo]
    return elegidas


def recortes_de_imagen(info, tam):
    img = cv2.imread(info["imagen"])
    if img is None:
        return []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    salida = []
    for e in leer_xml_pklot(info["xml"]):
        if e["ocupado"] is None:
            continue
        salida.append((recortar_espacio(rgb, e["poligono"], tam), e["ocupado"], info))
    return salida


def generar_jsons_espacios(imagenes, reservados):
    """Un JSON de espacios por parqueadero (cámara fija), desde el XML de una imagen del día reservado o la primera."""
    for parq in ("UFPR04", "UFPR05", "PUCPR"):
        cand = [f for f in imagenes if f["parqueadero"] == parq]
        cand = [f for f in cand if f["fecha"] == reservados.get(parq)] or cand
        # se elige el XML con más espacios etiquetados
        mejor = max(cand[:60], key=lambda f: len(leer_xml_pklot(f["xml"])))
        esp = asignar_ids_por_filas(leer_xml_pklot(mejor["xml"]))
        guardar_espacios(esp, RAIZ / "config" / f"espacios_{parq}.json")
        print(f"  config/espacios_{parq}.json: {len(esp)} espacios (de {Path(mejor['xml']).name})")


def main():
    cfg = cargar_config()
    fijar_semillas(cfg["semilla"])
    cd = cfg["datos"]
    tam = cd["tam_recorte"]
    crudos = ruta(cfg, "datos") / "crudos"
    salida = ruta(cfg, "datos") / "procesados" / "cnn"
    salida.mkdir(parents=True, exist_ok=True)

    # ---------------- CNRPark-EXT
    print("Leyendo CNRPark-EXT ...")
    df_cnr = listar_cnrext(crudos / "cnrpark_ext")
    print(f"  {len(df_cnr)} recortes disponibles")
    df_cnr = muestrear_estratificado(df_cnr, cd["max_cnrext"], ["clima", "etiqueta"], cfg["semilla"]).reset_index(drop=True)
    print(f"  {len(df_cnr)} recortes muestreados")

    # ---------------- PKLot
    print("Listando PKLot ...")
    imagenes = listar_pklot(crudos / "pklot")
    print(f"  {len(imagenes)} imágenes con XML")
    reservados = {p: elegir_dia_reservado(imagenes, p, v) for p, v in (cd.get("dias_reservados_video") or {}).items()}
    print(f"  días reservados para video (no se entrenan): {reservados}")
    guardar_json(reservados, ruta(cfg, "datos") / "procesados" / "dias_reservados.json")
    generar_jsons_espacios(imagenes, reservados)
    elegidas = muestrear_pklot(imagenes, cd["imagenes_pklot_por_grupo"], reservados, cfg["semilla"])
    print(f"  {len(elegidas)} imágenes PKLot elegidas para recortar")

    with ThreadPoolExecutor(8) as ex:
        rec_pk = [r for lista in ex.map(lambda i: recortes_de_imagen(i, tam), elegidas) for r in lista]
    print(f"  {len(rec_pk)} recortes PKLot")

    # ---------------- Escritura en un único arreglo
    n_total = len(df_cnr) + len(rec_pk)
    arr = np.lib.format.open_memmap(salida / "recortes.npy", mode="w+", dtype=np.uint8, shape=(n_total, tam, tam, 3))
    metadatos = []
    print("Leyendo y redimensionando recortes de CNRPark-EXT ...")
    with ThreadPoolExecutor(8) as ex:
        for i, rec in enumerate(ex.map(lambda r: leer_recorte_cnr(r, tam), df_cnr["ruta"])):
            f = df_cnr.iloc[i]
            arr[len(metadatos)] = rec if rec is not None else 0
            metadatos.append({"etiqueta": int(f["etiqueta"]), "fuente": "cnrext", "parqueadero": f["parqueadero"],
                              "clima": f["clima"], "origen": Path(f["ruta"]).name, "valido": rec is not None})
    for rec, ocup, info in rec_pk:
        arr[len(metadatos)] = rec
        metadatos.append({"etiqueta": int(ocup), "fuente": "pklot", "parqueadero": info["parqueadero"],
                          "clima": info["clima"], "origen": Path(info["imagen"]).name, "valido": True})
    arr.flush()
    meta = pd.DataFrame(metadatos)
    meta.insert(0, "idx", np.arange(len(meta)))
    meta = meta[meta["valido"]].drop(columns="valido")

    # ---------------- División estratificada 70/15/15
    p_train, p_val, p_test = cd["proporciones"]
    estrato = meta["fuente"] + "_" + meta["parqueadero"].str[:6] + "_" + meta["clima"] + "_" + meta["etiqueta"].astype(str)
    idx_train, idx_resto = train_test_split(meta.index, test_size=p_val + p_test, stratify=estrato, random_state=cfg["semilla"])
    idx_val, idx_test = train_test_split(idx_resto, test_size=p_test / (p_val + p_test), stratify=estrato[idx_resto],
                                         random_state=cfg["semilla"])
    meta["particion"] = ""
    meta.loc[idx_train, "particion"] = "train"
    meta.loc[idx_val, "particion"] = "val"
    meta.loc[idx_test, "particion"] = "test"
    meta.to_csv(salida / "metadatos.csv", index=False)

    resumen = {
        "total": int(len(meta)),
        "por_fuente": meta["fuente"].value_counts().to_dict(),
        "por_clase": {("ocupado" if k == 1 else "libre"): int(v) for k, v in meta["etiqueta"].value_counts().items()},
        "por_clima": meta["clima"].value_counts().to_dict(),
        "por_particion": meta["particion"].value_counts().to_dict(),
        "por_parqueadero_pklot": meta[meta.fuente == "pklot"]["parqueadero"].value_counts().to_dict(),
    }
    guardar_json(resumen, salida / "resumen.json")
    guardar_json(resumen, ruta(cfg, "resultados") / "resumen_datos_cnn.json")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
