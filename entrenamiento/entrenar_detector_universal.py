"""
Entrenamiento del Modelo 4: DetectorUniversal (modo automático), desde cero.

    python entrenamiento/entrenar_detector_universal.py
    python entrenamiento/entrenar_detector_universal.py --epocas 2 --muestras 200   # prueba rápida

Aumento de datos (para "cualquier situación"):
  * escala aleatoria (log-uniforme entre escala_min y escala_max): simula dron más alto/bajo y zoom.
  * recorte aleatorio de tam_entrada x tam_entrada (con relleno si la imagen es más pequeña).
  * volteo horizontal; en las fuentes CENITALES (dron) también volteo vertical y rotación de 90°,
    porque desde arriba los carros pueden aparecer en cualquier orientación.
  * color: brillo, contraste, saturación, tono, a veces gris (cámaras de noche/IR) y desenfoque.
Muestreo: cada lote mezcla fuentes según `pesos_fuente` (PKLot, la más grande, no domina).
Validación: AP@0.5 por (fuente, clase) en un subconjunto de val; se guarda el mejor por mAP medio.
"""
import argparse
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluacion.metricas_detector import average_precision  # noqa: E402
from inferencia.automatico import preparar_entrada  # noqa: E402
from inferencia.nms import nms  # noqa: E402
from modelos.detector_universal import (CLASES, DetectorUniversal, PerdidaDetectorUniversal,  # noqa: E402
                                        construir_objetivo, decodificar)
from utilidades.comun import cargar_config, contar_parametros, fijar_semillas, guardar_json, leer_json, ruta  # noqa: E402

CENITALES = {"carpk", "dron_parkinglot"}


def aumentar_color(img, rng):
    img = img.astype(np.float32)
    img = img * rng.uniform(0.6, 1.4) + rng.uniform(-30, 30)                 # contraste y brillo
    gris = img.mean(axis=2, keepdims=True)
    img = gris + (img - gris) * rng.uniform(0.5, 1.5)                        # saturación
    if rng.random() < 0.3:
        img = img[..., rng.permutation(3)]                                   # cambio de tono grueso
    if rng.random() < 0.1:
        img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)          # gris (noche / IR)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.6))
    return img


class DatasetUniversal(Dataset):
    def __init__(self, filas, raiz, cfg_u, entrenamiento=True):
        self.filas, self.raiz, self.cu, self.entrenamiento = filas, Path(raiz), cfg_u, entrenamiento
        self.tam = cfg_u["tam_entrada"]

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, i):
        f = self.filas[i]
        rng = np.random.default_rng((i * 7919 + int(time.time() * 1e6)) % (2 ** 32))
        img = cv2.cvtColor(cv2.imread(str(self.raiz / f["archivo"])), cv2.COLOR_BGR2RGB)
        cajas = np.array([c for c in f["cajas"]], np.float32).reshape(-1, 5)
        t = self.tam
        # escala aleatoria
        esc = math.exp(rng.uniform(math.log(self.cu["escala_min"]), math.log(self.cu["escala_max"])))
        h, w = img.shape[:2]
        nh, nw = max(8, round(h * esc)), max(8, round(w * esc))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
        cajas[:, :4] *= esc
        # recorte aleatorio t x t (relleno gris si hace falta)
        x0 = int(rng.integers(0, max(nw - t, 0) + 1))
        y0 = int(rng.integers(0, max(nh - t, 0) + 1))
        lienzo = np.full((t, t, 3), 114, np.uint8)
        px = int(rng.integers(0, max(t - nw, 0) + 1))
        py = int(rng.integers(0, max(t - nh, 0) + 1))
        trozo = img[y0:y0 + t, x0:x0 + t]
        lienzo[py:py + trozo.shape[0], px:px + trozo.shape[1]] = trozo
        if len(cajas):
            area0 = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
            cajas[:, [0, 2]] += px - x0
            cajas[:, [1, 3]] += py - y0
            cajas[:, [0, 2]] = cajas[:, [0, 2]].clip(px, px + trozo.shape[1])
            cajas[:, [1, 3]] = cajas[:, [1, 3]].clip(py, py + trozo.shape[0])
            area1 = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
            cajas = cajas[area1 >= 0.35 * np.maximum(area0, 1e-6)]
        img = lienzo
        # volteos / rotación
        if rng.random() < 0.5:
            img = img[:, ::-1]
            cajas[:, [0, 2]] = t - cajas[:, [2, 0]]
        if f["fuente"] in CENITALES:
            if rng.random() < 0.5:
                img = img[::-1]
                cajas[:, [1, 3]] = t - cajas[:, [3, 1]]
            if rng.random() < 0.5:  # rotación 90° antihoraria: (x, y) -> (y, t - x)
                img = np.rot90(img)
                cajas = np.stack([cajas[:, 1], t - cajas[:, 2], cajas[:, 3], t - cajas[:, 0], cajas[:, 4]], 1) \
                    if len(cajas) else cajas
        img = aumentar_color(np.ascontiguousarray(img), rng)
        objetivo = construir_objetivo(cajas, t, t)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return (x, *[torch.from_numpy(o) for o in objetivo], torch.tensor(f["etiquetadas"], dtype=torch.float32))


@torch.no_grad()
def predecir_filas(modelo, filas, raiz, dev, lado, umbral_nms=0.5, detector=None):
    """Predicciones por imagen (sin aumentos) a lado mayor `lado`. Devuelve lista de dicts clase->(cajas, puntajes)
    en píxeles de la imagen guardada."""
    modelo.eval()
    salida = []
    lados = sorted(lado) if isinstance(lado, (list, tuple)) else [lado]  # lista = multiescala
    for f in filas:
        rgb = cv2.cvtColor(cv2.imread(str(Path(raiz) / f["archivo"])), cv2.COLOR_BGR2RGB)
        partes = []
        if detector is not None:  # p. ej. AnalizadorAutomatico.detectar (resolución "auto")
            partes.append(detector(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
            lados = []
        for n_escala, ld in enumerate(lados):
            entrada, esc = preparar_entrada(rgb, int(ld))
            x = torch.from_numpy(entrada).permute(2, 0, 1)[None].float().div(255).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                s = modelo(x)
            c_, p_, k_ = decodificar(s, umbral=0.02)[0]
            c_ = c_ / esc
            if n_escala > 0:  # misma regla que inferencia/automatico.py: escalas grandes -> solo objetos pequeños
                lado_min = np.minimum(c_[:, 2] - c_[:, 0], c_[:, 3] - c_[:, 1]) * lados[0] / max(rgb.shape[:2])
                m = lado_min < 22
                c_, p_, k_ = c_[m], p_[m], k_[m]
            partes.append((c_, p_, k_))
        cajas = np.concatenate([t[0] for t in partes])
        punt = np.concatenate([t[1] for t in partes])
        cls = np.concatenate([t[2] for t in partes])
        por_clase = {}
        for c in range(3):
            m = cls == c
            cc, pp = cajas[m], punt[m]
            if len(cc):
                k = nms(cc, pp, umbral_nms)
                cc, pp = cc[k], pp[k]
            por_clase[c] = (cc, pp)
        salida.append(por_clase)
    return salida


def evaluar(modelo, filas, raiz, dev, lado):
    """AP@0.5 por (fuente, clase) en las clases etiquetadas de cada fuente."""
    preds = predecir_filas(modelo, filas, raiz, dev, lado)
    por_fuente = defaultdict(list)
    for f, p in zip(filas, preds):
        por_fuente[f["fuente"]].append((f, p))
    filas_res = []
    for fuente, items in sorted(por_fuente.items()):
        etiq = items[0][0]["etiquetadas"]
        for c in range(3):
            if not etiq[c]:
                continue
            gt = [np.array([b[:4] for b in f["cajas"] if b[4] == c], np.float32).reshape(-1, 4) for f, _ in items]
            pr = [p[c] for _, p in items]
            if sum(len(g) for g in gt) == 0:
                continue
            ap, _, _ = average_precision(pr, gt)
            filas_res.append({"fuente": fuente, "clase": CLASES[c], "AP50": ap, "n_imagenes": len(items),
                              "n_objetos": int(sum(len(g) for g in gt))})
    return pd.DataFrame(filas_res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=None)
    ap.add_argument("--muestras", type=int, default=None)
    ap.add_argument("--val-por-fuente", type=int, default=40)
    ap.add_argument("--trabajadores", type=int, default=3)
    a = ap.parse_args()
    cfg = cargar_config()
    cu = cfg["detector_universal"]
    fijar_semillas(cfg["semilla"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    raiz = ruta(cfg, "datos") / "procesados" / "universal"
    pesos, res = ruta(cfg, "pesos"), ruta(cfg, "resultados")
    filas = leer_json(raiz / "anotaciones.json")
    tr = [f for f in filas if f["particion"] == "train"]
    rnd = random.Random(cfg["semilla"])
    va = []
    for fuente in sorted({f["fuente"] for f in filas}):
        g = [f for f in filas if f["particion"] == "val" and f["fuente"] == fuente]
        rnd.shuffle(g)
        va += g[:a.val_por_fuente]
    print(f"train: {len(tr)} imágenes | val (subconjunto): {len(va)}")

    # muestreo ponderado por fuente
    cuenta = defaultdict(int)
    for f in tr:
        cuenta[f["fuente"]] += 1
    pesos_m = [cu["pesos_fuente"].get(f["fuente"], 0.05) / cuenta[f["fuente"]] for f in tr]
    n_muestras = a.muestras or cu["muestras_por_epoca"]
    sampler = WeightedRandomSampler(pesos_m, n_muestras, replacement=True)
    dl = DataLoader(DatasetUniversal(tr, raiz, cu), batch_size=cu["tam_lote"], sampler=sampler,
                    num_workers=a.trabajadores, persistent_workers=a.trabajadores > 0, pin_memory=True, drop_last=True)

    modelo = DetectorUniversal(tuple(cu["canales"]), cu["canal_cuello"]).to(dev)
    print(f"Parámetros: {contar_parametros(modelo):,}")
    crit = PerdidaDetectorUniversal()
    opt = torch.optim.AdamW(modelo.parameters(), lr=cu["lr"], weight_decay=cu["weight_decay"])
    epocas = a.epocas or cu["epocas"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cu["lr"], total_steps=epocas * len(dl), pct_start=0.1)
    escalador = torch.amp.GradScaler(enabled=dev.type == "cuda")

    historial, mejor, sin_mejora = [], -1.0, 0
    for ep in range(1, epocas + 1):
        modelo.train()
        t0, suma, partes = time.time(), 0.0, defaultdict(float)
        for it, lote in enumerate(dl):
            x, calor, ind, cls, tam, des, msk, etiq = [t.to(dev, non_blocking=True) for t in lote]
            x = x.float().div(255)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                s = modelo(x)
            perdida, d = crit(s, calor, ind, cls, tam, des, msk, etiq)
            opt.zero_grad(set_to_none=True)
            escalador.scale(perdida).backward()
            escalador.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 10.0)
            escalador.step(opt)
            escalador.update()
            sched.step()
            suma += perdida.item()
            for k, v in d.items():
                partes[k] += v
        n_it = len(dl)
        df_val = evaluar(modelo, va, raiz, dev, cu["lado_inferencia"])
        map_medio = float(df_val["AP50"].mean())
        fila = {"epoca": ep, "perdida": suma / n_it, **{f"perdida_{k}": v / n_it for k, v in partes.items()},
                "val_mAP50": map_medio, "segundos": time.time() - t0}
        for _, r in df_val.iterrows():
            fila[f"AP_{r['fuente']}_{r['clase']}"] = r["AP50"]
        historial.append(fila)
        print(f"época {ep:2d} | pérdida {fila['perdida']:.3f} (calor {fila['perdida_calor']:.3f}) | "
              f"val mAP50 {map_medio:.3f} | {fila['segundos']:.0f}s", flush=True)
        pd.DataFrame(historial).to_csv(res / "historial_detector_universal.csv", index=False)
        if map_medio > mejor:
            mejor, sin_mejora = map_medio, 0
            torch.save({"estado": modelo.state_dict(), "canales": list(cu["canales"]), "canal_cuello": cu["canal_cuello"],
                        "clases": list(CLASES), "epoca": ep, "val_mAP50": map_medio}, pesos / "detector_universal.pth")
        else:
            sin_mejora += 1
            if sin_mejora >= cu["paciencia"]:
                print(f"Parada temprana en la época {ep}")
                break

    h = pd.DataFrame(historial)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoca"], h["perdida"], label="total")
    ax[0].plot(h["epoca"], h["perdida_calor"], label="mapa de calor")
    ax[0].set(title="Pérdida de entrenamiento", xlabel="época")
    ax[0].legend()
    ax[1].plot(h["epoca"], h["val_mAP50"], "o-")
    ax[1].set(title="mAP@0.5 medio en validación", xlabel="época", ylim=(0, 1))
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res / "curvas_detector_universal.png", dpi=120)
    guardar_json({"mejor_val_mAP50": mejor, "epocas_entrenadas": len(historial),
                  "parametros": contar_parametros(modelo)}, res / "resumen_entrenamiento_detector_universal.json")
    print(f"Mejor val mAP50: {mejor:.4f}")


if __name__ == "__main__":
    main()
