"""
Evaluación del Modelo 2 (MiniDetector): precisión, recall, mAP@0.5 (a mano), curva PR, métricas por
parqueadero y ejemplos visuales de aciertos y errores.

    python evaluacion/evaluar_detector.py
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entrenamiento.entrenar_detector import cargar_datos_detector, cxcywh_a_xyxy, predecir  # noqa: E402
from evaluacion.metricas_detector import average_precision, precision_recall_en_umbral  # noqa: E402
from modelos.mini_detector import MiniDetector  # noqa: E402
from utilidades.comun import cargar_config, guardar_json, obtener_dispositivo, ruta  # noqa: E402


def main():
    cfg = cargar_config()
    cd = cfg["detector"]
    dev = obtener_dispositivo()
    res, pesos = ruta(cfg, "resultados"), ruta(cfg, "pesos")
    X, etiquetas, meta, part = cargar_datos_detector(cfg)
    ck = torch.load(pesos / "mini_detector.pth", map_location=dev, weights_only=False)
    modelo = MiniDetector(ck["B"], ck["tam_imagen"]).to(dev)
    modelo.load_state_dict(ck["estado"])
    idx = np.sort(part["test"])
    pred = predecir(modelo, X, idx, dev, 0.01, cd["umbral_nms"])
    gt = [cxcywh_a_xyxy(etiquetas[i]) for i in idx]
    ap, prec_curva, rec_curva = average_precision(pred, gt)
    op = [(c[p >= cd["umbral_confianza"]], p[p >= cd["umbral_confianza"]]) for c, p in pred]
    P, R, F1 = precision_recall_en_umbral(op, gt)
    filas = [{"conjunto": "test", "n_imagenes": len(idx), "mAP50": ap, "precision": P, "recall": R, "f1": F1}]
    meta_i = meta.set_index("idx")
    for parq in ("UFPR04", "UFPR05", "PUCPR"):
        m = (meta_i.loc[idx, "parqueadero"] == parq).to_numpy()
        ap_p, _, _ = average_precision([pred[i] for i in np.where(m)[0]], [gt[i] for i in np.where(m)[0]])
        P_p, R_p, F_p = precision_recall_en_umbral([op[i] for i in np.where(m)[0]], [gt[i] for i in np.where(m)[0]])
        filas.append({"conjunto": f"test {parq}", "n_imagenes": int(m.sum()), "mAP50": ap_p, "precision": P_p, "recall": R_p, "f1": F_p})
    df = pd.DataFrame(filas)
    df.to_csv(res / "metricas_detector.csv", index=False)
    guardar_json(filas[0], res / "metricas_detector.json")
    print(df.round(4).to_string(index=False))

    # Curva precisión-recall
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec_curva, prec_curva, lw=2)
    ax.set(title=f"Curva PR del MiniDetector (AP@0.5 = {ap:.3f})", xlabel="recall", ylabel="precisión", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res / "detector_curva_pr.png", dpi=120)
    plt.close(fig)

    # Ejemplos visuales: las 3 mejores y las 3 peores imágenes por F1
    # F1 por imagen; una imagen sin vehículos y sin predicciones es un acierto perfecto (F1 = 1)
    f1s = [1.0 if len(g) == 0 and len(o[0]) == 0 else precision_recall_en_umbral([o], [g])[2] for o, g in zip(op, gt)]
    orden = np.argsort(f1s)
    elegidas = list(orden[-3:][::-1]) + list(orden[:3])
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for ax, k in zip(axs.ravel(), elegidas):
        img = np.ascontiguousarray(X[idx[k]]).copy()
        t = img.shape[0]
        for c in gt[k]:
            x1, y1, x2, y2 = (c * t).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for c in op[k][0]:
            x1, y1, x2, y2 = (c * t).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        ax.imshow(img)
        ax.set_title(f"{'ACIERTO' if k in orden[-3:] else 'ERROR'} — {meta_i.loc[idx[k], 'parqueadero']} "
                     f"F1={f1s[k]:.2f}\nverde: caja real | rojo: predicción", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(res / "detector_ejemplos.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
