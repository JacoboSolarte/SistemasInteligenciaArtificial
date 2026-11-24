"""
Evaluación en video: compara el sistema completo (pipeline con suavizado temporal) contra el estado real
de los espacios en algunos frames (evaluacion/anotaciones_manuales.json) y reporta el % de acierto por frame.

    python evaluacion/evaluar_video.py
    python evaluacion/evaluar_video.py --suavizado 1      # sin suavizado, para comparar
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferencia.pipeline import AnalizadorParqueadero  # noqa: E402
from inferencia.recorte import cargar_espacios  # noqa: E402
from utilidades.comun import RAIZ, cargar_config, guardar_json, leer_json, ruta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anotaciones", default=str(RAIZ / "evaluacion" / "anotaciones_manuales.json"))
    ap.add_argument("--suavizado", type=int, default=None)
    ap.add_argument("--detector", action="store_true")
    a = ap.parse_args()
    cfg = cargar_config()
    res, pesos = ruta(cfg, "resultados"), ruta(cfg, "pesos")
    anot = leer_json(a.anotaciones)
    espacios = cargar_espacios(RAIZ / anot["espacios"])
    analizador = AnalizadorParqueadero(pesos / "cnn_espacios.pth", pesos / "mini_detector.pth", cfg, usar_detector=a.detector)
    filas = []
    for n in sorted({1, a.suavizado or cfg["inferencia"]["frames_suavizado"]}):
        sufijo = "" if n > 1 else "_sin_suavizado"
        _, historial = analizador.analizar_video(RAIZ / anot["video"], espacios, res / f"video_anotado{sufijo}.mp4", n_suavizado=n, cada_k=1)
        por_frame = {h["frame"]: h for h in historial}
        for k, info in anot["frames"].items():
            h = por_frame.get(int(k))
            if h is None:
                continue
            pred = {r["id"]: r["estado"] for r in h["espacios"]}
            ids = [i for i in info["estado"] if i in pred]
            aciertos = sum(pred[i] == info["estado"][i] for i in ids)
            real_ocup = sum(info["estado"][i] == "ocupado" for i in ids)
            filas.append({"suavizado_N": n, "frame": int(k), "imagen": info["imagen"], "espacios": len(ids),
                          "acierto_%": 100 * aciertos / len(ids), "ocupados_real": real_ocup,
                          "ocupados_sistema": sum(pred[i] == "ocupado" for i in ids),
                          "error_conteo": abs(real_ocup - sum(pred[i] == "ocupado" for i in ids))})
    df = pd.DataFrame(filas)
    df.to_csv(res / "evaluacion_video.csv", index=False)
    resumen = df.groupby("suavizado_N").agg(acierto_promedio=("acierto_%", "mean"), acierto_min=("acierto_%", "min"),
                                            error_conteo_promedio=("error_conteo", "mean"), frames=("frame", "count")).reset_index()
    resumen.to_csv(res / "evaluacion_video_resumen.csv", index=False)
    guardar_json(resumen.to_dict(orient="records"), res / "evaluacion_video_resumen.json")
    print(df.to_string(index=False))
    print(resumen.round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for n, g in df.groupby("suavizado_N"):
        ax[0].plot(g.frame, g["acierto_%"], "-o", label=f"N={n}")
        ax[1].plot(g.frame, g.ocupados_sistema, "-o", label=f"sistema N={n}")
    g = df[df.suavizado_N == df.suavizado_N.min()]
    ax[1].plot(g.frame, g.ocupados_real, "k--", lw=2, label="real (XML)")
    ax[0].set(title="Acierto por frame (%)", xlabel="frame", ylim=(min(80, df["acierto_%"].min() - 2), 100.5))
    ax[1].set(title="Espacios ocupados: sistema vs real", xlabel="frame")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend()
    fig.tight_layout()
    fig.savefig(res / "evaluacion_video.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
