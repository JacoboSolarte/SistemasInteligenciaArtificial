"""
Evaluación del Modelo 3 (TransformerAsistente):
  * accuracy y matriz de confusión en el test sintético (frases no vistas en entrenamiento)
  * accuracy en 40 preguntas escritas a mano (evaluacion/preguntas_manuales.csv), más difíciles y realistas

    python evaluacion/evaluar_asistente.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asistente.asistente import Asistente  # noqa: E402
from utilidades.comun import cargar_config, guardar_json, ruta  # noqa: E402

AQUI = Path(__file__).resolve().parent


def main():
    cfg = cargar_config()
    res = ruta(cfg, "resultados")
    asis = Asistente(ruta(cfg, "pesos") / "transformer_asistente.pth", cfg["asistente"]["umbral_intencion"])
    etiquetas = asis.intenciones
    salida = {}
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (nombre, df) in zip(axs, (("test sintético", pd.read_csv(ruta(cfg, "datos") / "procesados" / "asistente" / "test.csv")),
                                      ("40 preguntas manuales", pd.read_csv(AQUI / "preguntas_manuales.csv")))):
        pred = [asis.clasificar(t) for t in df.texto]
        df = df.assign(prediccion=[p[0] for p in pred], confianza=[p[1] for p in pred])
        acc = accuracy_score(df.intencion, df.prediccion)
        baja = (df.confianza < asis.umbral).mean()
        salida[nombre] = {"n": len(df), "accuracy": acc, "fraccion_confianza_baja": baja,
                          "reporte": classification_report(df.intencion, df.prediccion, labels=etiquetas, output_dict=True, zero_division=0)}
        print(f"\n=== {nombre}: accuracy = {acc:.4f} (n={len(df)}) | con confianza < {asis.umbral}: {100*baja:.1f} %")
        print(classification_report(df.intencion, df.prediccion, labels=etiquetas, zero_division=0))
        cm = confusion_matrix(df.intencion, df.prediccion, labels=etiquetas)
        ConfusionMatrixDisplay(cm, display_labels=etiquetas).plot(ax=ax, cmap="Purples", colorbar=False, xticks_rotation=45)
        ax.set_title(f"{nombre} — accuracy {acc:.3f}")
        sufijo = "sintetico" if "sint" in nombre else "manual"
        df.to_csv(res / f"asistente_predicciones_{sufijo}.csv", index=False)
        errores = df[df.intencion != df.prediccion]
        if len(errores):
            print("Errores:\n" + errores.head(15).to_string(index=False))
    fig.tight_layout()
    fig.savefig(res / "asistente_matrices_confusion.png", dpi=110)
    plt.close(fig)
    guardar_json(salida, res / "metricas_asistente.json")
    pd.DataFrame([{"conjunto": k, "n": v["n"], "accuracy": v["accuracy"], "confianza_baja": v["fraccion_confianza_baja"]}
                  for k, v in salida.items()]).to_csv(res / "metricas_asistente.csv", index=False)


if __name__ == "__main__":
    main()
