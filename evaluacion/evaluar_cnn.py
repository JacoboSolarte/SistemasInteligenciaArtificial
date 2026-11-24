"""
Evaluación del Modelo 1 (CNNEspacios):
  * conjunto de prueba general: accuracy, precisión, recall, F1, matriz de confusión y curva ROC
  * desglose por fuente y por clima
  * generalización: por parqueadero (train UFPR04+UFPR05 -> test PUCPR) y por clima (train soleado -> test lluvioso)

    python evaluacion/evaluar_cnn.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from entrenamiento.entrenar_cnn import cargar_datos_cnn, metricas, particiones, predecir_probs  # noqa: E402
from modelos.cnn_espacios import cargar_cnn  # noqa: E402
from utilidades.comun import cargar_config, obtener_dispositivo, ruta  # noqa: E402


def graficar_cm_roc(y, prob, titulo, archivo):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    cm = confusion_matrix(y, (prob >= 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["libre", "ocupado"]).plot(ax=ax[0], cmap="Blues", values_format="d", colorbar=False)
    ax[0].set_title("Matriz de confusión")
    fpr, tpr, _ = roc_curve(y, prob)
    ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.4f}")
    ax[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax[1].set(title="Curva ROC", xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(archivo, dpi=120)
    plt.close(fig)
    return cm, auc(fpr, tpr)


def main():
    cfg = cargar_config()
    dev = obtener_dispositivo()
    res, pesos = ruta(cfg, "resultados"), ruta(cfg, "pesos")
    X, meta = cargar_datos_cnn(cfg)
    y_all = np.zeros(len(X), dtype=np.int64)
    y_all[meta.idx.to_numpy()] = meta.etiqueta.to_numpy()
    meta_i = meta.set_index("idx")
    filas = []

    # ---------------- Prueba general con el modelo de producción
    modelo = cargar_cnn(pesos / "cnn_espacios.pth", dev)
    part = particiones(meta, "general", cfg["semilla"])
    idx, prob = predecir_probs(modelo, X, part["test"], dev)
    y = y_all[idx]
    cm, a = graficar_cm_roc(y, prob, "CNNEspacios — conjunto de prueba (70/15/15)", res / "cnn_test_cm_roc.png")
    filas.append({"evaluacion": "test general", "n": len(y), **metricas(y, prob), "auc": a,
                  "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]})
    sub = meta_i.loc[idx]
    for col in ("fuente", "clima"):
        for val in sorted(sub[col].unique()):
            m = (sub[col] == val).to_numpy()
            filas.append({"evaluacion": f"test general | {col}={val}", "n": int(m.sum()), **metricas(y[m], prob[m])})
    pk = (sub.fuente == "pklot").to_numpy()
    for val in sorted(sub[pk].parqueadero.unique()):
        m = pk & (sub.parqueadero == val).to_numpy()
        filas.append({"evaluacion": f"test general | parqueadero={val}", "n": int(m.sum()), **metricas(y[m], prob[m])})

    # ---------------- Generalización
    for exp, desc in (("parqueadero", "train UFPR04+UFPR05 -> test PUCPR"), ("clima", "train soleado -> test lluvioso")):
        archivo_res = res / f"resumen_cnn_{exp}.csv"
        if not archivo_res.exists():
            print(f"[aviso] falta entrenar el experimento '{exp}' (python entrenamiento/entrenar_cnn.py --experimento {exp})")
            continue
        r = pd.read_csv(archivo_res).sort_values("f1_val", ascending=False).iloc[0]
        m_exp = cargar_cnn(pesos / f"cnn_{exp}_{r['optimizador']}.pth", dev)
        p = particiones(meta, exp, cfg["semilla"])
        idx, prob = predecir_probs(m_exp, X, p["test"], dev)
        cm, a = graficar_cm_roc(y_all[idx], prob, f"Generalización por {exp}: {desc}", res / f"cnn_generalizacion_{exp}_cm_roc.png")
        filas.append({"evaluacion": f"generalización {exp} ({desc})", "n": len(idx), **metricas(y_all[idx], prob), "auc": a,
                      "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]})
        # comparación: el modelo general sobre ese mismo conjunto (ojo: el general SÍ vio esos dominios en train)
        idx2, prob2 = predecir_probs(modelo, X, np.intersect1d(p["test"], part["test"]), dev)
        filas.append({"evaluacion": f"referencia: modelo general en test de {exp} (dominio visto)", "n": len(idx2),
                      **metricas(y_all[idx2], prob2)})

    df = pd.DataFrame(filas)
    df.to_csv(res / "metricas_cnn.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
