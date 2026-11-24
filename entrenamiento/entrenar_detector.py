"""
Entrena el Modelo 2 (MiniDetector) desde cero.

    python scripts/preparar_detector.py
    python entrenamiento/entrenar_detector.py

El modelo se selecciona por el mAP@0.5 de validación (calculado a mano en evaluacion/metricas_detector.py).
"""
import copy
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluacion.metricas_detector import average_precision, precision_recall_en_umbral  # noqa: E402
from inferencia.nms import postprocesar  # noqa: E402
from modelos.mini_detector import MiniDetector, PerdidaDetector, construir_objetivo, decodificar  # noqa: E402
from utilidades.comun import (cargar_config, contar_parametros, fijar_semillas, guardar_json, leer_json,  # noqa: E402
                              obtener_dispositivo, ruta)


def cargar_datos_detector(cfg):
    carpeta = ruta(cfg, "datos") / "procesados" / "detector"
    X = np.load(carpeta / "imagenes.npy", mmap_mode="r")
    etiquetas = leer_json(carpeta / "etiquetas.json")
    meta = pd.read_csv(carpeta / "metadatos.csv")
    meta = meta[meta.valido]
    part = {p: meta.loc[meta.particion == p, "idx"].to_numpy() for p in ("train", "val", "test")}
    return X, etiquetas, meta, part


def cxcywh_a_xyxy(cajas):
    c = np.asarray(cajas, dtype=np.float64).reshape(-1, 5)[:, 1:]
    return np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], 1)


def lote(X, etiquetas, idx, dev, aumentar=False, rng=None):
    idx = np.sort(idx)
    x = torch.from_numpy(np.ascontiguousarray(X[idx])).to(dev).permute(0, 3, 1, 2).float().div_(255)
    cajas = [[c[1:] for c in etiquetas[i]] for i in idx]
    if aumentar:
        n = len(idx)
        # brillo / contraste
        b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2) + m) * b
        # flip horizontal (también se reflejan las cajas: cx -> 1 - cx)
        voltear = rng.random(n) < 0.5
        for k in np.where(voltear)[0]:
            x[k] = x[k].flip(2)
            cajas[k] = [[1 - cx, cy, w, h] for cx, cy, w, h in cajas[k]]
        x = x.clamp_(0, 1)
    return x, cajas


@torch.no_grad()
def predecir(modelo, X, idx, dev, umbral_conf, umbral_nms, tam_lote=16):
    modelo.eval()
    """Devuelve una lista (por imagen, en orden de índice) de (cajas xyxy normalizadas, puntajes)."""
    salidas = []
    idx = np.sort(idx)
    for i in range(0, len(idx), tam_lote):
        sub = idx[i:i + tam_lote]
        x = torch.from_numpy(np.ascontiguousarray(X[sub])).to(dev).permute(0, 3, 1, 2).float().div_(255)
        punt, cajas = decodificar(modelo(x))
        for p, c in zip(punt.cpu().numpy(), cajas.cpu().numpy()):
            salidas.append(postprocesar(p, c, umbral_conf, umbral_nms))
    return salidas


def evaluar_conjunto(modelo, X, etiquetas, idx, dev, cd):
    idx = np.sort(idx)
    pred_map = predecir(modelo, X, idx, dev, 0.01, cd["umbral_nms"])        # umbral bajo para la curva PR
    gt = [cxcywh_a_xyxy(etiquetas[i]) for i in idx]
    ap, _, _ = average_precision([(c, p) for c, p in pred_map], gt)
    pred_op = [(c[p >= cd["umbral_confianza"]], p[p >= cd["umbral_confianza"]]) for c, p in pred_map]
    prec, rec, f1 = precision_recall_en_umbral(pred_op, gt)
    return {"mAP50": ap, "precision": prec, "recall": rec, "f1": f1}


def main():
    cfg = cargar_config()
    cd = cfg["detector"]
    fijar_semillas(cfg["semilla"])
    torch.backends.cudnn.benchmark = True       # más rápido para la red grande (sacrifica determinismo exacto)
    torch.backends.cudnn.deterministic = False
    dev = obtener_dispositivo()
    X, etiquetas, meta, part = cargar_datos_detector(cfg)
    print({k: len(v) for k, v in part.items()})
    modelo = MiniDetector(cd["cajas_por_celda"], cd["tam_imagen"]).to(dev)
    print(f"Parámetros del MiniDetector: {contar_parametros(modelo):,}  | cuadrícula {modelo.S}x{modelo.S}, B={modelo.B}")
    _, desc = construir_objetivo([[c[1:] for c in etiquetas[i]] for i in part["train"]], modelo.S, modelo.B)
    total = sum(len(etiquetas[i]) for i in part["train"])
    print(f"Cajas de train que no caben en su celda (descartadas): {desc}/{total} ({100*desc/max(total,1):.1f} %)")

    perdida = PerdidaDetector(cd["lambda_coord"], cd["lambda_noobj"])
    opt = torch.optim.Adam(modelo.parameters(), lr=cd["lr"], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cd["lr"], total_steps=cd["epocas"] * int(np.ceil(len(part["train"]) / cd["tam_lote"])),
                                                pct_start=0.15)
    usar_amp = dev.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)
    rng = np.random.default_rng(cfg["semilla"])
    hist, mejor, mejor_estado, sin = [], -1, None, 0
    for ep in range(1, cd["epocas"] + 1):
        t0 = time.time()
        modelo.train()
        perm = rng.permutation(part["train"])
        suma, partes, n = 0.0, {"coord": 0, "obj": 0, "noobj": 0}, 0
        for i in range(0, len(perm), cd["tam_lote"]):
            x, cajas = lote(X, etiquetas, perm[i:i + cd["tam_lote"]], dev, aumentar=True, rng=rng)
            obj, _ = construir_objetivo(cajas, modelo.S, modelo.B)
            with torch.autocast(device_type=dev.type, enabled=usar_amp):
                salida = modelo(x)
            loss, det = perdida(salida.float(), obj.to(dev))
            opt.zero_grad(set_to_none=True)
            escalador.scale(loss).backward()
            escalador.step(opt)
            escalador.update()
            sched.step()
            suma += loss.item() * len(cajas)
            for k in partes:
                partes[k] += det[k] * len(cajas)
            n += len(cajas)
        # pérdida de validación
        modelo.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for i in range(0, len(part["val"]), 16):
                x, cajas = lote(X, etiquetas, part["val"][i:i + 16], dev)
                obj, _ = construir_objetivo(cajas, modelo.S, modelo.B)
                vl += perdida(modelo(x).float(), obj.to(dev))[0].item() * len(cajas)
                vn += len(cajas)
        m = evaluar_conjunto(modelo, X, etiquetas, part["val"], dev, cd) if ep % 2 == 0 or ep == cd["epocas"] else None
        r = {"epoca": ep, "loss_train": suma / n, "loss_val": vl / vn, **{f"{k}_train": v / n for k, v in partes.items()},
             "mAP50_val": m["mAP50"] if m else np.nan, "segundos": time.time() - t0}
        hist.append(r)
        print(f"  época {ep:2d} | loss {r['loss_train']:.3f} (coord {r['coord_train']:.2f} obj {r['obj_train']:.2f} "
              f"noobj {r['noobj_train']:.2f}) | val loss {r['loss_val']:.3f}" + (f" | mAP@0.5 val {m['mAP50']:.4f} P {m['precision']:.3f} R {m['recall']:.3f}" if m else "")
              + f" | {r['segundos']:.0f}s")
        if m:
            if m["mAP50"] > mejor:
                mejor, mejor_estado, sin = m["mAP50"], copy.deepcopy(modelo.state_dict()), 0
            else:
                sin += 2
                if sin >= cd["paciencia"]:
                    print(f"  early stopping en la época {ep}")
                    break
    modelo.load_state_dict(mejor_estado)
    m_te = evaluar_conjunto(modelo, X, etiquetas, part["test"], dev, cd)
    print("TEST: " + "  ".join(f"{k}={v:.4f}" for k, v in m_te.items()))
    pesos, res = ruta(cfg, "pesos"), ruta(cfg, "resultados")
    torch.save({"estado": modelo.state_dict(), "B": modelo.B, "tam_imagen": modelo.tam_imagen, "metricas_test": m_te},
               pesos / "mini_detector.pth")
    h = pd.DataFrame(hist)
    h.to_csv(res / "historial_detector.csv", index=False)
    guardar_json({"mejor_mAP50_val": mejor, **{f"test_{k}": v for k, v in m_te.items()},
                  "parametros": contar_parametros(modelo)}, res / "metricas_detector_entrenamiento.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoca, h.loss_train, "--", label="train")
    ax[0].plot(h.epoca, h.loss_val, label="val")
    ax[0].set(title="Pérdida total del detector", xlabel="época")
    hv = h.dropna(subset=["mAP50_val"])
    ax[1].plot(hv.epoca, hv.mAP50_val, "-o", label="val")
    ax[1].set(title="mAP@0.5 (validación)", xlabel="época")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("MiniDetector")
    fig.tight_layout()
    fig.savefig(res / "curvas_detector.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
