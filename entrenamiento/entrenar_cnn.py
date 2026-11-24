"""
Entrenamiento del Modelo 1 (CNNEspacios) desde cero con gradiente descendente.

    python entrenamiento/entrenar_cnn.py                       # experimento general, compara SGD vs Adam
    python entrenamiento/entrenar_cnn.py --usar-mejor-config   # usa la mejor config de Hill Climbing / evolutiva
    python entrenamiento/entrenar_cnn.py --experimento parqueadero   # train UFPR04+UFPR05 -> test PUCPR
    python entrenamiento/entrenar_cnn.py --experimento clima         # train soleado -> test lluvioso

Gradiente descendente: en cada paso se calcula la pérdida (entropía cruzada) sobre un mini-lote,
loss.backward() obtiene el gradiente ∂L/∂θ por retropropagación, y el optimizador actualiza
    SGD con momentum:  v ← μ·v + ∇L ;  θ ← θ − η·v
    Adam:              usa promedios móviles del gradiente (m) y de su cuadrado (v) para adaptar η por parámetro.
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelos.cnn_espacios import crear_cnn  # noqa: E402
from utilidades.comun import (cargar_config, contar_parametros, fijar_semillas, guardar_json, leer_json,  # noqa: E402
                              obtener_dispositivo, ruta)


# ---------------------------------------------------------------------------
# Datos y particiones
# ---------------------------------------------------------------------------
def cargar_datos_cnn(cfg, en_memoria=True):
    carpeta = ruta(cfg, "datos") / "procesados" / "cnn"
    if not (carpeta / "recortes.npy").exists():
        raise FileNotFoundError(f"No existe {carpeta/'recortes.npy'}. Ejecuta primero scripts/preparar_cnn.py")
    X = np.load(carpeta / "recortes.npy", mmap_mode=None if en_memoria else "r")
    meta = pd.read_csv(carpeta / "metadatos.csv")
    return X, meta


def particiones(meta, experimento="general", semilla=42):
    """Devuelve dict con arreglos de índices (sobre X) para train / val / test."""
    rng = np.random.default_rng(semilla)

    def separar_val(idx, frac=0.15):
        idx = rng.permutation(idx)
        n = int(len(idx) * frac)
        return idx[n:], idx[:n]

    if experimento == "general":
        return {p: meta.loc[meta.particion == p, "idx"].to_numpy() for p in ("train", "val", "test")}
    if experimento == "parqueadero":
        pk = meta[meta.fuente == "pklot"]
        tr, va = separar_val(pk.loc[pk.parqueadero.isin(["UFPR04", "UFPR05"]), "idx"].to_numpy())
        return {"train": tr, "val": va, "test": pk.loc[pk.parqueadero == "PUCPR", "idx"].to_numpy()}
    if experimento == "clima":
        tr, va = separar_val(meta.loc[meta.clima == "soleado", "idx"].to_numpy())
        return {"train": tr, "val": va, "test": meta.loc[meta.clima == "lluvioso", "idx"].to_numpy()}
    raise ValueError(experimento)


def lote_a_tensor(X, idx, dispositivo):
    idx = np.sort(idx)
    x = torch.from_numpy(np.ascontiguousarray(X[idx])).to(dispositivo, non_blocking=True)
    return x.permute(0, 3, 1, 2).float().div_(255.0), idx


# ---------------------------------------------------------------------------
# Aumento de datos en GPU (brillo/contraste, flip, rotación pequeña, desenfoque)
# ---------------------------------------------------------------------------
def _kernel_gauss(sigma, tam=5, dispositivo="cpu"):
    ax = torch.arange(tam, device=dispositivo).float() - tam // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = torch.outer(g, g)
    return (k / k.sum()).expand(3, 1, tam, tam).contiguous()


def aumentar(x):
    n, dev = x.shape[0], x.device
    # brillo y contraste aleatorios (simulan sol fuerte, nubes o atardecer)
    brillo = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.65, 1.35)
    contraste = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    media = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - media) * contraste + media) * brillo
    # flip horizontal
    voltear = torch.rand(n, device=dev) < 0.5
    x = torch.where(voltear[:, None, None, None], x.flip(3), x)
    # rotación pequeña (±10°) con una matriz afín
    ang = torch.empty(n, device=dev).uniform_(-10, 10) * math.pi / 180
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    malla = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, malla, padding_mode="border", align_corners=False)
    # desenfoque gaussiano en ~30 % del lote (simula lluvia, desenfoque o baja resolución)
    desenf = torch.rand(n, device=dev) < 0.3
    if desenf.any():
        k = _kernel_gauss(float(np.random.uniform(0.6, 1.5)), 5, dev)
        borrosa = F.conv2d(F.pad(x[desenf], (2, 2, 2, 2), mode="replicate"), k, groups=3)
        x = x.clone()
        x[desenf] = borrosa
    return x.clamp_(0, 1)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
@torch.no_grad()
def predecir_probs(modelo, X, idx, dispositivo, tam_lote=1024):
    """Devuelve (y_real ordenado, P(ocupado)) — necesita las etiquetas desde fuera."""
    modelo.eval()
    idx = np.sort(np.asarray(idx))
    probs = []
    for i in range(0, len(idx), tam_lote):
        x, _ = lote_a_tensor(X, idx[i:i + tam_lote], dispositivo)
        probs.append(torch.softmax(modelo(x), 1)[:, 1].cpu())
    return idx, torch.cat(probs).numpy() if probs else np.zeros(0)


def metricas(y, prob, umbral=0.5):
    pred = (prob >= umbral).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0)}


def crear_optimizador(modelo, c):
    if c.get("optimizador", "adam") == "sgd":
        return torch.optim.SGD(modelo.parameters(), lr=c["lr"], momentum=c.get("momentum", 0.9),
                               weight_decay=c.get("weight_decay", 0), nesterov=True)
    return torch.optim.Adam(modelo.parameters(), lr=c["lr"], weight_decay=c.get("weight_decay", 0))


def entrenar_cnn(c, X, etiquetas, idx_train, idx_val, dispositivo, epocas=None, semilla=42, verbose=True,
                 tam_entrada=64):
    """Entrena una CNNEspacios con la configuración `c`. Devuelve (modelo con los mejores pesos, historial)."""
    fijar_semillas(semilla)
    modelo = crear_cnn(c, tam_entrada).to(dispositivo)
    opt = crear_optimizador(modelo, c)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=c.get("scheduler_factor", 0.5),
                                                       patience=c.get("scheduler_paciencia", 2))
    criterio = nn.CrossEntropyLoss()
    epocas = epocas or c["epocas"]
    tl = int(c["tam_lote"])
    y_tensor = torch.from_numpy(etiquetas).long()
    historial, mejor, mejor_estado, sin_mejora = [], float("inf"), None, 0
    rng = np.random.default_rng(semilla)
    for ep in range(1, epocas + 1):
        t0 = time.time()
        modelo.train()
        perm = rng.permutation(idx_train)
        suma_loss, aciertos, vistos = 0.0, 0, 0
        for i in range(0, len(perm), tl):
            x, idx = lote_a_tensor(X, perm[i:i + tl], dispositivo)
            y = y_tensor[idx].to(dispositivo)
            if c.get("aumento", True):
                x = aumentar(x)
            logits = modelo(x)
            loss = criterio(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()          # gradiente por retropropagación
            opt.step()               # paso de gradiente descendente
            suma_loss += loss.item() * len(y)
            aciertos += (logits.argmax(1) == y).sum().item()
            vistos += len(y)
        # validación
        modelo.eval()
        with torch.no_grad():
            vl, va, vn = 0.0, 0, 0
            for i in range(0, len(idx_val), 1024):
                x, idx = lote_a_tensor(X, idx_val[i:i + 1024], dispositivo)
                y = y_tensor[idx].to(dispositivo)
                lo = modelo(x)
                vl += criterio(lo, y).item() * len(y)
                va += (lo.argmax(1) == y).sum().item()
                vn += len(y)
        r = {"epoca": ep, "loss_train": suma_loss / vistos, "acc_train": aciertos / vistos,
             "loss_val": vl / vn, "acc_val": va / vn, "lr": opt.param_groups[0]["lr"], "segundos": time.time() - t0}
        historial.append(r)
        sched.step(r["loss_val"])
        if verbose:
            print(f"  época {ep:2d} | loss {r['loss_train']:.4f} acc {r['acc_train']:.4f} | "
                  f"val loss {r['loss_val']:.4f} acc {r['acc_val']:.4f} | lr {r['lr']:.1e} | {r['segundos']:.1f}s")
        if r["loss_val"] < mejor - 1e-4:
            mejor, mejor_estado, sin_mejora = r["loss_val"], copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= c.get("paciencia", 4):
                if verbose:
                    print(f"  early stopping en la época {ep}")
                break
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo, historial


def graficar_curvas(historiales: dict, archivo, titulo):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for nombre, h in historiales.items():
        h = pd.DataFrame(h)
        ax[0].plot(h.epoca, h.loss_train, "--", label=f"{nombre} train")
        ax[0].plot(h.epoca, h.loss_val, "-", label=f"{nombre} val")
        ax[1].plot(h.epoca, h.acc_train, "--", label=f"{nombre} train")
        ax[1].plot(h.epoca, h.acc_val, "-", label=f"{nombre} val")
    ax[0].set(title="Pérdida (entropía cruzada)", xlabel="época")
    ax[1].set(title="Accuracy", xlabel="época")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(archivo, dpi=120)
    plt.close(fig)


def guardar_modelo(modelo, c, archivo, extra=None):
    torch.save({"estado": modelo.state_dict(), "config_modelo": c, "tam_entrada": 64, **(extra or {})}, archivo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experimento", default="general", choices=["general", "parqueadero", "clima"])
    ap.add_argument("--optimizador", default="ambos", choices=["ambos", "adam", "sgd"])
    ap.add_argument("--usar-mejor-config", action="store_true",
                    help="usa resultados/mejor_config.json (salida de la optimización de hiperparámetros)")
    ap.add_argument("--epocas", type=int, default=None)
    a = ap.parse_args()

    cfg = cargar_config()
    dev = obtener_dispositivo()
    print(f"Dispositivo: {dev}")
    X, meta = cargar_datos_cnn(cfg)
    etiquetas = np.zeros(len(X), dtype=np.int64)
    etiquetas[meta.idx.to_numpy()] = meta.etiqueta.to_numpy()
    part = particiones(meta, a.experimento, cfg["semilla"])
    print({k: len(v) for k, v in part.items()})

    base = dict(cfg["cnn"])
    sufijo = ""
    if a.usar_mejor_config:
        mejor = leer_json(ruta(cfg, "resultados") / "mejor_config.json")
        base.update(mejor["config"])
        sufijo = "_optimizada"
        print(f"Usando la mejor configuración encontrada ({mejor['metodo']}): {mejor['config']}")

    opts = ["sgd", "adam"] if a.optimizador == "ambos" else [a.optimizador]
    if a.usar_mejor_config and a.optimizador == "ambos":
        opts = [base["optimizador"]]
    res_dir, pesos_dir = ruta(cfg, "resultados"), ruta(cfg, "pesos")
    historiales, resumen = {}, []
    for o in opts:
        c = dict(base, optimizador=o)
        if o == "sgd" and not a.usar_mejor_config:
            c["lr"] = 0.01          # SGD necesita un learning rate mayor que Adam
        print(f"\n=== Entrenando CNN ({a.experimento}) con {o.upper()}  lr={c['lr']} ===")
        modelo, hist = entrenar_cnn(c, X, etiquetas, part["train"], part["val"], dev, a.epocas, cfg["semilla"])
        idx_te, prob = predecir_probs(modelo, X, part["test"], dev)
        m = metricas(etiquetas[idx_te], prob)
        print(f"  TEST {a.experimento}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        nombre = f"{o}{sufijo}"
        historiales[nombre.upper()] = hist
        pd.DataFrame(hist).to_csv(res_dir / f"historial_cnn_{a.experimento}_{nombre}.csv", index=False)
        idx_va, prob_va = predecir_probs(modelo, X, part["val"], dev)
        f1_val = metricas(etiquetas[idx_va], prob_va)["f1"]
        resumen.append({"experimento": a.experimento, "optimizador": o, "config": c, "f1_val": f1_val,
                        "epocas": len(hist), "parametros": contar_parametros(modelo),
                        "tiempo_s": sum(h["segundos"] for h in hist), **{f"test_{k}": v for k, v in m.items()}})
        guardar_modelo(modelo, c, pesos_dir / f"cnn_{a.experimento}_{nombre}.pth", {"metricas_test": m})

    graficar_curvas(historiales, res_dir / f"curvas_cnn_{a.experimento}{sufijo}.png",
                    f"CNNEspacios — experimento '{a.experimento}'" + (" (config optimizada)" if sufijo else ""))
    df = pd.DataFrame(resumen)
    df.drop(columns="config").to_csv(res_dir / f"resumen_cnn_{a.experimento}{sufijo}.csv", index=False)
    # el mejor (por F1 de VALIDACIÓN, nunca por test) del experimento general se copia como modelo de producción
    if a.experimento == "general":
        mejor = df.sort_values("f1_val", ascending=False).iloc[0]
        nombre = f"{mejor['optimizador']}{sufijo}"
        ck = torch.load(pesos_dir / f"cnn_general_{nombre}.pth", weights_only=False)
        torch.save(ck, pesos_dir / "cnn_espacios.pth")
        guardar_json({"origen": f"cnn_general_{nombre}.pth", "f1_val": float(mejor["f1_val"]),
                      "test_f1": float(mejor["test_f1"])}, res_dir / "modelo_produccion_cnn.json")
        print(f"\nModelo de producción: pesos/cnn_espacios.pth  <- cnn_general_{nombre}.pth")
    print(df.drop(columns="config").to_string(index=False))


if __name__ == "__main__":
    main()
