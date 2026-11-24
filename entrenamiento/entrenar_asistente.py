"""
Entrena el Modelo 3 (TransformerAsistente) desde cero sobre el dataset sintético de intenciones.

    python asistente/generar_dataset.py
    python entrenamiento/entrenar_asistente.py
"""
import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asistente.generar_dataset import INTENCIONES  # noqa: E402
from asistente.tokenizador import Tokenizador  # noqa: E402
from modelos.transformer_asistente import TransformerAsistente  # noqa: E402
from utilidades.comun import cargar_config, contar_parametros, fijar_semillas, obtener_dispositivo, ruta  # noqa: E402


def a_tensores(df, tok):
    X = torch.tensor([tok.codificar(t) for t in df.texto])
    y = torch.tensor([INTENCIONES.index(i) for i in df.intencion])
    return X, y


def main():
    cfg = cargar_config()
    ca = cfg["asistente"]
    fijar_semillas(cfg["semilla"])
    dev = obtener_dispositivo()
    carpeta = ruta(cfg, "datos") / "procesados" / "asistente"
    tr, va, te = (pd.read_csv(carpeta / f"{n}.csv") for n in ("train", "val", "test"))
    tok = Tokenizador(max_tokens=ca["max_tokens"]).construir(tr.texto.tolist())   # vocabulario SOLO del train
    print(f"Vocabulario: {len(tok)} tokens | train={len(tr)} val={len(va)} test={len(te)}")
    Xtr, ytr = a_tensores(tr, tok)
    Xva, yva = a_tensores(va, tok)
    Xte, yte = a_tensores(te, tok)

    modelo = TransformerAsistente(len(tok), len(INTENCIONES), ca["d_modelo"], ca["cabezas"], ca["capas"], ca["d_ff"],
                                  ca["dropout"], ca["max_tokens"]).to(dev)
    print(f"Parámetros del Transformer: {contar_parametros(modelo):,}")
    opt = torch.optim.AdamW(modelo.parameters(), lr=ca["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ca["epocas"])
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    hist, mejor, mejor_estado, sin = [], -1, None, 0
    g = torch.Generator().manual_seed(cfg["semilla"])
    for ep in range(1, ca["epocas"] + 1):
        modelo.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tl, ta = 0.0, 0
        for i in range(0, len(perm), ca["tam_lote"]):
            b = perm[i:i + ca["tam_lote"]]
            x, y = Xtr[b].to(dev), ytr[b].to(dev)
            lo = modelo(x)
            loss = crit(lo, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(b)
            ta += (lo.argmax(1) == y).sum().item()
        sched.step()
        modelo.eval()
        with torch.no_grad():
            lo = modelo(Xva.to(dev))
            vl = crit(lo, yva.to(dev)).item()
            vacc = (lo.argmax(1).cpu() == yva).float().mean().item()
        hist.append({"epoca": ep, "loss_train": tl / len(Xtr), "acc_train": ta / len(Xtr), "loss_val": vl, "acc_val": vacc})
        print(f"  época {ep:2d} | loss {tl/len(Xtr):.4f} acc {ta/len(Xtr):.4f} | val loss {vl:.4f} acc {vacc:.4f}")
        if vacc > mejor:
            mejor, mejor_estado, sin = vacc, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin += 1
            if sin >= ca["paciencia"]:
                print(f"  early stopping en la época {ep}")
                break
    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        acc_te = (modelo(Xte.to(dev)).argmax(1).cpu() == yte).float().mean().item()
    print(f"Accuracy en test (frases sintéticas no vistas): {acc_te:.4f}")

    pesos, res = ruta(cfg, "pesos"), ruta(cfg, "resultados")
    torch.save({"estado": modelo.state_dict(), "vocab": tok.vocab, "intenciones": INTENCIONES,
                "config": {k: ca[k] for k in ("d_modelo", "cabezas", "capas", "d_ff", "max_tokens")}},
               pesos / "transformer_asistente.pth")
    tok.guardar(pesos / "tokenizador_asistente.json")
    h = pd.DataFrame(hist)
    h.to_csv(res / "historial_asistente.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoca, h.loss_train, "--", label="train")
    ax[0].plot(h.epoca, h.loss_val, label="val")
    ax[1].plot(h.epoca, h.acc_train, "--", label="train")
    ax[1].plot(h.epoca, h.acc_val, label="val")
    ax[0].set(title="Pérdida", xlabel="época")
    ax[1].set(title="Accuracy de intención", xlabel="época")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("TransformerAsistente")
    fig.tight_layout()
    fig.savefig(res / "curvas_asistente.png", dpi=120)
    plt.close(fig)
    print("Pesos guardados en pesos/transformer_asistente.pth")


if __name__ == "__main__":
    main()
