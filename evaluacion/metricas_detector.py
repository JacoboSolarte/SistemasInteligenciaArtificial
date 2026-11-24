"""Métricas del detector implementadas a mano: precisión, recall y mAP@0.5 (AP de la única clase 'vehículo')."""
import numpy as np

from inferencia.nms import iou_cajas


def emparejar(pred_cajas, pred_puntajes, gt_cajas, umbral_iou=0.5):
    """Para una imagen: marca cada predicción como TP (1) o FP (0), en orden de puntaje descendente.
    Cada caja real solo puede emparejarse una vez."""
    orden = np.argsort(-np.asarray(pred_puntajes))
    usado = np.zeros(len(gt_cajas), dtype=bool)
    tp = np.zeros(len(orden))
    for k, i in enumerate(orden):
        if len(gt_cajas) == 0:
            break
        ious = iou_cajas(pred_cajas[i], gt_cajas)
        ious[usado] = -1
        j = int(np.argmax(ious))
        if ious[j] >= umbral_iou:
            tp[k], usado[j] = 1, True
    return np.asarray(pred_puntajes)[orden], tp


def average_precision(lista_pred, lista_gt, umbral_iou=0.5):
    """lista_pred: [(cajas[M,4], puntajes[M])] por imagen ; lista_gt: [cajas[G,4]] por imagen.
    Devuelve AP (área bajo la curva precisión-recall con interpolación de todos los puntos), precisión, recall."""
    puntajes, tps = [], []
    n_gt = sum(len(g) for g in lista_gt)
    for (pc, pp), g in zip(lista_pred, lista_gt):
        s, t = emparejar(np.asarray(pc).reshape(-1, 4), pp, np.asarray(g).reshape(-1, 4), umbral_iou)
        puntajes.append(s)
        tps.append(t)
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])
    puntajes, tps = np.concatenate(puntajes), np.concatenate(tps)
    orden = np.argsort(-puntajes)
    tps = tps[orden]
    tp_acum, fp_acum = np.cumsum(tps), np.cumsum(1 - tps)
    recall = tp_acum / n_gt
    precision = tp_acum / np.maximum(tp_acum + fp_acum, 1e-12)
    # interpolación: precisión monótonamente decreciente
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    cambios = np.where(r[1:] != r[:-1])[0]
    ap = float(np.sum((r[cambios + 1] - r[cambios]) * p[cambios + 1]))
    return ap, precision, recall


def precision_recall_en_umbral(lista_pred, lista_gt, umbral_iou=0.5):
    tp = fp = 0
    n_gt = sum(len(g) for g in lista_gt)
    for (pc, pp), g in zip(lista_pred, lista_gt):
        _, t = emparejar(np.asarray(pc).reshape(-1, 4), pp, np.asarray(g).reshape(-1, 4), umbral_iou)
        tp += t.sum()
        fp += len(t) - t.sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return float(prec), float(rec), float(f1)
