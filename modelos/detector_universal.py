"""
Modelo 4: DetectorUniversal — detector de CENTROS (estilo "objetos como puntos") con 3 clases:
    0 = espacio libre      1 = espacio ocupado      2 = vehículo
Es el modelo del MODO AUTOMÁTICO: encuentra los espacios y los vehículos sin JSON de calibración.
Diseñado capa por capa y entrenado desde cero (sin pesos preentrenados).

Idea (explicable en la sustentación):
  * Backbone convolucional residual propio que reduce la imagen 32 veces (s2, s4, s8, s16, s32).
  * Cuello tipo "pirámide": se sube de nuevo la resolución (s32 -> s16 -> s8 -> s4) sumando los mapas
    del backbone (conexiones laterales), para combinar contexto global con detalle fino.
    La salida queda a 1/4 de la imagen: con entrada 512 -> mapa 128x128, suficiente para carros
    pequeños vistos desde un dron.
  * Tres cabezas sobre ese mapa:
      - mapa de calor [3 canales]: para cada clase, la probabilidad de que en ese píxel esté el
        CENTRO de un objeto (entrenado con "focal loss" y gaussianas alrededor de cada centro).
      - tamaño [3x2 canales]: log(ancho/4) y log(alto/4) de la caja, por clase (un espacio y un carro
        en el mismo punto pueden tener tamaños distintos).
      - desfase [2 canales]: corrección sub-píxel del centro (el mapa está a 1/4 de resolución).
  * Decodificación: los máximos locales del mapa de calor (max-pool 3x3) son las detecciones;
    no hace falta comparar cajas ancla. Luego un NMS propio limpia duplicados.
  * Es totalmente convolucional: en inferencia acepta cualquier tamaño múltiplo de 32.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASES = ("libre", "ocupado", "vehiculo")
PASO = 4  # resolución de salida = entrada / 4


def conv_bn(c_in, c_out, k=3, s=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False), nn.BatchNorm2d(c_out),
                         nn.LeakyReLU(0.1, inplace=True))


class BloqueResidual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = conv_bn(c, c)
        self.c2 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.leaky_relu(x + self.c2(self.c1(x)), 0.1)


def etapa(c_in, c_out, n_res):
    """Reduce a la mitad (conv con paso 2) y aplica n bloques residuales."""
    return nn.Sequential(conv_bn(c_in, c_out, 3, 2), *[BloqueResidual(c_out) for _ in range(n_res)])


class DetectorUniversal(nn.Module):
    def __init__(self, canales=(24, 48, 96, 160, 256), canal_cuello=64, n_clases=3):
        super().__init__()
        c1, c2, c3, c4, c5 = canales
        self.n_clases = n_clases
        self.e1 = etapa(3, c1, 0)      # 1/2
        self.e2 = etapa(c1, c2, 1)     # 1/4
        self.e3 = etapa(c2, c3, 2)     # 1/8
        self.e4 = etapa(c3, c4, 2)     # 1/16
        self.e5 = etapa(c4, c5, 2)     # 1/32
        # conexiones laterales 1x1 y suavizado 3x3 tras cada suma
        self.lat5, self.lat4 = nn.Conv2d(c5, canal_cuello, 1), nn.Conv2d(c4, canal_cuello, 1)
        self.lat3, self.lat2 = nn.Conv2d(c3, canal_cuello, 1), nn.Conv2d(c2, canal_cuello, 1)
        self.suave4, self.suave3, self.suave2 = (conv_bn(canal_cuello, canal_cuello) for _ in range(3))

        def cabeza(c_out):
            return nn.Sequential(conv_bn(canal_cuello, canal_cuello), nn.Conv2d(canal_cuello, c_out, 1))

        self.calor = cabeza(n_clases)
        self.tam = cabeza(2 * n_clases)
        self.desfase = cabeza(2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # al inicio casi ningún píxel es un centro: sesgo para p = 0.1 (estabiliza la focal loss)
        nn.init.constant_(self.calor[-1].bias, -math.log((1 - 0.1) / 0.1))
        nn.init.constant_(self.tam[-1].bias, math.log(8.0))  # tamaño inicial ~ 8 celdas = 32 px

    def forward(self, x):
        """x: [N,3,H,W] en [0,1] (H, W múltiplos de 32) -> dict con 'calor' (logits), 'tam', 'desfase'
        a resolución H/4 x W/4."""
        x = (x - 0.5) / 0.25
        f2 = self.e2(self.e1(x))
        f3 = self.e3(f2)
        f4 = self.e4(f3)
        f5 = self.e5(f4)
        p = self.lat5(f5)
        p = self.suave4(self.lat4(f4) + F.interpolate(p, scale_factor=2, mode="nearest"))
        p = self.suave3(self.lat3(f3) + F.interpolate(p, scale_factor=2, mode="nearest"))
        p = self.suave2(self.lat2(f2) + F.interpolate(p, scale_factor=2, mode="nearest"))
        return {"calor": self.calor(p), "tam": self.tam(p), "desfase": self.desfase(p)}


# ---------------------------------------------------------------------------------- objetivos
def radio_gaussiana(alto, ancho, solape_min=0.7):
    """Radio tal que una caja desplazada ese radio aún tenga IoU >= solape_min con la real (fórmula clásica
    de CornerNet/CenterNet). Se usa para el ancho de la gaussiana alrededor de cada centro."""
    a1, b1 = 1, alto + ancho
    c1 = ancho * alto * (1 - solape_min) / (1 + solape_min)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (alto + ancho)
    c2 = (1 - solape_min) * ancho * alto
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * solape_min, -2 * solape_min * (alto + ancho)
    c3 = (solape_min - 1) * ancho * alto
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return max(0, int(min(r1, r2, r3)))


def dibujar_gaussiana(mapa, cx, cy, radio):
    d = 2 * radio + 1
    sigma = d / 6
    y, x = np.ogrid[-radio:radio + 1, -radio:radio + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    alto, ancho = mapa.shape
    iz, de = min(cx, radio), min(ancho - cx, radio + 1)
    ar, ab = min(cy, radio), min(alto - cy, radio + 1)
    zona = mapa[cy - ar:cy + ab, cx - iz:cx + de]
    parche = g[radio - ar:radio + ab, radio - iz:radio + de]
    if min(zona.shape) > 0 and min(parche.shape) > 0:
        np.maximum(zona, parche, out=zona)


def construir_objetivo(cajas, alto_img, ancho_img, max_obj=512):
    """cajas: array [K,5] (x1,y1,x2,y2,clase) en píxeles de la imagen de entrada.
    Devuelve los tensores (numpy) que usa la pérdida."""
    ho, wo = alto_img // PASO, ancho_img // PASO
    calor = np.zeros((3, ho, wo), np.float32)
    ind = np.zeros(max_obj, np.int64)
    cls = np.zeros(max_obj, np.int64)
    tam = np.zeros((max_obj, 2), np.float32)
    des = np.zeros((max_obj, 2), np.float32)
    msk = np.zeros(max_obj, np.float32)
    k = 0
    for x1, y1, x2, y2, c in cajas:
        w, h = (x2 - x1) / PASO, (y2 - y1) / PASO
        if w < 0.5 or h < 0.5 or k >= max_obj:
            continue
        cx, cy = (x1 + x2) / 2 / PASO, (y1 + y2) / 2 / PASO
        ix, iy = min(int(cx), wo - 1), min(int(cy), ho - 1)
        dibujar_gaussiana(calor[int(c)], ix, iy, radio_gaussiana(math.ceil(h), math.ceil(w)))
        ind[k], cls[k] = iy * wo + ix, int(c)
        tam[k] = (math.log(w), math.log(h))
        des[k] = (cx - ix, cy - iy)
        msk[k] = 1
        k += 1
    return calor, ind, cls, tam, des, msk


# ---------------------------------------------------------------------------------- pérdida
class PerdidaDetectorUniversal(nn.Module):
    """Focal loss sobre el mapa de calor (con reducción de penalización cerca de los centros) +
    L1 sobre log-tamaño y desfase en los centros. `etiquetadas` [N,3] anula la pérdida de una clase
    en las imágenes cuya fuente no la etiqueta (p. ej. espacios en CARPK)."""

    def __init__(self, peso_tam=1.0, peso_desfase=1.0, alfa=2, beta=4):
        super().__init__()
        self.peso_tam, self.peso_desfase, self.alfa, self.beta = peso_tam, peso_desfase, alfa, beta

    def forward(self, salida, calor_obj, ind, cls, tam_obj, des_obj, msk, etiquetadas):
        n, c, h, w = salida["calor"].shape
        logit = salida["calor"].float()
        p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
        pos = calor_obj.eq(1).float()
        neg = 1 - pos
        m = etiquetadas.float().view(n, c, 1, 1)
        l_pos = -torch.log(p) * (1 - p) ** self.alfa * pos
        l_neg = -torch.log(1 - p) * p ** self.alfa * (1 - calor_obj) ** self.beta * neg
        n_pos = (pos * m).sum().clamp(min=1)
        perdida_calor = ((l_pos + l_neg) * m).sum() / n_pos

        # tamaño y desfase en los centros de los objetos
        tam = salida["tam"].float().view(n, c, 2, h * w)
        des = salida["desfase"].float().view(n, 2, h * w)
        idx = ind.clamp(max=h * w - 1)
        des_pred = des.gather(2, idx.unsqueeze(1).expand(n, 2, idx.shape[1])).permute(0, 2, 1)  # [N,K,2]
        tam_cls = tam.permute(0, 3, 1, 2).reshape(n, h * w, c * 2)  # [N,HW,C*2]
        tam_sel = tam_cls.gather(1, idx.unsqueeze(2).expand(n, idx.shape[1], c * 2))  # [N,K,C*2]
        tam_sel = tam_sel.view(n, idx.shape[1], c, 2).gather(2, cls.view(n, -1, 1, 1).expand(n, idx.shape[1], 1, 2)).squeeze(2)
        mk = msk.unsqueeze(2)
        n_obj = msk.sum().clamp(min=1)
        perdida_tam = (F.l1_loss(tam_sel, tam_obj, reduction="none") * mk).sum() / n_obj
        perdida_des = (F.l1_loss(des_pred, des_obj, reduction="none") * mk).sum() / n_obj
        total = perdida_calor + self.peso_tam * perdida_tam + self.peso_desfase * perdida_des
        return total, {"calor": perdida_calor.item(), "tam": perdida_tam.item(), "desfase": perdida_des.item()}


# ---------------------------------------------------------------------------------- decodificación
@torch.no_grad()
def decodificar(salida, k=600, umbral=0.05):
    """Devuelve, por imagen, (cajas [M,4] en píxeles de la entrada, puntajes [M], clases [M])."""
    calor = torch.sigmoid(salida["calor"].float())
    n, c, h, w = calor.shape
    picos = (F.max_pool2d(calor, 3, 1, 1) == calor).float() * calor  # máximos locales
    punt, idx = picos.view(n, -1).topk(min(k, c * h * w))
    clase = idx // (h * w)
    pix = idx % (h * w)
    ys, xs = (pix // w).float(), (pix % w).float()
    des = salida["desfase"].float().view(n, 2, -1)
    tam = salida["tam"].float().view(n, c, 2, -1)
    resultados = []
    for b in range(n):
        dx = des[b, 0, pix[b]]
        dy = des[b, 1, pix[b]]
        lw = tam[b, clase[b], 0, pix[b]]
        lh = tam[b, clase[b], 1, pix[b]]
        cx, cy = (xs[b] + dx) * PASO, (ys[b] + dy) * PASO
        bw, bh = torch.exp(lw.clamp(max=8)) * PASO, torch.exp(lh.clamp(max=8)) * PASO
        cajas = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        sel = punt[b] >= umbral
        resultados.append((cajas[sel].cpu().numpy(), punt[b][sel].cpu().numpy(), clase[b][sel].cpu().numpy()))
    return resultados
