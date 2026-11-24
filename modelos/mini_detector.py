"""
Modelo 2: MiniDetector — detector de vehículos de una etapa basado en una cuadrícula S x S.

Idea (explicable en la sustentación):
  * La imagen (416x416) pasa por un backbone convolucional propio que la reduce 32 veces -> mapa 13x13.
  * Cada celda de la cuadrícula "es responsable" de los vehículos cuyo centro cae dentro de ella.
  * Una convolución 1x1 produce, por celda y por cada una de las B cajas:
        [objetividad, x, y, w, h]
    - objetividad: logit de "hay un vehículo aquí" (se pasa por sigmoide)
    - x, y: posición del centro DENTRO de la celda, en [0,1] (sigmoide)
    - w, h: raíz cuadrada del ancho/alto relativo a la imagen, en [0,1] (sigmoide)
      (predecir la raíz hace que un error en cajas pequeñas pese más que en cajas grandes)
  * La pérdida (escrita a mano) combina BCE para la objetividad y MSE para las coordenadas.
  * El postproceso (umbral + NMS propio) está en inferencia/nms.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bloque_conv(c_in, c_out, k=3, pool=True):
    capas = [nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
             nn.BatchNorm2d(c_out),
             nn.LeakyReLU(0.1, inplace=True)]
    if pool:
        capas.append(nn.MaxPool2d(2))
    return nn.Sequential(*capas)


class MiniDetector(nn.Module):
    def __init__(self, cajas_por_celda: int = 2, tam_imagen: int = 416):
        super().__init__()
        self.B = cajas_por_celda
        self.tam_imagen = tam_imagen
        # Backbone: 416 -> 208 -> 104 -> 52 -> 26 -> 13
        self.backbone = nn.Sequential(
            bloque_conv(3, 16),
            bloque_conv(16, 32),
            bloque_conv(32, 64),
            bloque_conv(64, 128),
            bloque_conv(128, 256),
            bloque_conv(256, 512, pool=False),
            bloque_conv(512, 256, k=1, pool=False),
            bloque_conv(256, 512, pool=False),
        )
        self.S = tam_imagen // 32
        self.cabeza = nn.Conv2d(512, self.B * 5, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
        nn.init.zeros_(self.cabeza.bias)
        # sesgo inicial negativo en la objetividad: al inicio casi ninguna celda tiene vehículo
        with torch.no_grad():
            self.cabeza.bias[0::5] = -4.0

    def forward(self, x):
        """x: [N,3,416,416] en [0,1]  ->  salida cruda [N, S, S, B, 5]."""
        x = (x - 0.5) / 0.5
        y = self.cabeza(self.backbone(x))                    # [N, B*5, S, S]
        n, _, s, _ = y.shape
        return y.view(n, self.B, 5, s, s).permute(0, 3, 4, 1, 2).contiguous()


def decodificar(salida: torch.Tensor):
    """Convierte la salida cruda en (puntaje, cajas xyxy normalizadas [0,1]) por cada predictor.
    Devuelve puntajes [N, S*S*B] y cajas [N, S*S*B, 4]."""
    n, s, _, b, _ = salida.shape
    obj = torch.sigmoid(salida[..., 0])
    xy = torch.sigmoid(salida[..., 1:3])
    wh = torch.sigmoid(salida[..., 3:5]) ** 2
    filas = torch.arange(s, device=salida.device).view(1, s, 1, 1).float()
    cols = torch.arange(s, device=salida.device).view(1, 1, s, 1).float()
    cx = (cols + xy[..., 0]) / s
    cy = (filas + xy[..., 1]) / s
    cajas = torch.stack([cx - wh[..., 0] / 2, cy - wh[..., 1] / 2,
                         cx + wh[..., 0] / 2, cy + wh[..., 1] / 2], dim=-1)
    return obj.view(n, -1), cajas.view(n, -1, 4).clamp(0, 1)


def construir_objetivo(cajas_por_imagen, S: int, B: int):
    """Convierte listas de cajas [cx,cy,w,h] normalizadas en el tensor objetivo [N,S,S,B,5].
    Cada caja se asigna a la celda donde cae su centro. Si en una celda hay varias, se ordenan por x
    y la k-ésima va al predictor k (las que excedan B se descartan y se cuentan)."""
    n = len(cajas_por_imagen)
    obj = torch.zeros(n, S, S, B, 5)
    descartadas = 0
    for i, cajas in enumerate(cajas_por_imagen):
        celdas = {}
        for cx, cy, w, h in cajas:
            fila, col = min(int(cy * S), S - 1), min(int(cx * S), S - 1)
            celdas.setdefault((fila, col), []).append((cx, cy, w, h))
        for (fila, col), lista in celdas.items():
            lista.sort(key=lambda c: c[0])
            descartadas += max(0, len(lista) - B)
            for k, (cx, cy, w, h) in enumerate(lista[:B]):
                obj[i, fila, col, k] = torch.tensor([1.0, cx * S - col, cy * S - fila, w ** 0.5, h ** 0.5])
    return obj, descartadas


class PerdidaDetector(nn.Module):
    """Pérdida escrita a mano:
        L = λ_coord * MSE(coords | hay objeto) + BCE(obj | hay objeto) + λ_noobj * BCE(obj | no hay objeto)
    """

    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lc, self.ln = lambda_coord, lambda_noobj

    def forward(self, salida, objetivo):
        n = salida.shape[0]
        con_obj = objetivo[..., 0] == 1
        sin_obj = ~con_obj
        bce = F.binary_cross_entropy_with_logits(salida[..., 0], objetivo[..., 0], reduction="none")
        p_obj = bce[con_obj].sum()
        p_noobj = bce[sin_obj].sum()
        pred_coord = torch.sigmoid(salida[..., 1:5])[con_obj]
        p_coord = F.mse_loss(pred_coord, objetivo[..., 1:5][con_obj], reduction="sum")
        total = (self.lc * p_coord + p_obj + self.ln * p_noobj) / n
        return total, {"coord": p_coord.item() / n, "obj": p_obj.item() / n, "noobj": p_noobj.item() / n}
