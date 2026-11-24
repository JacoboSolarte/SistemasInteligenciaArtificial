"""
Modelo 1: CNNEspacios — clasificador de un espacio de parqueo en {0: libre, 1: ocupado}.

Entrada: recorte RGB 64x64 (tensor float en [0,1], forma [N, 3, 64, 64]).
Arquitectura (configurable):
    Bloque k: Conv(3x3, padding=1) -> BatchNorm -> ReLU -> MaxPool(2)   (3 bloques: 64->32->16->8)
    Flatten -> Linear(->neuronas) -> ReLU -> Dropout -> Linear(neuronas->2)
Todos los pesos se inicializan aleatoriamente (inicialización de Kaiming) y se entrenan desde cero.
"""
import torch
import torch.nn as nn

CLASES = ["libre", "ocupado"]


class CNNEspacios(nn.Module):
    def __init__(self, filtros=(16, 32, 64), neuronas_densa=128, dropout=0.5, tam_entrada=64):
        super().__init__()
        bloques = []
        canales_entrada = 3
        for canales in filtros:
            bloques += [
                nn.Conv2d(canales_entrada, canales, kernel_size=3, padding=1),
                nn.BatchNorm2d(canales),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            canales_entrada = canales
        self.extractor = nn.Sequential(*bloques)
        lado = tam_entrada // (2 ** len(filtros))          # 64 / 8 = 8
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(canales_entrada * lado * lado, neuronas_densa),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(neuronas_densa, len(CLASES)),
        )
        self._inicializar()

    def _inicializar(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalización simple: [0,1] -> [-1,1]
        x = (x - 0.5) / 0.5
        return self.clasificador(self.extractor(x))   # logits [N, 2]

    @torch.no_grad()
    def predecir(self, x):
        """Devuelve (clase, confianza, prob_ocupado) para un lote."""
        self.eval()
        prob = torch.softmax(self.forward(x), dim=1)
        conf, clase = prob.max(dim=1)
        return clase, conf, prob[:, 1]


def crear_cnn(cfg_cnn: dict, tam_entrada: int = 64) -> CNNEspacios:
    """Construye la CNN a partir del diccionario de hiperparámetros (sección 'cnn' del YAML)."""
    return CNNEspacios(
        filtros=tuple(cfg_cnn.get("filtros", (16, 32, 64))),
        neuronas_densa=int(cfg_cnn.get("neuronas_densa", 128)),
        dropout=float(cfg_cnn.get("dropout", 0.5)),
        tam_entrada=tam_entrada,
    )


def cargar_cnn(ruta_pesos, dispositivo="cpu") -> CNNEspacios:
    """Carga un checkpoint guardado por entrenamiento/entrenar_cnn.py (incluye la arquitectura)."""
    ck = torch.load(ruta_pesos, map_location=dispositivo, weights_only=False)
    modelo = crear_cnn(ck["config_modelo"], ck.get("tam_entrada", 64))
    modelo.load_state_dict(ck["estado"])
    return modelo.to(dispositivo).eval()
