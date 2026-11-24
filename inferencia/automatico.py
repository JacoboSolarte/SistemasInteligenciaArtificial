"""
MODO AUTOMÁTICO (sin JSON de espacios): el DetectorUniversal encuentra en la imagen
    * espacios libres  (verde)
    * espacios ocupados (rojo)
    * vehículos         (azul)
y se construye el mismo "estado" que en el modo calibrado, para que la app y el asistente funcionen igual.

Reglas de postproceso (todas propias):
  1. NMS por clase (elimina detecciones duplicadas de la misma clase).
  2. NMS cruzado libre/ocupado: un mismo espacio no puede ser libre y ocupado a la vez; gana el de
     mayor puntaje.
  3. Consistencia espacio-vehículo: si un espacio detectado como LIBRE contiene claramente un vehículo
     detectado (el centro del vehículo cae dentro y cubre buena parte del espacio), se corrige a OCUPADO.
  4. Regla de FILAS (inferencia/filas.py): los vehículos alineados en filas sin espacio detectado cuentan
     como espacios ocupados, y los huecos entre carros de una misma fila como espacios libres inferidos.
  5. Vehículos que siguen sin estar en ningún espacio se reportan aparte ("fuera de espacio":
     circulando o mal parqueados).

En video se usa un seguidor por IoU: cada espacio detectado se asocia con el del frame anterior y su
estado solo cambia si el nuevo estado se mantiene N frames (igual que el suavizado del modo calibrado).

    an = AnalizadorAutomatico("pesos/detector_universal.pth")
    anotada, estado = an.analizar_imagen(imagen_bgr)
"""
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferencia.filas import inferir_por_filas  # noqa: E402
from inferencia.nms import nms  # noqa: E402
from utilidades.comun import cargar_config, obtener_dispositivo  # noqa: E402

VERDE, ROJO, AZUL, AMARILLO = (60, 200, 60), (40, 40, 230), (255, 150, 0), (0, 215, 255)


def preparar_entrada(imagen_rgb, lado):
    """Reescala para que el lado mayor sea `lado` y rellena (gris) hasta múltiplos de 32.
    Devuelve (imagen_rellenada, escala)."""
    h, w = imagen_rgb.shape[:2]
    esc = lado / max(h, w)
    nh, nw = max(32, round(h * esc)), max(32, round(w * esc))
    img = cv2.resize(imagen_rgb, (nw, nh), interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_LINEAR)
    ph, pw = -(-nh // 32) * 32, -(-nw // 32) * 32
    lienzo = np.full((ph, pw, 3), 114, np.uint8)
    lienzo[:nh, :nw] = img
    return lienzo, esc


def cargar_detector_universal(ruta, dispositivo):
    from modelos.detector_universal import DetectorUniversal
    ck = torch.load(ruta, map_location=dispositivo, weights_only=False)
    m = DetectorUniversal(tuple(ck["canales"]), ck["canal_cuello"]).to(dispositivo).eval()
    m.load_state_dict(ck["estado"])
    return m


def contiene(espacio, vehiculo, min_cubre=0.35):
    """¿El vehículo está dentro del espacio? (centro dentro + intersección >= min_cubre del espacio)."""
    cx, cy = (vehiculo[0] + vehiculo[2]) / 2, (vehiculo[1] + vehiculo[3]) / 2
    if not (espacio[0] <= cx <= espacio[2] and espacio[1] <= cy <= espacio[3]):
        return False
    iw = max(0.0, min(espacio[2], vehiculo[2]) - max(espacio[0], vehiculo[0]))
    ih = max(0.0, min(espacio[3], vehiculo[3]) - max(espacio[1], vehiculo[1]))
    area = max((espacio[2] - espacio[0]) * (espacio[3] - espacio[1]), 1e-6)
    return iw * ih / area >= min_cubre


def postprocesar_automatico(cajas, puntajes, clases, umbral_conf=0.3, umbral_nms=0.5, filas=False):
    """Aplica las reglas 1-5 (la 4 solo si filas=True). Devuelve (espacios, vehiculos):
    espacios = [{caja, estado, confianza, corregido}] ; vehiculos = [{caja, confianza, en_espacio}]"""
    sel = puntajes >= umbral_conf
    cajas, puntajes, clases = cajas[sel], puntajes[sel], clases[sel]
    # 1. NMS por clase
    guardar = []
    for c in (0, 1, 2):
        idx = np.where(clases == c)[0]
        if len(idx):
            guardar += list(idx[nms(cajas[idx], puntajes[idx], umbral_nms)])
    guardar = np.array(sorted(guardar), dtype=int)
    cajas, puntajes, clases = cajas[guardar], puntajes[guardar], clases[guardar]
    # 2. NMS cruzado entre libre (0) y ocupado (1)
    esp_idx = np.where(clases <= 1)[0]
    if len(esp_idx):
        esp_idx = esp_idx[nms(cajas[esp_idx], puntajes[esp_idx], 0.4)]
    veh_idx = np.where(clases == 2)[0]
    vehiculos = [{"caja": cajas[i].tolist(), "confianza": float(puntajes[i]), "en_espacio": False} for i in veh_idx]
    espacios = []
    for i in esp_idx:
        e = {"caja": cajas[i].tolist(), "estado": "ocupado" if clases[i] == 1 else "libre",
             "confianza": float(puntajes[i]), "corregido": False}
        # 3. consistencia con los vehículos
        for v in vehiculos:
            if contiene(e["caja"], v["caja"]):
                v["en_espacio"] = True
                if e["estado"] == "libre" and v["confianza"] > e["confianza"]:
                    e["estado"], e["corregido"] = "ocupado", True
                    e["confianza"] = v["confianza"]
        espacios.append(e)
    if filas:
        espacios = inferir_por_filas(espacios, vehiculos)
    return espacios, vehiculos


def asignar_ids(espacios):
    """IDs legibles por filas (A1, A2, ..., B1, ...) según la posición de cada espacio."""
    if not espacios:
        return espacios
    from herramientas.pklot_a_json import asignar_ids_por_filas
    polis = [{"poligono": [[e["caja"][0], e["caja"][1]], [e["caja"][2], e["caja"][1]],
                           [e["caja"][2], e["caja"][3]], [e["caja"][0], e["caja"][3]]], "pklot_id": i}
             for i, e in enumerate(espacios)]
    for p in asignar_ids_por_filas(polis):
        espacios[p["pklot_id"]]["id"] = p["id"]
    return espacios


def construir_estado_automatico(espacios, vehiculos, umbral_baja, fuente=""):
    """Mismo formato que inferencia.pipeline.construir_estado (+ campos de vehículos)."""
    por_espacio = []
    for e in espacios:
        por_espacio.append({"id": e["id"], "estado": e["estado"], "confianza": round(e["confianza"], 4),
                            "prob_ocupado": round(e["confianza"] if e["estado"] == "ocupado" else 1 - e["confianza"], 4),
                            "fuente": e.get("origen") or ("detector_universal" + ("+vehiculo" if e["corregido"] else "")),
                            "discrepancia": bool(e["corregido"]), "dudoso": bool(e["confianza"] < umbral_baja),
                            "caja": [round(v, 1) for v in e["caja"]]})
    total = len(por_espacio)
    ocupados = sum(r["estado"] == "ocupado" for r in por_espacio)
    confs = [r["confianza"] for r in por_espacio]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fuente": fuente,
        "modo": "automatico",
        "total": total,
        "libres": total - ocupados,
        "ocupados": ocupados,
        "porcentaje_ocupacion": round(100.0 * ocupados / total, 2) if total else 0.0,
        "confianza_promedio": round(float(np.mean(confs)), 4) if confs else 0.0,
        "espacios_libres": [r["id"] for r in por_espacio if r["estado"] == "libre"],
        "espacios_ocupados": [r["id"] for r in por_espacio if r["estado"] == "ocupado"],
        "espacios_baja_confianza": [r["id"] for r in por_espacio if r["dudoso"]],
        "discrepancias": [r["id"] for r in por_espacio if r["discrepancia"]],
        "usar_detector": True,
        "vehiculos_detectados": len(vehiculos),
        "vehiculos_fuera_de_espacio": sum(not v["en_espacio"] for v in vehiculos),
        "espacios": por_espacio,
    }


def dibujar_automatico(imagen, estado, vehiculos, mostrar_vehiculos=True):
    """Espacios como rectángulos semitransparentes (verde/rojo, borde amarillo si es dudoso) y
    vehículos como cajas azules. Incluye panel con libres/ocupados/ocupación/vehículos."""
    salida = imagen.copy()
    capa = imagen.copy()
    escala = max(imagen.shape[1] / 1280, 0.45)
    for r in estado["espacios"]:
        x1, y1, x2, y2 = (int(v) for v in r["caja"])
        cv2.rectangle(capa, (x1, y1), (x2, y2), ROJO if r["estado"] == "ocupado" else VERDE, -1)
    cv2.addWeighted(capa, 0.30, salida, 0.70, 0, salida)
    for r in estado["espacios"]:
        x1, y1, x2, y2 = (int(v) for v in r["caja"])
        color = ROJO if r["estado"] == "ocupado" else VERDE
        cv2.rectangle(salida, (x1, y1), (x2, y2), AMARILLO if r["dudoso"] else color, max(1, int(2 * escala)))
    if mostrar_vehiculos:
        for v in vehiculos:
            x1, y1, x2, y2 = (int(t) for t in v["caja"])
            cv2.rectangle(salida, (x1, y1), (x2, y2), AZUL, 1)
    if len(estado["espacios"]) <= 60:
        for r in estado["espacios"]:
            x1, y1, x2, y2 = (int(v) for v in r["caja"])
            fs = 0.35 * escala
            (tw, th), _ = cv2.getTextSize(r["id"], cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(salida, (cx - tw // 2 - 1, cy - th - 1), (cx + tw // 2 + 1, cy + 2), (0, 0, 0), -1)
            cv2.putText(salida, r["id"], (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    lineas = [(f"Libres: {estado['libres']}", VERDE), (f"Ocupados: {estado['ocupados']}", ROJO),
              (f"Ocupacion: {estado['porcentaje_ocupacion']:.1f}%", (255, 255, 255)),
              (f"Vehiculos: {estado['vehiculos_detectados']}", AZUL)]
    alto, ancho = int(26 * escala), int(250 * escala)
    cv2.rectangle(salida, (8, 8), (8 + ancho, 14 + alto * len(lineas)), (0, 0, 0), -1)
    for i, (t, c) in enumerate(lineas):
        cv2.putText(salida, t, (16, 8 + alto * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * escala, c,
                    max(1, int(2 * escala)), cv2.LINE_AA)
    return salida


class SeguidorEspacios:
    """Suavizado temporal para el modo automático (cámara fija o que se mueve despacio).
    Cada espacio detectado se asocia por IoU con una 'pista'; la pista guarda los últimos estados y
    solo cambia de estado si el nuevo se repite N frames seguidos. Las pistas no vistas en
    `olvido` frames se eliminan. Las IDs de las pistas se mantienen estables entre frames."""

    def __init__(self, n_frames=2, umbral_iou=0.4, olvido=5):
        self.n, self.umbral_iou, self.olvido = max(1, int(n_frames)), umbral_iou, olvido
        self.pistas = []  # dicts: caja, estado, historia(deque), sin_ver, id
        self.siguiente = 1

    def actualizar(self, espacios):
        # matriz de IoU detecciones x pistas (vectorizada) y asociación voraz por mayor IoU
        asignacion = {}
        if espacios and self.pistas:
            D = np.array([e["caja"] for e in espacios], np.float64)
            P = np.array([p["caja"] for p in self.pistas], np.float64)
            x1 = np.maximum(D[:, None, 0], P[None, :, 0])
            y1 = np.maximum(D[:, None, 1], P[None, :, 1])
            x2 = np.minimum(D[:, None, 2], P[None, :, 2])
            y2 = np.minimum(D[:, None, 3], P[None, :, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area_d = (D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1])
            area_p = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])
            iou = inter / np.maximum(area_d[:, None] + area_p[None, :] - inter, 1e-9)
            pares = np.argwhere(iou >= self.umbral_iou)
            pares = pares[np.argsort(-iou[pares[:, 0], pares[:, 1]])]
            usadas_d, usadas_p = set(), set()
            for d, k in pares:
                if d not in usadas_d and k not in usadas_p:
                    asignacion[int(d)] = int(k)
                    usadas_d.add(d)
                    usadas_p.add(k)
        vistas = set()
        for d, e in enumerate(espacios):
            if d in asignacion:
                p = self.pistas[asignacion[d]]
            else:
                p = {"caja": e["caja"], "estado": e["estado"], "historia": deque(maxlen=self.n), "sin_ver": 0,
                     "id": f"P{self.siguiente}"}
                self.siguiente += 1
                self.pistas.append(p)
            vistas.add(id(p))
            p["caja"] = [0.6 * a + 0.4 * b for a, b in zip(p["caja"], e["caja"])]
            p["historia"].append(e["estado"])
            p["sin_ver"] = 0
            if len(p["historia"]) == self.n and len(set(p["historia"])) == 1:
                p["estado"] = p["historia"][-1]
            e["estado_crudo"], e["estado"] = e["estado"], p["estado"]
            e["caja"], e["id_pista"] = p["caja"], p["id"]
        for p in self.pistas:
            if id(p) not in vistas:
                p["sin_ver"] += 1
        self.pistas = [p for p in self.pistas if p["sin_ver"] <= self.olvido]
        return espacios


class AnalizadorAutomatico:
    def __init__(self, ruta_pesos, cfg=None, dispositivo=None):
        self.cfg = cfg or cargar_config()
        self.cu = dict(self.cfg["detector_universal"])
        # la escala de confianza del detector es más baja que la de la CNN: umbral de "dudoso" propio
        self.umbral_baja = self.cu.get("umbral_dudoso", 0.4)
        self.dispositivo = dispositivo or obtener_dispositivo()
        self.modelo = cargar_detector_universal(ruta_pesos, self.dispositivo)

    @torch.no_grad()
    def detectar(self, imagen_bgr, lado=None):
        """Devuelve (cajas [M,4] en píxeles de la imagen original, puntajes, clases).
        lado = número (una escala) o lista de lados (MULTIESCALA: se analiza la imagen a varios tamaños y
        se juntan las detecciones; el NMS del postproceso elimina los duplicados). Así funcionan tanto
        carros grandes (dron bajo / cámara cercana) como diminutos (dron alto)."""
        from modelos.detector_universal import decodificar
        base = lado if lado is not None else self.cu["lado_inferencia"]
        if base == "auto":
            # RESOLUCIÓN AUTOMÁTICA: se analiza a 800 px; si los objetos detectados son diminutos
            # (mediana del lado menor < tam_auto px, típico de dron alto) se agrega la escala 1280.
            c, p, k = self.detectar(imagen_bgr, 800)
            m = p >= self.cu["umbral_confianza"]
            lado_min = np.minimum(c[m, 2] - c[m, 0], c[m, 3] - c[m, 1]) * 800 / max(imagen_bgr.shape[:2])
            self.ultima_escala = "800"
            if len(lado_min) and np.median(lado_min) < self.cu.get("tam_auto", 24):
                self.ultima_escala = "800+1280"
                return self.detectar(imagen_bgr, [800, 1280])
            return c, p, k
        lados = sorted(base) if isinstance(base, (list, tuple)) else [base]
        rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
        h, w = imagen_bgr.shape[:2]
        todas = []
        for n_escala, ld in enumerate(lados):
            entrada, esc = preparar_entrada(rgb, int(ld))
            x = torch.from_numpy(entrada).permute(2, 0, 1)[None].float().div(255).to(self.dispositivo)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.dispositivo.type == "cuda"):
                salida = self.modelo(x)
            cajas, punt, clases = decodificar(salida, umbral=0.05)[0]
            cajas = cajas / esc
            cajas[:, [0, 2]] = cajas[:, [0, 2]].clip(0, w)
            cajas[:, [1, 3]] = cajas[:, [1, 3]].clip(0, h)
            if n_escala > 0:
                # escalas grandes solo aportan objetos PEQUEÑOS (los que la escala base no resuelve bien);
                # los objetos medianos/grandes se toman de la escala base (evita duplicados y falsos espacios)
                lado_min = np.minimum(cajas[:, 2] - cajas[:, 0], cajas[:, 3] - cajas[:, 1]) * lados[0] / max(h, w)
                m = lado_min < self.cu.get("tam_pequeno", 22)
                cajas, punt, clases = cajas[m], punt[m], clases[m]
            todas.append((cajas, punt, clases))
        return (np.concatenate([t[0] for t in todas]), np.concatenate([t[1] for t in todas]),
                np.concatenate([t[2] for t in todas]))

    def analizar_imagen(self, imagen_bgr, seguidor=None, fuente="imagen", dibujar=True,
                        umbral_conf=None, lado=None, mostrar_vehiculos=True, filas=None):
        cajas, punt, clases = self.detectar(imagen_bgr, lado)
        filas = self.cu.get("inferir_por_filas", True) if filas is None else filas
        espacios, vehiculos = postprocesar_automatico(cajas, punt, clases, umbral_conf or self.cu["umbral_confianza"],
                                                      self.cu["umbral_nms"], filas)
        if seguidor is not None:  # en video: IDs estables de las pistas (P1, P2, ...)
            seguidor.actualizar(espacios)
            for e in espacios:
                e["id"] = e["id_pista"]
        else:                     # en imagen: IDs por filas (A1, A2, ..., B1, ...)
            espacios = asignar_ids(espacios)
        estado = construir_estado_automatico(espacios, vehiculos, self.umbral_baja, fuente)
        anotada = dibujar_automatico(imagen_bgr, estado, vehiculos, mostrar_vehiculos) if dibujar else None
        return anotada, estado

    def analizar_video(self, ruta_video, ruta_salida, n_suavizado=None, cada_k=None, progreso=None,
                       umbral_conf=None, lado=None, max_lado_salida=1280, filas=None):
        from inferencia.pipeline import EscritorVideo
        n = n_suavizado or self.cfg["inferencia"]["frames_suavizado"]
        k = max(1, int(cada_k or self.cfg["inferencia"]["procesar_cada_k"]))
        cap = cv2.VideoCapture(str(ruta_video))
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el video {ruta_video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        escritor = EscritorVideo(ruta_salida, fps / k)
        seg = SeguidorEspacios(n)
        historial, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % k == 0:
                h, w = frame.shape[:2]
                if max(h, w) > max_lado_salida:  # videos 4K: se reducen para no saturar memoria
                    e = max_lado_salida / max(h, w)
                    frame = cv2.resize(frame, (round(w * e), round(h * e)), interpolation=cv2.INTER_AREA)
                anotada, estado = self.analizar_imagen(frame, seg, f"frame {i}", True, umbral_conf, lado, True, filas)
                estado["frame"], estado["segundo"] = i, round(i / fps, 2)
                historial.append(estado)
                escritor.escribir(anotada)
                if progreso:
                    progreso(i, total)
            i += 1
        cap.release()
        escritor.cerrar()
        for h in historial[:-1]:  # el detalle por espacio solo se guarda del último frame (ahorra memoria)
            h.pop("espacios", None)
        return str(ruta_salida), historial
