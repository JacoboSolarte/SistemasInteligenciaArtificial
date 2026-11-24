"""
Pipeline de inferencia: imagen/video -> recorte de espacios -> CNN (+ detector) -> fusión -> estado + visualización.

    analizador = AnalizadorParqueadero("pesos/cnn_espacios.pth")
    anotada, estado = analizador.analizar_imagen(imagen_bgr, espacios)
    ruta, historial = analizador.analizar_video("video.mp4", espacios, "salida.mp4")
"""
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferencia.dibujo import dibujar_cajas, dibujar_espacios, dibujar_panel  # noqa: E402
from inferencia.fusion import fusionar  # noqa: E402
from inferencia.nms import postprocesar  # noqa: E402
from inferencia.recorte import recortar_todos  # noqa: E402
from modelos.cnn_espacios import cargar_cnn  # noqa: E402
from utilidades.comun import cargar_config, obtener_dispositivo  # noqa: E402


class SuavizadorTemporal:
    """Un espacio solo cambia de estado si la nueva predicción se mantiene durante N frames seguidos."""

    def __init__(self, n_frames: int = 3):
        self.n = max(1, int(n_frames))
        self.estado = {}      # id -> estado confirmado
        self.candidato = {}   # id -> (estado_nuevo, cuántos frames seguidos lo hemos visto)

    def actualizar(self, por_espacio):
        for r in por_espacio:
            i, nuevo = r["id"], r["estado"]
            if i not in self.estado:            # primer frame: se acepta directamente
                self.estado[i] = nuevo
                continue
            if nuevo == self.estado[i]:
                self.candidato.pop(i, None)
            else:
                cand, cuenta = self.candidato.get(i, (nuevo, 0))
                cuenta = cuenta + 1 if cand == nuevo else 1
                if cuenta >= self.n:
                    self.estado[i] = nuevo
                    self.candidato.pop(i, None)
                else:
                    self.candidato[i] = (nuevo, cuenta)
            r["estado_crudo"] = nuevo
            r["estado"] = self.estado[i]
        return por_espacio


def construir_estado(por_espacio, umbral_baja, fuente="", usar_detector=False, n_vehiculos=None):
    """Diccionario serializable a JSON que consume el asistente."""
    total = len(por_espacio)
    ocupados = sum(r["estado"] == "ocupado" for r in por_espacio)
    for r in por_espacio:
        r["dudoso"] = bool(r["confianza"] < umbral_baja)
    confs = [r["confianza"] for r in por_espacio]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fuente": fuente,
        "total": total,
        "libres": total - ocupados,
        "ocupados": ocupados,
        "porcentaje_ocupacion": round(100.0 * ocupados / total, 2) if total else 0.0,
        "confianza_promedio": round(float(np.mean(confs)), 4) if confs else 0.0,
        "espacios_libres": [r["id"] for r in por_espacio if r["estado"] == "libre"],
        "espacios_ocupados": [r["id"] for r in por_espacio if r["estado"] == "ocupado"],
        "espacios_baja_confianza": [r["id"] for r in por_espacio if r["dudoso"]],
        "discrepancias": [r["id"] for r in por_espacio if r.get("discrepancia")],
        "usar_detector": bool(usar_detector),
        "vehiculos_detectados": n_vehiculos,
        "espacios": por_espacio,
    }


class AnalizadorParqueadero:
    def __init__(self, ruta_cnn, ruta_detector=None, cfg=None, dispositivo=None, usar_detector=None):
        self.cfg = cfg or cargar_config()
        self.ci = dict(self.cfg["inferencia"])
        self.dispositivo = dispositivo or obtener_dispositivo()
        self.cnn = cargar_cnn(ruta_cnn, self.dispositivo)
        self.tam = self.cfg["datos"]["tam_recorte"]
        self.detector = None
        usar = self.ci["usar_detector"] if usar_detector is None else usar_detector
        if usar and ruta_detector and Path(ruta_detector).exists():
            from modelos.mini_detector import MiniDetector
            ck = torch.load(ruta_detector, map_location=self.dispositivo, weights_only=False)
            self.detector = MiniDetector(ck["B"], ck["tam_imagen"]).to(self.dispositivo).eval()
            self.detector.load_state_dict(ck["estado"])
        self.usar_detector = self.detector is not None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def clasificar_espacios(self, imagen_bgr, espacios) -> np.ndarray:
        """Recorta todos los espacios (perspectiva a 64x64) y los clasifica EN LOTE. Devuelve P(ocupado)."""
        rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
        recortes = recortar_todos(rgb, espacios, self.tam)
        x = torch.from_numpy(recortes).permute(0, 3, 1, 2).float().div(255).to(self.dispositivo)
        _, _, p_ocupado = self.cnn.predecir(x)
        return p_ocupado.cpu().numpy()

    @torch.no_grad()
    def detectar(self, imagen_bgr):
        """Corre el MiniDetector y devuelve cajas en píxeles de la imagen original + puntajes."""
        from modelos.mini_detector import decodificar
        h, w = imagen_bgr.shape[:2]
        t = self.detector.tam_imagen
        rgb = cv2.cvtColor(cv2.resize(imagen_bgr, (t, t)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self.dispositivo)
        punt, cajas = decodificar(self.detector(x))
        cd = self.cfg["detector"]
        cajas, punt = postprocesar(punt[0].cpu().numpy(), cajas[0].cpu().numpy(), cd["umbral_confianza"], cd["umbral_nms"])
        return cajas * np.array([w, h, w, h]), punt

    def analizar_imagen(self, imagen_bgr, espacios, suavizador=None, fuente="imagen", dibujar=True):
        """Devuelve (imagen_anotada_bgr, estado)."""
        prob = self.clasificar_espacios(imagen_bgr, espacios)
        cajas = punt = None
        if self.usar_detector:
            cajas, punt = self.detectar(imagen_bgr)
        por_espacio = fusionar(espacios, prob, cajas, self.ci["umbral_cnn"], self.ci["umbral_iou_fusion"])
        if suavizador is not None:
            suavizador.actualizar(por_espacio)
        estado = construir_estado(por_espacio, self.ci["umbral_baja_confianza"], fuente,
                                  self.usar_detector, None if cajas is None else int(len(cajas)))
        anotada = None
        if dibujar:
            anotada = dibujar_espacios(imagen_bgr, espacios, por_espacio)
            if cajas is not None:
                anotada = dibujar_cajas(anotada, cajas, punt)
            anotada = dibujar_panel(anotada, estado)
        return anotada, estado

    def analizar_video(self, ruta_video, espacios, ruta_salida, n_suavizado=None, cada_k=None, progreso=None):
        """Procesa 1 de cada K frames, con suavizado temporal de N frames.
        Devuelve (ruta_video_anotado, historial_de_estados)."""
        n = n_suavizado or self.ci["frames_suavizado"]
        k = max(1, int(cada_k or self.ci["procesar_cada_k"]))
        cap = cv2.VideoCapture(str(ruta_video))
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el video {ruta_video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        escritor = EscritorVideo(ruta_salida, fps / k)
        suav = SuavizadorTemporal(n)
        historial, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % k == 0:
                anotada, estado = self.analizar_imagen(frame, espacios, suav, fuente=f"frame {i}")
                estado["frame"] = i
                estado["segundo"] = round(i / fps, 2)
                historial.append(estado)
                escritor.escribir(anotada)
                if progreso:
                    progreso(i, total)
            i += 1
        cap.release()
        escritor.cerrar()
        return str(ruta_salida), historial


class EscritorVideo:
    """Escribe MP4 H.264 (reproducible en el navegador) con imageio-ffmpeg; si no está, usa OpenCV (mp4v)."""

    def __init__(self, ruta, fps):
        self.ruta, self.fps = str(ruta), max(float(fps), 1.0)
        self.w = None
        try:
            import imageio.v2 as imageio
            self.w = imageio.get_writer(self.ruta, fps=self.fps, codec="libx264", quality=7, macro_block_size=1)
            self.modo = "imageio"
        except Exception:
            self.modo = "cv2"

    def escribir(self, frame_bgr):
        if self.modo == "imageio":
            self.w.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            if self.w is None:
                h, w = frame_bgr.shape[:2]
                self.w = cv2.VideoWriter(self.ruta, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.w.write(frame_bgr)

    def cerrar(self):
        if self.w is not None:
            self.w.close() if self.modo == "imageio" else self.w.release()
