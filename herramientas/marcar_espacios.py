"""
Herramienta de calibración: marcar a mano los espacios de una cámara.

Uso:
    python herramientas/marcar_espacios.py --fuente video.mp4 --nombre mi_parqueadero
    python herramientas/marcar_espacios.py --fuente foto.jpg --nombre mi_parqueadero

Hay dos modos (se cambia con la tecla f):
    ESPACIO : cada 4 clics forman 1 espacio.
    FILA    : (semiautomático) se marcan las 4 esquinas de una FILA completa de espacios, en el orden
              sup-izq -> sup-der -> inf-der -> inf-izq (el borde 1->2 es el que se divide), se escribe con el
              teclado cuántos espacios tiene la fila y se pulsa Enter: la fila se parte en N espacios iguales.
              Sirve también para filas verticales o inclinadas (solo cambia el orden de las esquinas).

Controles:
    clic izquierdo : agrega un vértice
    f              : cambiar entre modo ESPACIO y modo FILA
    0-9 / Enter    : (modo FILA) número de espacios de la fila y confirmar; Backspace borra un dígito
    z              : deshacer (último vértice, o último espacio / última fila completa)
    s              : guardar en config/espacios_<nombre>.json
    q / ESC / X    : salir (también guarda automáticamente lo marcado)
Los IDs se asignan automáticamente por filas (A1, A2, ..., B1, ...) al guardar; con --ids-secuenciales
se usan E1, E2, ... en el orden en que se marcaron.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from herramientas.pklot_a_json import asignar_ids_por_filas  # noqa: E402
from inferencia.recorte import cargar_espacios, guardar_espacios  # noqa: E402

VERTICES_POR_ESPACIO = 4


def dividir_fila(esquinas, n):
    """Parte el cuadrilátero de una fila en n espacios: interpola linealmente sobre el borde 0->1 y,
    en paralelo, sobre el borde 3->2. Devuelve n polígonos de 4 vértices (enteros)."""
    p0, p1, p2, p3 = (np.asarray(e, dtype=float) for e in esquinas)
    t = np.linspace(0.0, 1.0, n + 1)
    arriba = p0 + (p1 - p0) * t[:, None]
    abajo = p3 + (p2 - p3) * t[:, None]
    return [np.rint([arriba[i], arriba[i + 1], abajo[i + 1], abajo[i]]).astype(int).tolist() for i in range(n)]


def primer_frame(fuente):
    if Path(fuente).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(fuente))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise IOError("No se pudo leer el primer frame del video")
        return frame
    img = cv2.imread(str(fuente))
    if img is None:
        raise IOError(f"No se pudo leer la imagen {fuente}")
    return img


class Marcador:
    def __init__(self, imagen, espacios=None):
        self.base = imagen
        self.espacios = [e["poligono"] for e in (espacios or [])]
        self.grupos = [1] * len(self.espacios)  # cuántos espacios agregó cada acción (para deshacer filas)
        self.actual = []
        self.modo_fila = False
        self.digitos = ""
        self.aviso = ""

    def clic(self, evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN and len(self.actual) < VERTICES_POR_ESPACIO:
            self.actual.append([int(x), int(y)])
            if len(self.actual) == VERTICES_POR_ESPACIO and not self.modo_fila:
                self.espacios.append(self.actual)
                self.grupos.append(1)
                self.actual = []

    def cambiar_modo(self):
        self.modo_fila = not self.modo_fila
        self.actual, self.digitos = [], ""

    def tecla_fila(self, tecla):
        """Maneja dígitos, Backspace y Enter en modo FILA."""
        if not self.modo_fila or len(self.actual) < VERTICES_POR_ESPACIO:
            return
        if ord("0") <= tecla <= ord("9") and len(self.digitos) < 3:
            self.digitos += chr(tecla)
        elif tecla == 8:
            self.digitos = self.digitos[:-1]
        elif tecla in (13, 10) and self.digitos and int(self.digitos) > 0:
            nuevos = dividir_fila(self.actual, int(self.digitos))
            self.espacios.extend(nuevos)
            self.grupos.append(len(nuevos))
            self.actual, self.digitos = [], ""

    def deshacer(self):
        if self.digitos:
            self.digitos = ""
        elif self.actual:
            self.actual.pop()
        elif self.grupos:
            n = self.grupos.pop()
            del self.espacios[-n:]

    def dibujar(self):
        img = self.base.copy()
        for i, p in enumerate(self.espacios):
            pts = np.array(p, np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(img, str(i + 1), (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        for x, y in self.actual:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        if len(self.actual) > 1:
            cerrado = len(self.actual) == VERTICES_POR_ESPACIO
            cv2.polylines(img, [np.array(self.actual, np.int32)], cerrado, (0, 0, 255), 1)
        if self.modo_fila and len(self.actual) == VERTICES_POR_ESPACIO:
            n = int(self.digitos) if self.digitos else 0
            for p in (dividir_fila(self.actual, n) if n > 0 else []):  # vista previa de la división
                cv2.polylines(img, [np.array(p, np.int32)], True, (0, 165, 255), 1)
            ayuda = f"FILA: escribe cuantos espacios y Enter -> {self.digitos or '_'}"
        else:
            ayuda = "FILA: clic en 4 esquinas (sup-izq, sup-der, inf-der, inf-izq)" if self.modo_fila else "ESPACIO: 4 clics por espacio"
        lineas = [f"Espacios: {len(self.espacios)} | f=modo z=deshacer s=guardar q=salir (guarda al salir)", ayuda]
        if self.aviso:
            lineas.append(self.aviso)
        for i, t in enumerate(lineas):
            cv2.putText(img, t, (10, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(img, t, (10, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", required=True, help="imagen o video")
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--ids-secuenciales", action="store_true")
    ap.add_argument("--modo-fila", action="store_true", help="empezar en modo FILA (semiautomático)")
    a = ap.parse_args()
    salida = Path(__file__).resolve().parents[1] / "config" / f"espacios_{a.nombre}.json"
    previos = cargar_espacios(salida) if salida.exists() else None
    m = Marcador(primer_frame(a.fuente), previos)
    m.modo_fila = a.modo_fila
    ventana = "marcar espacios"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ventana, m.clic)

    def guardar():
        # Si quedó una fila con sus 4 esquinas y su número escrito pero sin Enter, se confirma sola
        if m.modo_fila and len(m.actual) == VERTICES_POR_ESPACIO and m.digitos:
            m.tecla_fila(13)
        if not m.espacios:
            m.aviso = "Nada que guardar: confirma al menos una fila con Enter"
            return False
        if a.ids_secuenciales:
            esp = [{"id": f"E{i + 1}", "poligono": p} for i, p in enumerate(m.espacios)]
        else:
            esp = asignar_ids_por_filas([{"poligono": p, "pklot_id": i} for i, p in enumerate(m.espacios)])
            for e in esp:
                e.pop("pklot_id", None)
        guardar_espacios(esp, salida)
        m.aviso = f"GUARDADO: {salida.name} ({len(esp)} espacios)"
        print(f"Guardado: {salida} ({len(esp)} espacios)", flush=True)
        return True

    while True:
        cv2.imshow(ventana, m.dibujar())
        tecla = cv2.waitKey(20) & 0xFF
        if tecla != 255 and ord("A") <= tecla <= ord("Z"):
            tecla = tecla + 32  # acepta mayúsculas (Bloq Mayús activado)
        if tecla == ord("z"):
            m.deshacer()
        elif tecla == ord("f"):
            m.cambiar_modo()
        elif m.modo_fila and (ord("0") <= tecla <= ord("9") or tecla in (8, 10, 13)):
            m.tecla_fila(tecla)
        elif tecla == ord("s"):
            guardar()
        elif tecla in (ord("q"), 27):
            break
        # cerrar con la X de la ventana
        if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
            break
    # Al salir se guarda automáticamente para no perder lo marcado
    if m.espacios or (m.modo_fila and m.digitos):
        guardar()
    else:
        print("No se marcó ningún espacio; no se creó el JSON.", flush=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
