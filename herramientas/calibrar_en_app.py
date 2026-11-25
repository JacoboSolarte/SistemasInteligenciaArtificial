"""
Editor de calibración dentro de la app Streamlit: marcar los espacios de una cámara con clics sobre la imagen
(equivalente web de herramientas/marcar_espacios.py) y obtener el JSON [{id, poligono}] sin salir de la app.

Dos formas de marcar:
    FILA     : 4 clics en las esquinas de una fila completa (sup-izq -> sup-der -> inf-der -> inf-izq); DESPUÉS se
               elige cuántos espacios tiene (con vista previa de la división) y se pulsa "Agregar fila": la fila se
               parte en N espacios iguales (dividir_fila). Es la forma rápida.
    ESPACIO  : 4 clics = 1 espacio.
Los IDs se asignan por filas (A1, A2, ..., B1, ...) igual que en la herramienta de escritorio.
"""
import json

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from herramientas.marcar_espacios import VERTICES_POR_ESPACIO, dividir_fila
from herramientas.pklot_a_json import asignar_ids_por_filas
from inferencia.recorte import guardar_espacios

ANCHO_VISTA = 1000  # ancho (px) con el que se muestra la imagen para marcar
COLORES = [(80, 200, 80), (80, 160, 255), (255, 170, 60), (200, 90, 220), (60, 210, 210)]


def _estado(fuente):
    c = st.session_state.get("calib")
    if c is None or c["fuente"] != fuente:
        c = {"fuente": fuente, "grupos": [], "puntos": [], "ultimo_clic": None}
        st.session_state.calib = c
    return c


def espacios_marcados(grupos):
    """Lista [{id, poligono}] con IDs por filas, o None si no hay nada marcado."""
    poligonos = [p for g in grupos for p in g]
    if not poligonos:
        return None
    esp = asignar_ids_por_filas([{"poligono": p, "pklot_id": i} for i, p in enumerate(poligonos)])
    for e in esp:
        e.pop("pklot_id", None)
    return esp


def _dibujar(img, c, escala, n_previa=None):
    vista = cv2.resize(img, (int(img.shape[1] / escala), int(img.shape[0] / escala)), interpolation=cv2.INTER_AREA)
    capa = vista.copy()
    esp = espacios_marcados(c["grupos"]) or []
    for i, e in enumerate(esp):
        pts = (np.asarray(e["poligono"], float) / escala).astype(np.int32)
        cv2.fillPoly(capa, [pts], COLORES[i % len(COLORES)])
    vista = cv2.addWeighted(capa, 0.35, vista, 0.65, 0)
    for e in esp:
        pts = (np.asarray(e["poligono"], float) / escala).astype(np.int32)
        cv2.polylines(vista, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(vista, e["id"], (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vista, e["id"], (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    if n_previa and len(c["puntos"]) == VERTICES_POR_ESPACIO:
        # vista previa de la fila pendiente: cómo quedaría dividida en n_previa espacios
        for poli in dividir_fila(c["puntos"], n_previa):
            cv2.polylines(vista, [(np.asarray(poli, float) / escala).astype(np.int32)], True, (0, 230, 255), 2, cv2.LINE_AA)
    pts = [(int(x / escala), int(y / escala)) for x, y in c["puntos"]]
    for a, b in zip(pts, pts[1:]):
        cv2.line(vista, a, b, (0, 0, 255), 2, cv2.LINE_AA)
    for k, p in enumerate(pts):
        cv2.circle(vista, p, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(vista, str(k + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(vista, cv2.COLOR_BGR2RGB)


def editor_calibracion(img_bgr, fuente, carpeta_config):
    """Muestra el editor y devuelve la lista de espacios marcados (o None)."""
    c = _estado(fuente)
    escala = max(img_bgr.shape[1] / ANCHO_VISTA, 1.0)

    st.subheader("✏️ Calibración: marca los espacios con clics")
    forma = st.radio("Forma de marcar", ["Fila completa (4 esquinas y luego cuántos espacios)",
                                         "Espacio individual (4 clics)"], horizontal=True, key="calib_forma")
    por_fila = forma.startswith("Fila")
    fila_pendiente = por_fila and len(c["puntos"]) == VERTICES_POR_ESPACIO
    if por_fila:
        st.caption("1) Haz clic en las 4 esquinas de la fila en este orden: **1** arriba-izquierda → **2** arriba-derecha "
                   "→ **3** abajo-derecha → **4** abajo-izquierda.  2) Elige cuántos espacios tiene (verás la división en "
                   "amarillo) y pulsa **✅ Agregar fila**.")
    else:
        st.caption("Haz clic en las 4 esquinas del espacio (en orden, rodeándolo); se agrega solo al cuarto clic.")

    n_fila = None
    if fila_pendiente:
        a1, a2 = st.columns([1, 1])
        n_fila = int(a1.number_input("¿Cuántos espacios tiene esta fila?", 1, 60, 6, key="calib_n"))
        a2.write("")
        if a2.button("✅ Agregar fila", type="primary", width="stretch"):
            c["grupos"].append(dividir_fila(c["puntos"], n_fila))
            c["puntos"] = []
            st.rerun()

    clic = streamlit_image_coordinates(_dibujar(img_bgr, c, escala, n_fila), width=int(img_bgr.shape[1] / escala),
                                       key=f"calib_img_{fuente}", cursor="crosshair")
    if clic and clic.get("unix_time") != c["ultimo_clic"]:
        c["ultimo_clic"] = clic.get("unix_time")
        if fila_pendiente:
            st.toast("Ya marcaste las 4 esquinas: elige el número de espacios y pulsa ✅ Agregar fila (o ↩️ Deshacer).")
        else:
            f = img_bgr.shape[1] / clic.get("width", img_bgr.shape[1] / escala)
            c["puntos"].append([int(round(clic["x"] * f)), int(round(clic["y"] * f))])
            if len(c["puntos"]) == VERTICES_POR_ESPACIO and not por_fila:
                c["grupos"].append([c["puntos"]])
                c["puntos"] = []
            st.rerun()

    esp = espacios_marcados(c["grupos"])
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("↩️ Deshacer", width="stretch", disabled=not (c["puntos"] or c["grupos"])):
        if c["puntos"]:
            c["puntos"].pop()
        else:
            c["grupos"].pop()
        st.rerun()
    if b2.button("🗑️ Borrar todo", width="stretch", disabled=not (c["puntos"] or c["grupos"])):
        c["grupos"], c["puntos"] = [], []
        st.rerun()
    nombre = st.text_input("Nombre de la calibración", value="mi_parqueadero", key="calib_nombre")
    if b3.button("💾 Guardar en config/", width="stretch", disabled=esp is None):
        destino = carpeta_config / f"espacios_{nombre}.json"
        guardar_espacios(esp, destino)
        st.success(f"Guardado en config/{destino.name}: ya aparece en la lista «Del proyecto».")
    b4.download_button("⬇️ Descargar JSON", json.dumps(esp or [], ensure_ascii=False, indent=1),
                       file_name=f"espacios_{nombre}.json", mime="application/json", width="stretch",
                       disabled=esp is None)
    total = len(esp) if esp else 0
    pend = (" · fila lista: elige cuántos espacios y pulsa ✅ Agregar fila" if fila_pendiente
            else f" · {len(c['puntos'])}/4 clics de la figura actual" if c["puntos"] else "")
    st.info(f"**{total} espacios marcados**{pend}. Cuando termines, pulsa **🔍 Analizar** en la barra lateral.")
    return esp
