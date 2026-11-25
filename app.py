"""
Aplicación Streamlit: detección de ocupación de parqueaderos + asistente conversacional (Transformer propio).

    streamlit run app.py
"""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from asistente.asistente import Asistente
from herramientas.calibrar_en_app import editor_calibracion, espacios_marcados
from inferencia.automatico import AnalizadorAutomatico
from inferencia.pipeline import AnalizadorParqueadero
from inferencia.recorte import validar_espacios
from utilidades.comun import RAIZ, cargar_config, obtener_dispositivo, ruta

st.set_page_config(page_title="Parqueadero IA", page_icon="🅿️", layout="wide")
CFG = cargar_config()
PESOS, RESULTADOS = ruta(CFG, "pesos"), ruta(CFG, "resultados")
EJEMPLOS = RAIZ / "ejemplos"


@st.cache_resource(show_spinner="Cargando modelos...")
def cargar_analizador(usar_detector: bool):
    return AnalizadorParqueadero(PESOS / "cnn_espacios.pth", PESOS / "mini_detector.pth", CFG, usar_detector=usar_detector)


@st.cache_resource(show_spinner="Cargando detector universal...")
def cargar_automatico():
    return AnalizadorAutomatico(PESOS / "detector_universal.pth", CFG)


@st.cache_resource(show_spinner=False)
def cargar_asistente():
    p = PESOS / "transformer_asistente.pth"
    return Asistente(p, CFG["asistente"]["umbral_intencion"], obtener_dispositivo()) if p.exists() else None


def imagen_fuente(archivo_subido, ejemplo, tipo):
    """Imagen BGR de la fuente elegida (primer frame si es video) y su nombre; (None, None) si no hay."""
    if archivo_subido is None and ejemplo is None:
        return None, None
    nombre = archivo_subido.name if archivo_subido else ejemplo
    if tipo == "Imagen":
        datos = archivo_subido.getvalue() if archivo_subido else (EJEMPLOS / ejemplo).read_bytes()
        return cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR), nombre
    if archivo_subido:
        tmp = Path(tempfile.gettempdir()) / f"calib_{archivo_subido.name}"
        tmp.write_bytes(archivo_subido.getvalue())
    else:
        tmp = EJEMPLOS / ejemplo
    cap = cv2.VideoCapture(str(tmp))
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else None), nombre


def estado_sesion():
    for k, v in {"estado": None, "historial": None, "anotada": None, "video": None, "mensajes": []}.items():
        st.session_state.setdefault(k, v)


estado_sesion()
st.title("🅿️ Detección inteligente de ocupación de parqueaderos")
st.caption("Modo automático (DetectorUniversal propio) · modo calibrado (CNN + MiniDetector propios) · "
           "asistente con Transformer propio — todo entrenado desde cero")
HAY_AUTO = (PESOS / "detector_universal.pth").exists()

if not (PESOS / "cnn_espacios.pth").exists() and not HAY_AUTO:
    st.error("No hay pesos entrenados. Entrena primero: `python entrenamiento/entrenar_cnn.py`")
    st.stop()

# ---------------------------------------------------------------- Barra lateral
with st.sidebar:
    st.header("1. Fuente")
    tipo = st.radio("Tipo de entrada", ["Imagen", "Video"], horizontal=True)
    ext = ["jpg", "jpeg", "png"] if tipo == "Imagen" else ["mp4", "avi", "mov", "mkv"]
    ejemplos = sorted(p.name for p in EJEMPLOS.glob("*") if p.suffix.lower().lstrip(".") in ext)
    origen = st.radio("Origen", ["Ejemplo del proyecto", "Subir archivo"], horizontal=True)
    archivo_subido, ejemplo = None, None
    if origen == "Subir archivo":
        archivo_subido = st.file_uploader(f"Sube un(a) {tipo.lower()}", type=ext)
    else:
        ejemplo = st.selectbox("Ejemplo", ejemplos) if ejemplos else None

    st.header("2. Modo de detección")
    modos = (["Automático (sin JSON)"] if HAY_AUTO else []) + ["Calibrado (JSON de espacios)"]
    modo = st.radio("Modo", modos, help="Automático: el modelo encuentra solo los espacios y los vehículos, en cualquier "
                    "imagen (dron, edificio, lateral, suelo). Calibrado: usa los espacios marcados de una cámara fija "
                    "(más preciso para esa cámara).")
    automatico = modo.startswith("Automático")
    espacios, calibrar_en_app = None, False
    if not automatico:
        jsons = sorted(p.name for p in (RAIZ / "config").glob("espacios_*.json"))
        sugerido = next((j for j in jsons if ejemplo and j.replace("espacios_", "").replace(".json", "") in ejemplo), jsons[0] if jsons else None)
        modo_json = st.radio("JSON de espacios", ["Del proyecto", "Subir JSON", "Marcar en la app"], horizontal=True,
                             help="Marcar en la app: dibuja los espacios con clics sobre la imagen (pestaña Análisis)")
        if modo_json == "Marcar en la app":
            calibrar_en_app = True
            c = st.session_state.get("calib")
            fuente_actual = archivo_subido.name if archivo_subido else ejemplo
            if c and c["fuente"] == fuente_actual:
                espacios = espacios_marcados(c["grupos"])
            st.caption(f"Espacios marcados: **{len(espacios) if espacios else 0}** (márcalos en la pestaña 📷 Análisis)")
        elif modo_json == "Del proyecto" and jsons:
            elegido = st.selectbox("Archivo", jsons, index=jsons.index(sugerido) if sugerido in jsons else 0)
            espacios = json.loads((RAIZ / "config" / elegido).read_text(encoding="utf-8"))
        else:
            j = st.file_uploader("JSON [{id, poligono}]", type=["json"])
            if j:
                espacios = json.loads(j.read().decode("utf-8"))
        if not calibrar_en_app:
            st.caption("¿Cámara nueva? Elige «Marcar en la app» o marca los espacios con `python herramientas/marcar_espacios.py --fuente <imagen> --nombre <nombre> --modo-fila` (por filas, semiautomático) y luego elige `espacios_<nombre>.json` aquí.")

    st.header("3. Parámetros")
    cu = CFG["detector_universal"]
    if automatico:
        umbral_auto = st.slider("Confianza mínima de detección", 0.05, 0.9, float(cu["umbral_confianza"]), 0.05,
                                help="Más bajo = detecta más (y más falsas alarmas); más alto = solo lo muy seguro")
        opciones_res = {"Automática (recomendada)": "auto", "Multiescala 800 + 1280": [800, 1280],
                        "640 (carros muy grandes / cercanos)": 640,
                        "800": 800, "1024": 1024, "1280 (carros pequeños / dron alto)": 1280, "1600 (carros diminutos)": 1600}
        lado = opciones_res[st.selectbox("Resolución de análisis", list(opciones_res),
                                         help="Automática: analiza a 800 px y, si los carros salen diminutos (dron alto), "
                                              "agrega 1280 px. Multiescala: siempre a dos tamaños.")]
        usar_filas = st.checkbox("Regla de filas (inferir espacios a partir de carros alineados)",
                                 value=bool(cu.get("inferir_por_filas", True)),
                                 help="Útil en vista aérea: carros en fila sin espacio detectado = ocupados; "
                                      "huecos entre ellos = libres inferidos (borde amarillo)")
        ver_vehiculos = st.checkbox("Dibujar vehículos (azul)", value=True)
    else:
        hay_detector = (PESOS / "mini_detector.pth").exists()
        usar_detector = st.checkbox("Usar MiniDetector (fusión)", value=CFG["inferencia"]["usar_detector"] and hay_detector,
                                    disabled=not hay_detector, help=None if hay_detector else "No hay pesos del detector")
        umbral_cnn = st.slider("Umbral de confianza de la CNN (fusión)", 0.5, 1.0, float(CFG["inferencia"]["umbral_cnn"]), 0.01)
        umbral_iou = st.slider("Umbral IoU caja-espacio", 0.05, 0.9, float(CFG["inferencia"]["umbral_iou_fusion"]), 0.05)
    umbral_baja = st.slider("Confianza mínima (marca 'dudoso')", 0.1, 1.0,
                            float(cu.get("umbral_dudoso", 0.4) if automatico else CFG["inferencia"]["umbral_baja_confianza"]), 0.01)
    if tipo == "Video":
        n_suav = st.number_input("Suavizado: N frames para cambiar de estado", 1, 30, int(CFG["inferencia"]["frames_suavizado"]))
        cada_k = st.number_input("Procesar 1 de cada K frames", 1, 60, int(CFG["inferencia"]["procesar_cada_k"]))
    analizar = st.button("🔍 Analizar", type="primary", width="stretch")

# ---------------------------------------------------------------- Análisis
if analizar:
    if not automatico and espacios is None:
        st.sidebar.error("Faltan los espacios: marca al menos una fila en la pestaña Análisis" if calibrar_en_app
                         else "Falta el JSON de espacios")
    elif archivo_subido is None and ejemplo is None:
        st.sidebar.error("Falta la imagen o el video")
    else:
        try:
            nombre = archivo_subido.name if archivo_subido else ejemplo
            if automatico:
                an = cargar_automatico()
                an.umbral_baja = umbral_baja
            else:
                validar_espacios(espacios)
                an = cargar_analizador(usar_detector)
                an.ci.update(umbral_cnn=umbral_cnn, umbral_iou_fusion=umbral_iou, umbral_baja_confianza=umbral_baja)
            if tipo == "Imagen":
                datos = archivo_subido.read() if archivo_subido else (EJEMPLOS / ejemplo).read_bytes()
                img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("No se pudo leer la imagen")
                if automatico:
                    anotada, estado = an.analizar_imagen(img, fuente=nombre, umbral_conf=umbral_auto, lado=lado,
                                                         mostrar_vehiculos=ver_vehiculos, filas=usar_filas)
                else:
                    anotada, estado = an.analizar_imagen(img, espacios, fuente=nombre)
                st.session_state.update(anotada=anotada, estado=estado, historial=None, video=None)
            else:
                if archivo_subido:
                    tmp = Path(tempfile.gettempdir()) / f"entrada_{archivo_subido.name}"
                    tmp.write_bytes(archivo_subido.read())
                else:
                    tmp = EJEMPLOS / ejemplo
                salida = Path(tempfile.gettempdir()) / "video_anotado.mp4"
                barra = st.progress(0.0, text="Procesando video...")

                def prog(i, t):
                    barra.progress(min(i / max(t, 1), 1.0), text=f"frame {i}/{t}")

                if automatico:
                    _, hist = an.analizar_video(tmp, salida, int(n_suav), int(cada_k), progreso=prog,
                                                umbral_conf=umbral_auto, lado=lado, filas=usar_filas)
                else:
                    _, hist = an.analizar_video(tmp, espacios, salida, int(n_suav), int(cada_k), progreso=prog)
                barra.empty()
                st.session_state.update(video=salida.read_bytes(), historial=hist, estado=hist[-1] if hist else None, anotada=None)
            st.session_state.mensajes.append({"rol": "assistant", "texto": f"Analicé **{nombre}** (modo {'automático' if automatico else 'calibrado'}). ¡Pregúntame lo que quieras sobre la disponibilidad!"})
        except Exception as e:  # mostrar el error sin romper la app
            st.error(f"Error durante el análisis: {e}")

tab_analisis, tab_chat, tab_resultados = st.tabs(["📷 Análisis", "💬 Asistente", "📊 Resultados del entrenamiento"])

with tab_analisis:
    if calibrar_en_app:
        img_calib, nombre_calib = imagen_fuente(archivo_subido, ejemplo, tipo)
        if img_calib is None:
            st.warning("Elige primero una imagen o un video en la barra lateral para marcar sus espacios.")
        else:
            with st.expander("✏️ Calibración de espacios (modo calibrado)", expanded=True):
                editor_calibracion(img_calib, nombre_calib, RAIZ / "config")
    estado = st.session_state.estado
    if estado is None:
        st.info("Elige una imagen o un video en la barra lateral y pulsa **Analizar**.")
    else:
        hist = st.session_state.historial
        if hist:
            i = st.slider("Frame analizado", 0, len(hist) - 1, len(hist) - 1,
                          format="%d", help="El asistente responde con el estado del frame elegido")
            estado = hist[i]
            st.session_state.estado = estado
        es_auto = estado.get("modo") == "automatico"
        cols = st.columns(5 if es_auto else 4)
        cols[0].metric("🟢 Libres", estado["libres"])
        cols[1].metric("🔴 Ocupados", estado["ocupados"])
        cols[2].metric("Ocupación", f"{estado['porcentaje_ocupacion']:.1f} %")
        cols[3].metric("Confianza promedio", f"{estado['confianza_promedio']:.3f}")
        if es_auto:
            cols[4].metric("🔵 Vehículos", estado.get("vehiculos_detectados", 0),
                           help=f"{estado.get('vehiculos_fuera_de_espacio', 0)} fuera de un espacio detectado")
        if st.session_state.anotada is not None:
            st.image(cv2.cvtColor(st.session_state.anotada, cv2.COLOR_BGR2RGB), width="stretch",
                     caption="Verde = libre · Rojo = ocupado · borde amarillo = baja confianza · azul = vehículos / cajas del detector")
        if es_auto and estado["total"] == 0:
            st.warning("No se detectaron espacios. Prueba otra resolución de análisis o baja la confianza mínima.")
        if st.session_state.video is not None:
            st.video(st.session_state.video)
            st.download_button("Descargar video anotado", st.session_state.video, "video_anotado.mp4", "video/mp4")
            serie = pd.DataFrame([{"segundo": h["segundo"], "ocupación %": h["porcentaje_ocupacion"],
                                   "libres": h["libres"], "ocupados": h["ocupados"]} for h in hist]).set_index("segundo")
            st.subheader("Ocupación en el tiempo")
            st.line_chart(serie[["ocupación %"]])
            st.line_chart(serie[["libres", "ocupados"]])
        if estado.get("espacios_baja_confianza"):
            st.warning(f"Espacios con baja confianza: {', '.join(estado['espacios_baja_confianza'])}")
        if estado.get("discrepancias"):
            if es_auto:
                st.info(f"Espacios corregidos a 'ocupado' porque contienen un vehículo detectado: {', '.join(estado['discrepancias'])}")
            else:
                st.warning(f"La CNN y el detector no coinciden en: {', '.join(estado['discrepancias'])}")
        if estado.get("espacios"):
            st.subheader("Detalle por espacio")
            tabla = pd.DataFrame(estado["espacios"])
            cols = [c for c in ["id", "estado", "confianza", "prob_ocupado", "fuente", "iou_detector", "discrepancia", "dudoso", "caja"] if c in tabla]
            st.dataframe(tabla[cols], width="stretch", hide_index=True, height=300)
        st.download_button("Descargar estado (JSON)", json.dumps(estado, ensure_ascii=False, indent=2), "estado.json", "application/json")

with tab_chat:
    asistente = cargar_asistente()
    if asistente is None:
        st.error("Falta pesos/transformer_asistente.pth (python entrenamiento/entrenar_asistente.py)")
    else:
        st.caption("El Transformer clasifica tu pregunta en una intención; la respuesta se llena con los datos reales del análisis.")
        for m in st.session_state.mensajes:
            with st.chat_message(m["rol"]):
                st.markdown(m["texto"])
        pregunta = st.chat_input("Ej.: ¿hay cupo?, ¿qué tan lleno está?, ¿dónde puedo parquear?, ¿qué tan seguro estás?")
        if pregunta:
            respuesta, intencion, conf = asistente.responder(pregunta, st.session_state.estado)
            st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
            st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta + f"\n\n<sub>intención: `{intencion}` · confianza {conf:.2f}</sub>"})
            st.rerun()

with tab_resultados:
    st.caption(f"Gráficas y tablas generadas por los scripts de entrenamiento y evaluación (carpeta `{RESULTADOS.name}/`).")
    secciones = [
        ("Modelo 1 — CNNEspacios", ["curvas_cnn_general.png", "curvas_cnn_general_optimizada.png", "cnn_test_cm_roc.png",
                                     "cnn_generalizacion_parqueadero_cm_roc.png", "cnn_generalizacion_clima_cm_roc.png"],
         ["metricas_cnn.csv", "resumen_cnn_general.csv", "resumen_cnn_general_optimizada.csv"]),
        ("Optimización de hiperparámetros (Hill Climbing y estrategia evolutiva)", ["optimizacion_hiperparametros.png"],
         ["optimizacion_comparacion.csv"]),
        ("Modelo 2 — MiniDetector", ["curvas_detector.png", "detector_curva_pr.png", "detector_ejemplos.png"], ["metricas_detector.csv"]),
        ("Modelo 3 — TransformerAsistente", ["curvas_asistente.png", "asistente_matrices_confusion.png"], ["metricas_asistente.csv"]),
        ("Modelo 4 — DetectorUniversal (modo automático)", ["curvas_detector_universal.png", "detector_universal_ejemplos.png"],
         ["metricas_detector_universal.csv", "conteo_detector_universal.csv"]),
        ("Evaluación en video", ["evaluacion_video.png", "evaluacion_video_con_detector.png"], ["evaluacion_video_resumen_solo_cnn.csv", "evaluacion_video_resumen_con_detector.csv"]),
    ]
    for titulo, imgs, csvs in secciones:
        with st.expander(titulo, expanded=False):
            for c in csvs:
                if (RESULTADOS / c).exists():
                    st.markdown(f"**{c}**")
                    st.dataframe(pd.read_csv(RESULTADOS / c).round(4), width="stretch", hide_index=True)
            for im in imgs:
                if (RESULTADOS / im).exists():
                    st.image(str(RESULTADOS / im), caption=im, width="stretch")
            if not any((RESULTADOS / x).exists() for x in imgs + csvs):
                st.info("Aún no hay resultados para esta sección.")
