"""
Generación de respuestas: la intención predicha por el Transformer elige una plantilla
que se llena con el ESTADO REAL del análisis (salida de inferencia/pipeline.py).

Además de informar la disponibilidad, se da retroalimentación sobre el resultado del algoritmo:
confianza promedio de la CNN, espacios con baja confianza y discrepancias CNN vs. detector.
"""
import random

SUGERENCIAS = ["¿Hay puestos libres?", "¿Qué tan lleno está?", "¿Dónde puedo parquear?",
               "¿Cuántos carros hay?", "¿Qué tan seguro estás?", "Dame un reporte"]


def _lista(ids, maximo=12):
    if not ids:
        return "ninguno"
    txt = ", ".join(ids[:maximo])
    return txt + (f" y {len(ids) - maximo} más" if len(ids) > maximo else "")


def _por_fila(ids):
    """Agrupa IDs tipo A3, B12 por la letra de la fila."""
    filas = {}
    for i in ids:
        letra = "".join(c for c in i if c.isalpha()) or "?"
        filas.setdefault(letra, []).append(i)
    return filas


def retroalimentacion(estado, umbral_conf=0.85) -> str:
    """Texto sobre la calidad del resultado del algoritmo."""
    partes = [f"La confianza promedio de la detección es {estado['confianza_promedio']:.2f}"]
    dudosos = estado.get("espacios_baja_confianza", [])
    if dudosos:
        partes.append(f"; los espacios {_lista(dudosos, 8)} tienen baja confianza (posible sombra, oclusión o "
                      "un vehículo mal estacionado), conviene verificarlos")
    else:
        partes.append("; ningún espacio quedó con baja confianza")
    texto = "".join(partes) + "."
    if estado.get("modo") == "automatico":
        # modo automático: los puntajes del detector universal son más bajos que las probabilidades de la CNN
        disc = estado.get("discrepancias", [])
        if disc:
            texto += (f" {len(disc)} espacio(s) ({_lista(disc, 8)}) parecían libres pero contienen un vehículo "
                      "detectado, así que los conté como ocupados.")
        fuera = estado.get("vehiculos_fuera_de_espacio", 0)
        if fuera:
            texto += f" Hay {fuera} vehículo(s) fuera de un espacio (circulando o mal parqueados)."
        texto += (" Ojo: en modo automático los espacios se detectan sin calibración; para una cámara fija, el modo "
                  "calibrado es más preciso.")
        if estado["confianza_promedio"] < 0.45:
            texto += " ⚠️ La confianza general es baja: revisa la imagen (ángulo, iluminación o resolución de análisis)."
        return texto
    if estado.get("usar_detector"):
        disc = estado.get("discrepancias", [])
        if disc:
            texto += (f" Atención: la CNN y el detector de vehículos NO coinciden en {len(disc)} espacio(s) "
                      f"({_lista(disc, 8)}).")
        else:
            texto += " La CNN y el detector de vehículos coinciden en todos los espacios."
    if estado["confianza_promedio"] < umbral_conf:
        texto += " ⚠️ La confianza general es baja: el resultado podría no ser fiable (iluminación, lluvia o ángulo de cámara)."
    return texto


def responder(intencion: str, estado: dict | None, rng=None) -> str:
    rng = rng or random.Random()
    if intencion == "saludo":
        return rng.choice(["¡Hola! Soy el asistente del parqueadero. ", "¡Buenas! ", "¡Hola! ¿Qué más? "]) + \
            "Puedo decirte cuántos espacios hay libres, el % de ocupación, dónde parquear o qué tan confiable es la detección."
    if intencion == "despedida":
        return rng.choice(["¡Hasta luego! Que encuentres buen puesto. 🚗", "¡Chao! Aquí estaré si necesitas parquear.",
                           "¡Con gusto! Nos vemos."])
    if intencion == "fuera_de_tema":
        return ("Eso está fuera de lo que sé hacer 😅. Solo respondo sobre el parqueadero analizado. Prueba con: "
                + " / ".join(SUGERENCIAS[:4]))
    if not estado:
        return "Todavía no he analizado ninguna imagen o video. Sube uno en la barra lateral y pregúntame de nuevo."

    L, O, T, P = estado["libres"], estado["ocupados"], estado["total"], estado["porcentaje_ocupacion"]
    if intencion == "disponibilidad":
        if L == 0:
            return f"No, en este momento no hay espacios libres: los {T} están ocupados (100 % de ocupación)."
        return (f"Sí, hay {L} de {T} espacios libres ({P:.0f} % de ocupación). "
                f"Libres: {_lista(estado['espacios_libres'], 8)}.")
    if intencion == "porcentaje":
        nivel = "casi lleno" if P >= 90 else "bastante ocupado" if P >= 70 else "con ocupación media" if P >= 40 else "bastante vacío"
        return f"El parqueadero está al {P:.1f} % de ocupación ({O} ocupados de {T}); está {nivel}."
    if intencion == "ubicacion":
        if L == 0:
            return "No hay espacios libres en este momento, lo siento."
        filas = _por_fila(estado["espacios_libres"])
        detalle = "; ".join(f"fila {f}: {_lista(ids, 10)}" for f, ids in sorted(filas.items()))
        seguros = [r["id"] for r in estado["espacios"] if r["estado"] == "libre" and not r.get("dudoso")]
        rec = f" Te recomiendo {seguros[0]}, que es un libre con alta confianza." if seguros else ""
        return f"Puedes parquear en cualquiera de estos {L} espacios libres → {detalle}.{rec}"
    if intencion == "conteo_ocupados":
        txt = f"Hay {O} espacios ocupados de {T} (quedan {L} libres)."
        if estado.get("usar_detector") and estado.get("vehiculos_detectados") is not None:
            txt += f" El detector de vehículos encontró {estado['vehiculos_detectados']} vehículos en la imagen."
        return txt
    if intencion == "confianza":
        return retroalimentacion(estado)
    if intencion == "resumen":
        return (f"📋 Reporte: {L} de {T} espacios libres ({P:.0f} % de ocupación, {O} ocupados). "
                f"Libres: {_lista(estado['espacios_libres'], 10)}. " + retroalimentacion(estado))
    return no_entendi()


def no_entendi() -> str:
    return "No entendí bien tu pregunta 🤔. Puedes preguntarme, por ejemplo: " + " · ".join(SUGERENCIAS)
