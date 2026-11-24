"""
Genera el dataset sintético de intenciones en español (con variaciones colombianas).

Técnicas: plantillas + sinónimos + prefijos/sufijos conversacionales + ruido
(errores de tipeo, frases sin tildes, sin signos, mayúsculas aleatorias).

    python asistente/generar_dataset.py   ->  datos/procesados/asistente/{train,val,test}.csv
"""
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asistente.tokenizador import normalizar  # noqa: E402
from utilidades.comun import cargar_config, ruta  # noqa: E402

INTENCIONES = ["disponibilidad", "porcentaje", "ubicacion", "conteo_ocupados", "confianza",
               "resumen", "saludo", "despedida", "fuera_de_tema"]

S = {  # sinónimos
    "puesto": ["puesto", "puestos", "espacio", "espacios", "cupo", "cupos", "lugar", "lugares", "parqueadero",
               "celda", "celdas", "plaza", "plazas", "sitio", "sitios"],
    "libre": ["libre", "libres", "disponible", "disponibles", "desocupado", "desocupados", "vacío", "vacíos"],
    "carro": ["carro", "carros", "vehículo", "vehículos", "auto", "autos", "coche", "coches", "nave", "naves", "camioneta", "camionetas"],
    "parq": ["el parqueadero", "el parqueo", "el estacionamiento", "la zona", "el lote", "el parking", "esto"],
}
PREFIJOS = ["", "", "", "oye ", "hola, ", "parce ", "una pregunta, ", "disculpa, ", "por favor ", "oiga ", "hey ",
            "bueno ", "y ", "me puedes decir ", "quisiera saber ", "necesito saber ", "dime ", "me dices "]
SUFIJOS = ["", "", "", " por favor", " porfa", " ahora", " en este momento", " ya", " hoy", " pues", " ahorita", " gracias"]

PLANTILLAS = {
    "disponibilidad": [
        "hay {puesto} {libre}", "hay cupo", "quedan {puesto}", "queda algún {puesto} {libre}",
        "hay donde parquear", "puedo parquear", "hay {puesto} para mi {carro}", "cuántos {puesto} {libre} hay",
        "cuántos {puesto} quedan", "hay disponibilidad", "tienen {puesto} {libre}", "hay campo", "todavía hay cupo",
        "alcanzo a encontrar {puesto}", "está full {parq}", "está lleno {parq}", "hay {puesto} o está lleno",
        "me puedo estacionar", "cuántos {libre} hay", "existe algún {puesto} {libre}", "queda cupo en {parq}",
        "hay chance de parquear", "cabe otro {carro}", "hay espacio para estacionar", "número de {puesto} {libre}",
        "quedan {puesto} {libre} en {parq}", "se puede parquear", "consigo {puesto}", "cuántos cupos hay",
        "hay {puesto} desocupados", "disponibilidad de {puesto}", "cuántas plazas quedan", "algún {puesto} {libre}",
    ],
    "porcentaje": [
        "qué tan lleno está {parq}", "qué porcentaje de ocupación hay", "cuál es el porcentaje de ocupación",
        "qué tan ocupado está {parq}", "en qué porcentaje está {parq}", "cuánto está ocupado {parq}",
        "nivel de ocupación", "porcentaje de {puesto} ocupados", "tasa de ocupación", "qué tan full está",
        "está muy lleno {parq}", "está medio vacío {parq}", "cómo está de ocupado {parq}", "ocupación en porcentaje",
        "dame el porcentaje", "qué proporción está ocupada", "cuánto porcentaje hay ocupado", "qué tan congestionado está",
        "está a reventar {parq}", "qué % de ocupación tiene", "índice de ocupación de {parq}", "a qué capacidad está {parq}",
        "qué tanto se ha llenado {parq}", "está casi lleno", "ocupación actual", "cuánto falta para que se llene",
    ],
    "ubicacion": [
        "dónde puedo parquear", "qué {puesto} están {libre}", "cuáles {puesto} están {libre}", "dónde hay {puesto} {libre}",
        "en qué {puesto} me hago", "dónde estaciono", "dónde hay cupo", "cuál {puesto} está {libre}",
        "dime los {puesto} {libre}", "lista de {puesto} {libre}", "qué números de {puesto} están {libre}",
        "ubicación de los {puesto} {libre}", "dónde dejo el {carro}", "en cuál puedo parquear", "dónde me parqueo",
        "qué fila tiene {puesto} {libre}", "muéstrame los {puesto} {libre}", "cuáles quedan {libre}",
        "por dónde hay {puesto}", "dónde queda un {puesto} {libre}", "identificadores de los {puesto} {libre}",
        "en qué parte hay {puesto}", "qué {puesto} me recomiendas", "cuál me sirve para parquear", "ids de {puesto} {libre}",
    ],
    "conteo_ocupados": [
        "cuántos {carro} hay", "cuántos {puesto} están ocupados", "cuántos {carro} hay parqueados",
        "cuántos ocupados hay", "número de {carro}", "cuántos {carro} detectaste", "cuántos {carro} ves",
        "cantidad de {carro} en {parq}", "cuántos {puesto} ocupados", "cuántos {carro} están estacionados",
        "cuántos {carro} hay en {parq}", "conteo de {carro}", "cuenta los {carro}", "cuántos {puesto} usados hay",
        "cuántos {carro} encontraste", "total de {carro} parqueados", "cuántos {puesto} tienen {carro}",
        "cuántas naves hay", "cuántos están ocupados", "cuánta gente parqueó", "qué cantidad de {carro} hay",
        "cuántos {carro} hay ahorita", "dime cuántos {carro} hay",
    ],
    "confianza": [
        "qué tan seguro estás", "es confiable la detección", "qué tan confiable es el resultado", "estás seguro",
        "qué tan precisa es la detección", "puedo confiar en el resultado", "cuál es la confianza", "qué confianza tiene el modelo",
        "qué tan buena es la predicción", "hay errores en la detección", "qué {puesto} tienen baja confianza",
        "en qué {puesto} dudas", "el modelo se puede equivocar", "qué tan exacto es", "nivel de confianza",
        "qué tan fiable es", "seguro que está bien", "la cnn está segura", "cuál es la certeza", "hay {puesto} dudosos",
        "es correcto el análisis", "qué tan acertado es el sistema", "probabilidad de error", "confías en la detección",
        "qué tan seguro es el algoritmo", "coinciden el detector y la cnn", "hay discrepancias",
    ],
    "resumen": [
        "dame un reporte", "dame un resumen", "resumen del parqueadero", "cómo está {parq}", "cuál es el estado de {parq}",
        "infórmame del estado", "dame el estado general", "haz un reporte completo", "qué hay en {parq}", "estado actual",
        "reporte de ocupación", "dame toda la información", "cuéntame cómo está todo", "informe del análisis",
        "resultado del análisis", "qué encontraste", "dame los resultados", "dame el panorama", "cómo va {parq}",
        "muéstrame el resumen", "dame el informe", "qué detectaste en la imagen", "cuál fue el resultado",
        "reporte general", "qué tal está {parq}",
    ],
    "saludo": [
        "hola", "buenos días", "buenas tardes", "buenas noches", "buenas", "qué más", "quiubo", "hola cómo estás",
        "hey hola", "saludos", "qué tal", "hola asistente", "buen día", "holi", "qué hubo", "hola bot", "ey qué más",
        "cómo estás", "mucho gusto", "hola qué tal", "buenas buenas", "hello", "qué cuentas", "hola parce",
    ],
    "despedida": [
        "adiós", "chao", "hasta luego", "nos vemos", "gracias chao", "muchas gracias", "eso es todo", "bye",
        "hasta pronto", "me voy", "listo gracias", "chao gracias", "hasta mañana", "ya no necesito nada", "gracias por todo",
        "nos pillamos", "que estés bien", "terminamos", "salir", "hasta la próxima", "listo eso era todo", "fue un gusto",
    ],
    "fuera_de_tema": [
        "cuál es la capital de francia", "cuéntame un chiste", "qué hora es", "cómo está el clima", "quién ganó el partido",
        "recomiéndame una película", "cuánto es dos más dos", "qué es la fotosíntesis", "cómo hago un pastel",
        "quién es el presidente", "me gusta el fútbol", "qué música te gusta", "cuál es tu color favorito",
        "dónde queda bogotá", "cómo se dice hola en inglés", "qué es el amor", "cuántos años tienes", "cuál es el mejor celular",
        "háblame de historia", "cómo aprendo a programar", "dame una receta", "qué opinas de la política",
        "quién descubrió américa", "cuánto cuesta un dólar", "escribe un poema", "qué es python", "cómo se calcula el área de un círculo",
        "tengo hambre", "dónde compro zapatos", "va a llover mañana", "cuál es el río más largo", "pon música",
        "qué día es hoy", "me aburro", "cómo bajo de peso", "traduce esto al inglés", "quién es messi",
        "cuántos planetas hay", "explícame la relatividad", "qué serie me recomiendas",
    ],
}

# Plantillas adicionales (más variedad léxica general)
PLANTILLAS["disponibilidad"] += [
    "hay algo {libre}", "queda algo", "alcanza para un {carro} más", "hay hueco",
    "tengo dónde dejar el {carro}", "puedo entrar con el {carro}",
    "todavía cabe alguien", "ya no hay {puesto}", "se llenó o todavía hay", "voy llegando hay {puesto}", "puedo ir a parquear",
    "habrá {puesto} {libre}", "sí hay {puesto}", "hay {puesto} o no", "qué tal la disponibilidad", "me sirve ir para allá a parquear",
]
PLANTILLAS["porcentaje"] += [
    "cuánto se ha llenado", "qué tanto está lleno", "va por la mitad o qué", "ya casi se llena", "está vacío o lleno",
    "qué tan copado está", "qué porcentaje lleva", "en cuánto va la ocupación", "dime la ocupación en %", "qué tan saturado está {parq}",
]
PLANTILLAS["ubicacion"] += [
    "qué {puesto} me sirve", "en cuál me meto", "señálame un {puesto} {libre}", "cuál está vacío", "qué celdas sirven",
    "cuál {puesto} queda más cerca", "por qué lado hay {puesto}", "en qué zona hay {puesto} {libre}", "dime cuál puedo usar",
    "qué códigos tienen los {puesto} {libre}", "cuáles son los {libre}", "qué número de {puesto} uso", "a cuál me voy",
]
PLANTILLAS["conteo_ocupados"] += [
    "cuántos hay parqueados", "cuántos {carro} contaste", "cuántos {carro} hay adentro",
    "cuántos {puesto} se están usando", "cuántos {carro} metidos hay", "cuántos {puesto} llenos hay", "cuántos {carro} quedaron",
]
PLANTILLAS["confianza"] += [
    "te puedes equivocar", "estás seguro de eso", "qué tan bien funcionó la red", "le puedo creer al modelo", "eso es verdad",
    "qué margen de error tiene", "qué tan precisa fue la red neuronal", "funcionó bien la detección",
    "hay {puesto} con dudas", "cuál es la probabilidad de acierto", "qué tan seguro está el sistema",
]
PLANTILLAS["resumen"] += [
    "cómo está todo", "hazme un resumen", "dame un informe rápido", "qué pasó en el análisis", "resume el estado",
    "dame un balance", "cómo salió el análisis", "dame los datos generales", "cuéntame todo del parqueadero",
]
PLANTILLAS["saludo"] += ["holiii", "hola hola", "buenas noches parce", "hola cómo te va", "hola qué más pues", "épale", "buenos días señor"]
PLANTILLAS["despedida"] += ["gracias por la info", "muy amable chao", "listo me sirvió gracias",
                            "ya me voy gracias", "hasta luego gracias", "perfecto gracias chao"]
TEMAS = ["el fútbol", "la historia de colombia", "matemáticas", "física", "la música", "cocina", "películas", "el clima",
         "política", "literatura", "videojuegos", "la economía", "el dólar", "un libro", "animales", "la luna", "el universo",
         "la selección", "programación", "química", "biología", "un chiste", "la novela", "las elecciones", "un poema"]
PLANTILLAS["fuera_de_tema"] += [f"{v} {t}" for v in ("háblame de", "qué sabes de", "explícame", "cuéntame sobre", "qué opinas de",
                                                  "quién sabe de", "me ayudas con") for t in TEMAS]
PLANTILLAS["fuera_de_tema"] += ["cuánto es 15 al cuadrado", "quién escribió el quijote", "cómo preparo arroz", "qué equipo es el mejor",
                                "cuál es la montaña más alta", "dónde queda medellín", "cuántos habitantes tiene colombia",
                                "cómo se hace una arepa", "quién pintó la mona lisa", "cuál es tu nombre", "me cuentas algo divertido",
                                "cuánto mide la torre eiffel", "qué carrera estudio", "resuelve 3 por 7"]

# Fuera de tema generado por combinación (preguntas de cultura general) + texto sin sentido.
# Sirve para que el modelo aprenda a NO forzar una intención del parqueadero cuando la pregunta es de otro tema.
INTERROGATIVOS = ["cuál es", "qué es", "quién es", "dónde queda", "cuándo fue", "cómo funciona", "por qué existe", "cuánto vale",
                  "qué significa", "para qué sirve"]
OBJETOS = ["la capital de alemania", "el océano más grande", "el mejor restaurante", "la fórmula del agua", "un agujero negro",
           "el presidente de méxico", "la independencia", "la inflación", "un volcán", "el internet", "la gravedad",
           "un número primo", "la vacuna", "el cantante más famoso", "una galaxia", "el himno nacional", "la bolsa de valores",
           "el reguetón", "la inteligencia emocional", "el mundial", "un dinosaurio", "la democracia", "un eclipse", "el café",
           "la capital de italia", "el país más grande", "la segunda guerra mundial", "el bitcoin", "la célula", "un huracán"]
LETRAS = "abcdefghijklmnopqrstuvwxyz"


def frase_fuera_de_tema_generada(rng):
    if rng.random() < 0.25:     # texto sin sentido
        return " ".join("".join(rng.choice(LETRAS) for _ in range(rng.randint(2, 8))) for _ in range(rng.randint(1, 4)))
    return f"{rng.choice(INTERROGATIVOS)} {rng.choice(OBJETOS)}"

VECINAS = {  # teclas vecinas para simular errores de tipeo
    "a": "sq", "e": "wr", "i": "uo", "o": "ip", "u": "yi", "s": "ad", "r": "et", "t": "ry", "n": "bm", "l": "kñ",
    "c": "xv", "d": "sf", "p": "o", "m": "n", "b": "vn", "v": "cb", "q": "w", "h": "gj",
}


def quitar_tildes(t):
    return t.translate(str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU"))


def error_tipeo(texto, rng):
    palabras = texto.split()
    if not palabras:
        return texto
    i = rng.randrange(len(palabras))
    p = palabras[i]
    if len(p) < 4:
        return texto
    j = rng.randrange(1, len(p) - 1)
    op = rng.choice(["borrar", "duplicar", "intercambiar", "vecina"])
    if op == "borrar":
        p = p[:j] + p[j + 1:]
    elif op == "duplicar":
        p = p[:j] + p[j] + p[j:]
    elif op == "intercambiar":
        p = p[:j] + p[j + 1] + p[j] + p[j + 2:]
    elif p[j] in VECINAS:
        p = p[:j] + rng.choice(VECINAS[p[j]]) + p[j + 1:]
    palabras[i] = p
    return " ".join(palabras)


def rellenar(plantilla, rng):
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(S[m.group(1)]), plantilla)


def generar_frase(intencion, rng, k=None):
    """k: índice de plantilla (recorrido circular para que TODAS las plantillas aparezcan); None = al azar."""
    plantillas = PLANTILLAS[intencion]
    if intencion == "fuera_de_tema" and rng.random() < 0.4:
        base = frase_fuera_de_tema_generada(rng)
    else:
        base = rellenar(plantillas[k % len(plantillas)] if k is not None else rng.choice(plantillas), rng)
    if intencion in ("saludo", "despedida"):
        pref = rng.choice(["", "", "", "bueno ", "ok ", "listo "]) if intencion == "despedida" else ""
        suf = rng.choice(["", "", "", " amigo", " parce", " bot", "!"])
        frase = pref + base + suf
    else:
        frase = rng.choice(PREFIJOS) + base + rng.choice(SUFIJOS)
    # signos de pregunta (a veces solo el de cierre, como se escribe en el celular)
    if intencion not in ("saludo", "despedida") and rng.random() < 0.6:
        frase = ("¿" if rng.random() < 0.5 else "") + frase + "?"
    if rng.random() < 0.35:
        frase = quitar_tildes(frase)
    if rng.random() < 0.25:
        frase = error_tipeo(frase, rng)
    if rng.random() < 0.08:
        frase = error_tipeo(frase, rng)
    if rng.random() < 0.15:
        frase = frase.upper() if rng.random() < 0.3 else frase.capitalize()
    return re.sub(r"\s+", " ", frase).strip()


def generar(n_por_intencion=400, semilla=42):
    rng = random.Random(semilla)
    filas = []
    for it in INTENCIONES:
        vistas, intentos = set(), 0
        while len(vistas) < n_por_intencion and intentos < n_por_intencion * 60:
            f = generar_frase(it, rng, intentos)
            intentos += 1
            clave = normalizar(f)
            if clave and clave not in vistas:
                vistas.add(clave)
                filas.append({"texto": f, "intencion": it})
    return pd.DataFrame(filas)


def main():
    cfg = cargar_config()
    df = generar(cfg["asistente"]["frases_por_intencion"], cfg["semilla"])
    # Eliminar frases cuya forma normalizada aparezca en dos intenciones distintas (ambiguas)
    df["norm"] = df.texto.map(normalizar)
    ambiguas = df.groupby("norm").intencion.nunique()
    df = df[~df.norm.isin(ambiguas[ambiguas > 1].index)].drop(columns="norm").reset_index(drop=True)
    tr, resto = train_test_split(df, test_size=0.30, stratify=df.intencion, random_state=cfg["semilla"])
    va, te = train_test_split(resto, test_size=0.50, stratify=resto.intencion, random_state=cfg["semilla"])
    salida = ruta(cfg, "datos") / "procesados" / "asistente"
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, d in (("train", tr), ("val", va), ("test", te)):
        d.to_csv(salida / f"{nombre}.csv", index=False, encoding="utf-8")
    df.to_csv(salida / "dataset_completo.csv", index=False, encoding="utf-8")
    print(df.intencion.value_counts().to_string())
    print(f"train={len(tr)} val={len(va)} test={len(te)}  ->  {salida}")
    print(df.sample(12, random_state=1).to_string(index=False))


if __name__ == "__main__":
    main()
