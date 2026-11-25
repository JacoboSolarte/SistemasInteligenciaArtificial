# Reporte del proyecto — Sistema inteligente de detección de ocupación de parqueaderos

**Curso:** Sistemas de Inteligencia Artificial — Proyecto final (Proyecto 7)
**Entregables cubiertos:** código fuente (carpeta `parqueadero-ia/`), este reporte, guion de la presentación de
10 minutos (sección 11) y demostración funcional con imágenes y videos (sección 10).
**Aplicación en línea:** <https://parqueadero-ia.streamlit.app/>

---

## 1. Problema

Buscar parqueadero hace perder tiempo y combustible y genera congestión. La idea es que una cámara (fija en
un edificio o poste, o un dron) diga automáticamente **cuántos espacios hay libres, cuántos ocupados, qué
porcentaje de ocupación tiene el parqueadero y dónde están los espacios libres**, y que un usuario pueda
**preguntarle en lenguaje natural** ("¿hay cupo?", "¿dónde parqueo?") a un asistente.

Dificultades reales del problema:

* Cambios de iluminación: sol, nubes, lluvia y noche.
* Sombras y oclusiones: un carro tapa a otro.
* Ángulos de cámara muy distintos: dron, edificio, poste, suelo.
* Carros mal estacionados.
* Cada parqueadero es diferente.

## 2. Objetivos

**General:** construir un sistema que analice imágenes y videos de parqueaderos, detecte vehículos y espacios,
clasifique cada espacio como libre u ocupado, calcule la ocupación y responda preguntas en español mediante un
asistente basado en Transformers.

**Específicos:**

1. Diseñar y entrenar **desde cero** (sin pesos preentrenados, sin YOLO ni librerías de detección ya hechas)
   cuatro modelos:
   * una CNN clasificadora de espacios;
   * un detector de vehículos;
   * un Transformer para entender preguntas;
   * un detector universal de espacios y vehículos.
2. Comparar métodos de **gradiente descendente** (SGD con momentum vs. Adam).
3. Optimizar hiperparámetros con **Hill Climbing** y **estrategias evolutivas**.
4. Construir una aplicación que funcione con **imágenes y videos** y muestre los resultados de forma visual.
5. Evaluar con métricas estándar (accuracy, precisión, recall, F1, AUC, mAP@0.5) y ser honestos con las
   limitaciones.

## 3. Metodología (resumen)

1. **Datos:** descarga de datasets públicos, limpieza, unificación de formatos y división
   entrenamiento/validación/prueba (70/15/15) sin fugas entre conjuntos.
2. **Modelos:** diseño capa por capa en PyTorch, entrenamiento desde pesos aleatorios en una GPU NVIDIA
   GTX 1050 (3 GB).
3. **Optimización:**
   * Hill Climbing (básico y de máxima pendiente) y estrategia evolutiva (μ+λ) para elegir los
     hiperparámetros de la CNN.
   * Reentrenamiento con la mejor configuración.
4. **Integración:** pipeline de inferencia con dos modos (calibrado y automático), suavizado temporal para
   video, asistente conversacional y app web en Streamlit.
5. **Evaluación:**
   * Conjuntos de prueba nunca vistos.
   * Pruebas de generalización (otro parqueadero, otro clima).
   * Video, fotos y videos de internet.
   * Pruebas automáticas (pytest, 32 tests).

## 4. Datasets utilizados

Todos son públicos. **Solo se usaron imágenes y etiquetas; ningún modelo ni peso de terceros.**

| Dataset | Qué contiene | Para qué se usó |
|---|---|---|
| **PKLot** (UFPR, Brasil) | 12 416 fotos de 3 parqueaderos (UFPR04, UFPR05, PUCPR) tomadas desde edificios, con sol, nubes y lluvia. Cada foto trae un XML con el contorno de cada espacio y si está ocupado. | CNN, MiniDetector, detector universal, JSON de espacios, video de demostración |
| **CNRPark-EXT** (CNR, Italia) | 144 965 recortes de espacios de 9 cámaras, etiquetados libre/ocupado, con distintos climas | CNN |
| **CARPK** (Hsieh et al., ICCV 2017) | 1 448 fotos de dron de 4 parqueaderos con ~90 000 carros marcados (copia en Roboflow: 1 567 imágenes, 106 613 cajas) | Detector universal (vehículos vistos desde arriba) |
| **Roboflow Universe** (5 datasets, licencia CC BY 4.0) | Dron cenital, cámara lateral en poste, cámara a nivel de suelo, garaje y noche, con cajas de espacio libre/ocupado | Detector universal (variedad de ángulos) |
| **Dataset sintético propio** | 5 400 frases en español (600 por intención) generadas con plantillas, sinónimos, jerga y errores de tipeo | Transformer del asistente |

Preparación importante:

* **CNN:**
  * 133 099 recortes de 64×64 (60 000 de CNRPark-EXT + 73 100 de PKLot).
  * Los recortes de PKLot se generan con la misma transformación de perspectiva que usa la app, para que
    entrenamiento e inferencia vean lo mismo.
* **Video de demostración:** un día completo de UFPR04 (156 fotos) se dejó **fuera del entrenamiento** y con
  él se armó `ejemplos/video_UFPR04.mp4`.
* **Detector universal:**
  * Se unificaron 7 fuentes en un formato común de 3 clases: libre, ocupado, vehículo.
  * Problemas encontrados y resueltos:
    * El CARPK original pide contraseña a los autores; se usó la copia de Roboflow y se le devolvió su
      proporción 16:9.
    * 404 de 459 imágenes del dataset de dron venían con los carros tapados con rectángulos negros; se
      detectaron automáticamente y se descartaron.
    * Se re-particionaron los datos para que copias de una misma imagen no quedaran a la vez en
      entrenamiento y prueba.

## 5. Modelos (los 4 entrenados desde cero)

### Modelo 1 — CNNEspacios (clasificador de espacios) · 548 482 parámetros
* **Entrada:** recorte de un espacio de 64×64 píxeles.
* **Arquitectura:** 3 bloques convolucionales (Conv 3×3 + BatchNorm + ReLU + MaxPool; 16 → 32 → 64 filtros)
  → capa densa de 128 neuronas con Dropout → 2 salidas (libre / ocupado) con softmax.
* **Entrenamiento:**
  * Pérdida de entropía cruzada.
  * Se comparó **SGD con momentum vs. Adam**.
  * Scheduler que baja la tasa de aprendizaje cuando no mejora y parada temprana (early stopping).
  * Aumento de datos: brillo, contraste, volteo, rotación y desenfoque, para simular lluvia o sombras.
* **Salida:** estado del espacio + probabilidad (confianza).

### Modelo 2 — MiniDetector (detector de vehículos) · 2 891 162 parámetros
* **Arquitectura:** detector de una etapa con cuadrícula 13×13 sobre una imagen de 416×416. Cada celda predice
  2 cajas con objetividad, posición y tamaño.
* **Escrito a mano:**
  * la función de pérdida (MSE de coordenadas + entropía cruzada binaria de objetividad);
  * el **NMS** (eliminación de cajas duplicadas);
  * el **mAP@0.5**.
* **Uso:** en el modo calibrado sirve de "segunda opinión" cuando la CNN duda (regla de fusión).

### Modelo 3 — TransformerAsistente (entiende las preguntas) · 246 537 parámetros
* **Tokenizador propio** de subpalabras: vocabulario de 2 797 tokens; tolera errores de tipeo.
* **Arquitectura:** embedding + **codificación posicional sinusoidal** + 2 capas encoder. La **atención
  multi-cabeza (4 cabezas) está escrita a mano**: softmax(QKᵀ/√d)·V.
* **Salida:** clasifica la pregunta en 9 intenciones: disponibilidad, porcentaje, ubicación, conteo,
  confianza, resumen, saludo, despedida y fuera de tema.
* **Respuesta:** la intención elige una plantilla que se llena con los **datos reales** del análisis.
  Si la confianza es < 0.5, responde que no entendió.

### Modelo 4 — DetectorUniversal (modo automático) · 4 477 731 parámetros
* **Qué detecta:** en la imagen completa, y **sin necesidad de marcar nada**, 3 clases: espacio libre, espacio
  ocupado y vehículo.
* **Arquitectura:**
  * Backbone residual propio + "pirámide" que vuelve a subir la resolución hasta 1/4 de la imagen (para ver
    carros diminutos desde un dron).
  * 3 cabezas: mapa de calor de centros, tamaño y desfase. Estilo "objetos como puntos", sin cajas ancla.
* **Pérdida escrita a mano:** focal loss + L1. Si una fuente no etiqueta una clase (CARPK no marca espacios),
  esa clase se ignora en sus imágenes.
* **Postproceso propio:**
  * NMS por clase.
  * "Un espacio libre con un carro adentro pasa a ocupado".
  * **Regla de filas:** los carros alineados forman filas; los huecos entre ellos son espacios libres.
  * **Resolución automática:** si los carros salen diminutos, analiza también a 1280 px.

## 6. Implementación

### 6.1 Tecnologías
Python 3.13 y PyTorch (solo bloques básicos: `Conv2d`, `Linear`, `LayerNorm`, optimizadores). Además:

* OpenCV: imágenes, video y dibujo.
* NumPy y pandas.
* scikit-learn: divisiones estratificadas y métricas.
* Matplotlib: gráficas.
* Streamlit: la app web.
* pytest: pruebas.

Un test (`tests/test_restricciones.py`) revisa automáticamente que el código **no use YOLO, ultralytics,
torchvision ni pesos preentrenados**.

### 6.2 Dos modos de funcionamiento

| | **Modo calibrado** | **Modo automático** |
|---|---|---|
| Requiere | JSON con los espacios de la cámara, que se marca una sola vez | Nada |
| Modelos | CNN + MiniDetector (fusión) | DetectorUniversal + reglas |
| Mejor para | Una cámara fija instalada (máxima precisión) | Cualquier imagen o video: dron, edificio, lateral, suelo |

Para crear el JSON de una cámara nueva está la herramienta `herramientas/marcar_espacios.py`, con un **modo por
filas semiautomático**: se hacen 4 clics por fila completa y se escribe cuántos espacios tiene, y la fila se
divide sola.

### 6.3 Flujo para una IMAGEN
1. Se sube la imagen (.jpg, .jpeg o .png).
2. Se analiza según el modo:
   * **Calibrado:** se recorta cada espacio con corrección de perspectiva → la CNN los clasifica todos a la vez
     → si la CNN duda (< 0.80), el detector desempata.
   * **Automático:** el DetectorUniversal encuentra espacios y vehículos → se aplican las reglas.
3. Se construye el **estado**:
   * libres, ocupados, % de ocupación y confianza promedio;
   * lista de espacios libres y ocupados;
   * espacios dudosos;
   * vehículos detectados.
4. Se dibuja: **verde = libre, rojo = ocupado, azul = vehículo, borde amarillo = dudoso**, con un panel de
   conteos.
5. El asistente responde preguntas usando ese estado.

### 6.4 Flujo para un VIDEO
1. Se sube el video (.mp4, .avi, .mov o .mkv).
2. Se procesa **1 de cada K frames** (configurable) para ir más rápido.
3. **Suavizado temporal:** un espacio solo cambia de estado si el cambio se mantiene **N frames seguidos**.
   Así se evita el parpadeo por una sombra o un carro que pasa. En modo automático, un **seguidor por IoU**
   mantiene la identidad de cada espacio entre frames.
4. Resultados:
   * un **video anotado** descargable;
   * una **gráfica de ocupación en el tiempo**;
   * un deslizador para ver cualquier frame y preguntarle al asistente por ese momento.
5. **Velocidad en la GTX 1050:** unos 0.15–0.2 s por frame en modo automático.

### 6.5 Aplicación web (Streamlit)

Desplegada en Streamlit Community Cloud: <https://parqueadero-ia.streamlit.app/> (PyTorch solo CPU; entrada `despliegue/streamlit_app.py`).
* **Barra lateral:**
  1. Fuente: imagen o video, de ejemplo o subido.
  2. Modo de detección.
  3. Parámetros: umbrales, resolución, suavizado.
* **Pestaña Análisis:** métricas, imagen o video anotado, gráfica temporal, tabla por espacio y descarga del
  estado en JSON.
* **Pestaña Asistente:** chat en español con el Transformer.
* **Pestaña Resultados del entrenamiento:** todas las gráficas y tablas de evaluación.

### 6.6 Estructura del código
```
parqueadero-ia/
├── app.py                    app web
├── config/                   config.yaml (todos los hiperparámetros) + JSON de espacios
├── scripts/                  descarga y preparación de datos
├── modelos/                  los 4 modelos
├── entrenamiento/            scripts de entrenamiento
├── optimizacion/             Hill Climbing y estrategia evolutiva
├── asistente/                tokenizador, dataset sintético, respuestas
├── inferencia/               pipeline calibrado, automático, regla de filas, NMS, fusión, dibujo
├── herramientas/             marcado de espacios, conversor PKLot→JSON, creador de video
├── evaluacion/               scripts de evaluación y métricas escritas a mano
├── tests/                    32 pruebas automáticas
├── pesos/                    modelos entrenados (.pth)
├── resultados/               gráficas, tablas y registros
└── notebooks/                notebook para Google Colab
```

## 7. Resultados

### 7.1 CNN — SGD vs. Adam (conjunto de prueba, 19 965 recortes)

| Optimizador | Accuracy | F1 |
|---|---|---|
| SGD con momentum | 0.9936 | 0.9937 |
| Adam | 0.9943 | 0.9944 |
| **Adam + mejor configuración de Hill Climbing (modelo final)** | **0.9954** | **0.9955** (AUC 0.9993) |

**Generalización** (situaciones nunca vistas en entrenamiento):

| Experimento | F1 |
|---|---|
| Otro parqueadero: entrena con UFPR04 + UFPR05, prueba con PUCPR | 0.9886 |
| Otro clima: entrena con soleado, prueba con lluvioso | 0.9890 |

### 7.2 Optimización de hiperparámetros

| Método | Evaluaciones | Mejor F1 en validación |
|---|---|---|
| **Hill Climbing básico** | 13 | **0.9766** (lr 8.46e-4, dropout 0.30) |
| Hill Climbing de máxima pendiente | 13 | 0.9745 |
| Estrategia evolutiva (μ+λ) = (3+6) | 27 | 0.9739 |

Las diferencias entre métodos son pequeñas (del orden del ruido de entrenar solo 3 épocas por evaluación). La
mejor configuración se usó para el modelo final.

### 7.3 MiniDetector (prueba, 424 imágenes)

| | mAP@0.5 | Precisión | Recall |
|---|---|---|---|
| **Total** | **0.922** | 0.977 | 0.926 |
| UFPR04 | 0.989 | | |
| UFPR05 | 0.997 | | |
| PUCPR | 0.874 | | |

### 7.4 Asistente (Transformer)

| Conjunto | Accuracy de intención |
|---|---|
| Frases de prueba no vistas | **0.947** (810 frases) |
| 40 preguntas escritas a mano, con un estilo distinto | 0.85 |

### 7.5 Sistema completo en video (día de UFPR04 no usado en entrenamiento)

| Configuración | Acierto por frame |
|---|---|
| Solo CNN, con suavizado N = 2 | 99.66 % |
| **CNN + detector (fusión), con suavizado N = 2** | **99.86 %** |

### 7.6 DetectorUniversal — modo automático (prueba, imágenes nunca vistas)

| Tipo de imagen | AP@0.5 | Error en el % de ocupación |
|---|---|---|
| Dron, vehículos (CARPK) | **0.968** | — |
| Cámara en edificio (PKLot) | **0.94–0.97** | 5.0 puntos |
| Cámara lateral | 0.99 (prueba pequeña, optimista) | 1.8 puntos |
| Garaje | 0.88–0.92 | 7.2 puntos |
| Dron con espacios | 0.69–0.82 | 4.8 puntos |
| Nivel de suelo | 0.67–0.75 | 18.6 puntos |
| Noche | 0.02–0.51 | 47.8 puntos (no funciona) |

**mAP@0.5 medio:** 0.784.

**Fotos y videos de dron de internet** (sin etiquetas, revisión visual): en una foto con unos 22 carros, el
sistema reporta 22 ocupados y 75 libres, alineados con la demarcación del piso, **sin marcar nada a mano**.

## 8. Conclusiones

1. Es posible resolver el problema con **modelos pequeños entrenados desde cero**. Una CNN de solo 0.5 M
   parámetros alcanza **F1 = 0.9955** clasificando espacios y generaliza a otro parqueadero y a otro clima
   (F1 ≈ 0.99).
2. **Adam** convergió más rápido y un poco mejor que SGD. **Hill Climbing** encontró la mejor configuración con
   pocas evaluaciones; la estrategia evolutiva no superó a Hill Climbing con ese presupuesto tan pequeño.
3. **Combinar modelos** mejora el resultado: la fusión CNN + detector subió el acierto en video de 99.66 % a
   99.86 %.
4. El **Transformer** con atención escrita a mano entiende preguntas en español informal (0.947) y, al llenar
   plantillas con datos reales, **nunca inventa cifras**.
5. El **modo automático** quita la necesidad de calibrar y funciona con vistas muy distintas (dron, edificio,
   lateral). Pero la calidad depende de cuántos datos etiquetados hay de cada tipo de vista: es excelente donde
   hubo muchos datos (dron con vehículos, edificio) y débil donde hubo pocos (noche).
6. **Ser honestos con los datos fue clave.** Revisar los datasets permitió encontrar imágenes tapadas con
   negro, imágenes estiradas y posibles fugas entre entrenamiento y prueba, que habrían dado métricas falsas.

## 9. Limitaciones y trabajo futuro
* **Noche:** casi no hay datos; el modo automático no sirve de noche.
* **Vista de dron:** los espacios libres en los extremos de una fila o en filas vacías no se infieren.
* **Precisión:** para una cámara fija, el modo calibrado es más preciso que el automático.
* **Asistente:** es de clasificación + plantillas, no genera texto libre.
* **Trabajo futuro:**
  * Más datos nocturnos y de dron.
  * Detectar las líneas pintadas del piso.
  * Generación de texto en el asistente.

## 10. Demostración funcional (cómo correrlo)

```
cd parqueadero-ia
.venv\Scripts\activate
streamlit run app.py          → abrir http://localhost:8501
```

Demostración sugerida:

1. **Imagen, modo automático:** subir una foto aérea → muestra libres, ocupados y vehículos sin calibrar nada.
2. **Imagen, modo calibrado:**
   * Elegir el ejemplo `PUCPR_soleado.jpg` con `espacios_PUCPR.json`.
   * Muestra la confianza por espacio.
   * Muestra la tabla de detalle por espacio.
3. **Video:**
   * Elegir `video_UFPR04.mp4` (modo calibrado) o un video de dron (modo automático).
   * Muestra el video anotado y la gráfica de ocupación en el tiempo.
4. **Asistente:** preguntar "¿hay cupo?", "¿dónde parqueo?", "¿qué tan lleno está?", "¿qué tan seguro estás?",
   "¿cuál es la capital de Francia?" (fuera de tema).
5. **Pestaña Resultados del entrenamiento:** mostrar curvas, matrices de confusión, curva PR y comparación de
   optimizadores.

## 11. Guion de la presentación (10 minutos)

| Minuto | Contenido |
|---|---|
| 0–1 | Problema y objetivo |
| 1–2 | Datasets y preparación (PKLot, CNRPark-EXT, CARPK, Roboflow; limpieza de datos) |
| 2–4 | Los 4 modelos desde cero; en qué se diferencian; qué está escrito a mano |
| 4–5 | Gradiente descendente (SGD vs. Adam) y optimización (Hill Climbing, estrategia evolutiva) |
| 5–8 | **Demostración en vivo:** imagen automática → imagen calibrada → video → asistente |
| 8–9 | Resultados principales (tablas de la sección 7) |
| 9–10 | Conclusiones, limitaciones y trabajo futuro |

## 12. Preguntas probables en la sustentación (y respuestas cortas)

* **¿Por qué desde cero y no YOLO?**
  * Es la restricción del curso, y obliga a entender cada pieza: pérdida, NMS, atención, optimizador.
  * Además el problema lo permite: la CNN logró > 99 % sin transferencia de aprendizaje.
* **¿Qué es el NMS?**
  * Elimina detecciones duplicadas: ordena las cajas por puntaje, conserva la mejor y borra las que se solapan
    con ella más de un umbral de IoU.
  * Lo implementamos a mano en `inferencia/nms.py`.
* **¿Qué es el mAP@0.5?**
  * Es el área bajo la curva precisión–recall.
  * Una detección cuenta como correcta si su IoU con la caja real es ≥ 0.5.
* **¿Cómo funciona la atención?**
  * Cada palabra calcula qué tanto "mirar" a las demás: softmax(QKᵀ/√d)·V.
  * Con varias cabezas se aprenden relaciones distintas.
* **¿Por qué Adam fue mejor que SGD?**
  * Adapta la tasa de aprendizaje de cada parámetro con promedios del gradiente y de su cuadrado.
  * Converge más rápido con menos ajuste manual.
* **¿Hill Climbing vs. estrategia evolutiva?**
  * Hill Climbing mejora una sola solución con pequeños cambios.
  * La estrategia evolutiva mantiene una población de μ padres, genera λ hijos por mutación y se queda con los
    mejores. Explora más, pero necesita más evaluaciones.
* **¿Qué pasa si la cámara se mueve?**
  * En modo calibrado hay que volver a marcar los espacios.
  * El modo automático no depende de eso.
* **¿Cómo evitaron que el video "parpadee"?**
  * Con el suavizado temporal: un espacio cambia de estado solo si el nuevo estado se mantiene N frames.
* **¿El asistente inventa datos?**
  * No. El Transformer solo decide la intención, y la respuesta se llena con los números del análisis.
* **¿Cuál es la principal limitación?**
  * La noche y los tipos de vista con pocos datos etiquetados.
  * Lo medimos y lo reportamos en lugar de ocultarlo.
