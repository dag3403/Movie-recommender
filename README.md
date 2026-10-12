# Netflix Movie Insights

Aplicación local para explorar un catálogo de Netflix, consultar su distribución en
clusters y obtener recomendaciones de películas similares. Incluye un análisis
reproducible de la calidad de los clusters y comparaciones entre varias representaciones
de los datos.

## Contenido

- [Qué hace](#qué-hace)
- [Datos y representaciones](#datos-y-representaciones)
- [Resultados y conclusiones](#resultados-y-conclusiones)
- [Requisitos e instalación](#requisitos-e-instalación)
- [Ejecutar la aplicación web](#ejecutar-la-aplicación-web)
- [Ejecutar la aplicación de terminal](#ejecutar-la-aplicación-de-terminal)
- [Reproducir el análisis](#reproducir-el-análisis)
- [Pruebas y comprobaciones](#pruebas-y-comprobaciones)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Publicar en GitHub](#publicar-en-github)
- [Licencia y uso de los datos](#licencia-y-uso-de-los-datos)

## Qué hace

La aplicación ofrece una interfaz web local con cuatro vistas:

- **Resumen:** tamaño del catálogo, películas y estadísticas de los clusters.
- **Clusters:** distribución y listado paginado de películas agrupadas por
  características o descripción.
- **Recomendador:** elige una película, un enfoque y cuántas recomendaciones quieres.
  Las sugerencias se buscan en todas las películas elegibles del enfoque seleccionado,
  sin restringirse al cluster, y se ordenan por distancia a la película de referencia.
- **Catálogo:** búsqueda parcial sin distinguir mayúsculas, ordenación y paginación.
  Las películas se pueden seleccionar para iniciar una recomendación.

El catálogo se procesa y los clusters se calculan al iniciar la aplicación. Las
interacciones reutilizan esos resultados; las recomendaciones se almacenan en una
caché en memoria durante esa ejecución. La aplicación no sobrescribe el CSV ni guarda
datos del usuario.

## Datos y representaciones

El catálogo de entrada está en `data/netflix_titles.csv`. Los dos enfoques solo usan
películas, pero no necesariamente la misma cantidad:

### `char`: características estructuradas

Usa año de lanzamiento, duración, clasificación, países y géneros. Las variables
numéricas se escalan y las categóricas se codifican. Se eliminan los registros que no
tienen completos los campos requeridos. La aplicación ejecuta KMeans con **2 clusters**
y para explorar la distribución de grupos. Las recomendaciones comparan todas las
películas elegibles usando distancia euclídea, independientemente de su cluster.

### `desc`: descripciones de texto

Se usan las películas con descripción y se concatena el título con el texto descriptivo
antes de construir un único vector TF-IDF por película. De ese modo, los términos del
título pueden contribuir a los clusters y a las recomendaciones con el mismo
preprocesamiento que los términos de la descripción. El pipeline TF-IDF de
`src/peliculas/etl.py` aplica:

- `min_df=3` y `max_df=0.85` para filtrar términos por frecuencia documental.
- Unigramas y bigramas (`ngram_range=(1, 2)`).
- Frecuencia sublineal (`sublinear_tf=True`).
- Un máximo de **15 términos con mayor peso TF-IDF por texto combinado**.
- Normalización de cada vector después de seleccionar los términos.

La matriz TF-IDF es dispersa. La aplicación la agrupa con KMeans en **15 clusters** y
usa distancia coseno para ordenar recomendaciones. La matriz y las etiquetas conservan
el orden del mismo conjunto de películas para evitar desalinear títulos y resultados.
Los clusters se usan para explorar y resumir el catálogo; no restringen las candidatas
del recomendador, que busca por distancia coseno en todas las películas con descripción.

## Resultados y conclusiones

Los resultados numéricos siguientes proceden del análisis de
[`notebooks/plots.ipynb`](notebooks/plots.ipynb), con una muestra fija de 1.200
películas por enfoque para las comparaciones de silueta. En `desc`, el baseline
histórico usa solo la descripción, mientras que la variante TF-IDF enfocada combina
título y descripción. Además, algunas siluetas se calculan en espacios distintos, así
que las puntuaciones de `char` y `desc` **no se deben comparar directamente**, ni se
deben comparar siluetas de espacios diferentes como si fueran equivalentes.

### Características (`char`)

- El KMeans de referencia obtiene una silueta aproximada de **0,149** en el espacio de
  características original.
- El experimento de ponderación por bloques obtiene **0,500** en su representación
  ponderada y **0,269** al evaluar las etiquetas en el espacio original.
- La mejora tiene un coste de equilibrio: los clusters contienen **701** y **4.986**
  películas, frente a **1.859** y **3.828** en la referencia.
- El experimento de bloques equilibrados y sus etiquetas mejoradas pertenecen al
  notebook; la aplicación usa el pipeline de características de ETL y KMeans estándar.

### Descripciones (`desc`)

- El baseline TF-IDF + KMeans (solo descripción) obtiene una silueta aproximada de
  **0,0071**; el **46,0 %** de las películas de la muestra tiene silueta negativa.
- La variante enfocada combina título y descripción, limita cada texto combinado a sus
  **15 términos TF-IDF de mayor peso** y genera un vocabulario de **7.979 términos**.
- TF-IDF enfocado + NMF de 15 temas + KMeans obtiene una silueta de **0,5371** en el
  espacio de temas. Al evaluar sus etiquetas en el espacio TF-IDF de referencia, la
  silueta es **0,0056** y el porcentaje de siluetas negativas **37,5 %**, frente a
  **0,0071** y **46,0 %** del baseline. La silueta NMF en su propio espacio no es
  comparable directamente con la del baseline; los resultados no demuestran por sí
  solos una mayor similitud textual general.
- Los clusters del experimento NMF contienen entre **231** y **921** películas, frente
  a **129** y **2.466** en el baseline. La distribución mejora, aunque sigue siendo
  desigual y la calidad semántica requiere revisar ejemplos y términos
  representativos. NMF converge en **130 iteraciones** en esta ejecución.
- **NMF es una comparación experimental del notebook, no el algoritmo de producción
  de la aplicación.** La aplicación usa TF-IDF enfocado y KMeans estándar.

En resumen, las métricas internas son útiles para comparar alternativas dentro de
cada enfoque, pero no bastan para demostrar que las sugerencias son pertinentes para
una persona. El notebook incluye tamaños de cluster, estabilidad entre semillas y
pruebas funcionales del recomendador; conviene revisar ejemplos de películas y temas
antes de interpretar los grupos como categorías semánticas definitivas.

## Requisitos e instalación

- Python **3.12 o posterior**.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) para crear el entorno y
  mantener las dependencias sincronizadas con `uv.lock`.
- Git para clonar el repositorio.

Clona el repositorio y entra en su directorio raíz:

```bash
git clone https://github.com/dag3403/Movie-recommender.git
cd Movie-recommender
```

Instala las dependencias de la aplicación y las herramientas de desarrollo:

```bash
uv sync --locked --group dev
```

El comando crea `.venv` si hace falta y utiliza las versiones bloqueadas en `uv.lock`.
No es necesario activar manualmente el entorno para los comandos `uv run` de esta guía.

## Ejecutar la aplicación web

Desde la raíz del repositorio:

```bash
uv run --locked peliculas-web
```

La aplicación abre <http://127.0.0.1:8000>. La primera carga calcula las características,
vectoriza las descripciones y ajusta los dos KMeans, por lo que puede tardar. Detén el
servidor con `Ctrl+C`.

Para elegir otra dirección o puerto:

```bash
uv run --locked peliculas-web --host 127.0.0.1 --port 8001
```

La aplicación es un servidor de desarrollo local, no una configuración de despliegue
para producción.

## Ejecutar la aplicación de terminal

También está disponible el flujo interactivo:

```bash
uv run --locked peliculas
```

El programa pide un título, el número de recomendaciones y el enfoque (`char` o
`desc`). Para recomendar por descripción, la película debe tener una descripción y
pertenecer al conjunto procesado.

## Reproducir el análisis

Abre `notebooks/plots.ipynb` en VS Code con la extensión Jupyter o en un entorno
compatible con notebooks. Selecciona como kernel el intérprete del proyecto:

```text
.venv/bin/python
```

Luego elige **Run All / Ejecutar todo**. El notebook busca
`data/netflix_titles.csv` en `data/` o `../data/` respecto al directorio de trabajo
y recalcula los resultados; para reproducir las métricas, ejecútalo completo y en orden.
Incluye:

1. Análisis de calidad y distribuciones de los clusters de referencia.
2. Pruebas de variantes para los enfoques `char` y `desc`.
3. Comparación de baseline y mejoras sobre muestras fijas, con métricas calculadas
   dentro de cada representación.
4. Diez películas de prueba y cinco recomendaciones por película en cada enfoque,
   con validaciones para excluir el título consultado y no repetir recomendaciones.

Las salidas guardadas en el notebook son resultados de ejecuciones anteriores. Para
actualizarlas después de modificar datos o código, ejecuta todas las celdas y guarda
el notebook.

## Pruebas y comprobaciones

Ejecuta los tests:

```bash
uv run --locked --group dev python -m unittest discover -s tests
```

Comprueba el estilo de código:

```bash
uv run --locked --group dev ruff check src tests
```

Construye las distribuciones del paquete:

```bash
uv build
```

Comprueba los argumentos de la aplicación web:

```bash
uv run --locked peliculas-web --help
```

## Estructura del proyecto

```text
.
├── data/
│   └── netflix_titles.csv   # catálogo de entrada
├── notebooks/
│   └── plots.ipynb          # análisis, evaluación y recomendaciones de prueba
├── src/peliculas/
│   ├── etl.py               # preparación, TF-IDF, clustering y recomendación
│   ├── web.py               # servidor HTTP local y API
│   ├── static/              # interfaz web
│   └── consts.py            # rutas y constantes
├── tests/                   # pruebas unitarias
├── pyproject.toml           # metadatos, dependencias y comandos
└── uv.lock                  # versiones bloqueadas
```

## Publicar en GitHub

Antes de crear un commit, revisa que los cambios y archivos generados sean los
esperados:

```bash
git status --short
git diff --check
git diff --stat
```

Ejecuta las [pruebas y comprobaciones](#pruebas-y-comprobaciones). Después, añade los
archivos que quieres compartir, crea el commit y sube tu rama:

```bash
git add README.md pyproject.toml uv.lock src tests notebooks
git commit -m "feat: prioriza términos TF-IDF por descripción"
git push -u origin <nombre-de-tu-rama>
```

Abre un Pull Request en GitHub si el repositorio trabaja mediante revisión. Comprueba
que en la página del repositorio se vean el README, el notebook y la estructura
esperada. No subas `.venv`, `dist`, cachés, credenciales ni datos privados.

Antes de hacer público el repositorio, revisa también la sección sobre
[licencia y datos](#licencia-y-uso-de-los-datos). Este repositorio no configura
actualmente un workflow de CI; los comandos anteriores son comprobaciones locales.

## Licencia y uso de los datos

El proyecto no declara todavía una licencia de software. Antes de permitir que otras
personas reutilicen o distribuyan el código, el propietario debe elegir una licencia
apropiada y añadir su archivo (por ejemplo, `LICENSE`).

El CSV está incluido en `data/netflix_titles.csv`. Confirma y documenta la procedencia
del dataset y que sus condiciones permiten redistribuirlo públicamente antes de
publicar el repositorio abierto. La licencia que se elija para el código no concede
automáticamente derechos sobre el dataset.
