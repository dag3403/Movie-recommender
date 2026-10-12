# Netflix Movie Insights

Aplicación web local para explorar el catálogo incluido en `data/netflix_titles.csv`,
comparar los clusters del pipeline existente y consultar recomendaciones.

## Ejecutar

Desde la raíz del repositorio, con el entorno del proyecto:

```bash
uv run peliculas-web
```

Abre <http://127.0.0.1:8000>. La primera carga procesa las características, vectoriza
las descripciones y ejecuta una vez los dos `KMeans` existentes. Las interacciones
posteriores reutilizan esos resultados; las distancias de recomendación se guardan en
una caché en memoria durante la sesión. `Ctrl+C` detiene el servidor. Se puede cambiar
la dirección o el puerto con `uv run peliculas-web --host 127.0.0.1 --port 8001`.

## Secciones

- **Resumen:** estadísticas obtenidas del CSV y de los clusters calculados.
- **Exploración de clusters:** distribución y listado paginado de películas para
  `Characteristics` y `Description`.
- **Recomendador:** selecciona un título de película (las sugerencias distinguen cada
  registro), el enfoque y el número de resultados. Utiliza `recomendador` del ETL.
- **Catálogo:** búsqueda parcial sin distinción de mayúsculas, ordenación y paginación
  de películas y series. Las películas se pueden enviar directamente al recomendador.

Los clusters usan los parámetros que ya emplea `peliculas.main`: 2 para características
y 15 para descripciones. El clustering de características solo contiene registros con
los campos requeridos completos; el enfoque de descripción contiene películas con
descripción. La aplicación no guarda ni sobrescribe los datos originales.