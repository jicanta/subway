# Analisis de la red de subte de CABA

Este repositorio contiene el material de trabajo para el TP de Ciencia de Ciudades sobre la red de subte de la Ciudad de Buenos Aires. El proyecto combina analisis de red, demanda potencial, atraccion urbana y escenarios futuros a partir de datos operativos y espaciales.

## Alcance

El trabajo esta organizado en tres bloques principales:

- `analisis_conectividad_subte_caba.ipynb` y `analisis_conectividad_subte_caba.py`: construccion de la red base, calculo de metricas de centralidad y evaluacion de escenarios futuros con las lineas `F`, `G` e `I`.
- `analisis_mercado_y_atraccion_subte_caba.ipynb`: desarrollo narrativo de los ejercicios 2 y 3 del TP, con tablas y figuras listas para discutir resultados.
- `analisis_mercado_atraccion_subte_caba.py`: backend analitico del notebook de mercado y atraccion. Genera las salidas tabulares y graficas en `outputs/` y `figures/`.

## Estructura del repositorio

- `data/`: insumos base de la red y dataset local de molinetes 2024.
- `external_data/`: caches descargados automaticamente para radios censales y POI de OpenStreetMap.
- `figures/`: graficos exportados por los scripts y notebooks.
- `outputs/`: tablas CSV con resultados intermedios y finales.
- `requirements.txt`: dependencias de Python necesarias para reproducir el proyecto.

## Datos utilizados

### Red base

La red base se construye con archivos de texto locales:

- `data/estaciones.txt`
- `data/conexiones.txt`
- `data/estaciones_posicion.txt`
- `data/estaciones_posicion_diagrama.txt`

### Molinetes 2024

Para el analisis de mercado y atraccion, el codigo prioriza el dataset local ubicado en `data/molinetes-2024/` por sobre la fuente historica `external_data/BaseUnificadaEstaciones.csv`.

Con este dataset 2024:

- `Plaza de Mayo` de la linea `A` queda correctamente representada en molinetes.
- el empalme entre red y molinetes resulta completo para las `89` estaciones de la red base.
- el unico registro presente en molinetes pero ausente en la red base es `Echeverria` de la linea `B`.

### OpenStreetMap para el ejercicio 3

El ejercicio 3 si utiliza datos de `OpenStreetMap`, tal como pide la consigna. La descarga se hace via `Overpass API` y queda implementada en `analisis_mercado_atraccion_subte_caba.py`.

La construccion de `atraccion_total` usa POI descargados desde OSM con estas familias de tags:

- educacion: tags `amenity` como `school`, `college`, `university`, `kindergarten` y `library`
- salud: tags `amenity` como `hospital`, `clinic`, `doctors`, `dentist` y `pharmacy`
- comercio: tags `shop` como `supermarket`, `convenience`, `bakery`, `department_store`, `mall`, `clothes`, `books` y `shoes`

Los resultados descargados se cachean localmente en:

- `external_data/osm_educacion.json`
- `external_data/osm_salud.json`
- `external_data/osm_comercio.json`

En otras palabras, el ejercicio 3 no usa una proxy ad hoc inventada manualmente: usa POI efectivamente descargados desde OpenStreetMap y luego agregados por estacion dentro de un radio de `400 m`.

## Nota metodologica importante sobre diciembre 2024

Los archivos de diciembre se llaman:

- `202412_PAX15min-ABC-INCLUYEOTROMODOSDEPAGO.csv`
- `202412_PAX15min-DEH-INCLUYEOTROMODOSDEPAGO.csv`

Eso indica que diciembre 2024 incorpora otros modos de pago en la definicion operativa de validacion. En este proyecto esos archivos se incluyen porque forman parte del paquete anual recibido y mejoran la cobertura temporal del ejercicio, pero la serie 2024 no debe interpretarse como perfectamente homogenea mes a mes.

En consecuencia:

- `pasajeros_diarios_promedio` debe leerse como promedio por dia observado, no como promedio anual corrido estricto.
- las comparaciones entre estaciones son utiles para el TP, pero conviene evitar lecturas demasiado finas de diferencias pequenas entre meses o estaciones con cobertura parcial.

## Cobertura observada en el dataset 2024

La version actual del analisis deja documentados estos diagnosticos:

- periodo observado: `2024-01-01` a `2024-12-31`
- fechas efectivamente presentes en la red: `364`
- fechas faltantes dentro del periodo: `2`
- estaciones con cobertura completa: `66`
- estaciones con cobertura parcial: `23`

La cobertura parcial se concentra sobre todo en algunas estaciones de las lineas `D` y `B`, por lo que los niveles de validacion en esas estaciones deben compararse con esa salvedad en mente.

## Reproduccion

### 1. Crear entorno virtual

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Ejecutar el analisis de conectividad

```bash
.venv/bin/python analisis_conectividad_subte_caba.py
```

Esto regenera las salidas de conectividad, centralidad y escenarios futuros en `figures/` y `outputs/`.

### 3. Ejecutar el analisis de mercado y atraccion

```bash
.venv/bin/python -c "from analisis_mercado_atraccion_subte_caba import generate_outputs; generate_outputs()"
```

Esto regenera:

- `outputs/estaciones_mercado_atraccion.csv`
- `outputs/diagnosticos_mercado_atraccion.csv`
- `outputs/correlaciones_mercado_atraccion.csv`
- `outputs/matriz_od_demanda.csv`
- `outputs/matriz_od_atraccion.csv`
- las figuras asociadas en `figures/`

### 4. Reejecutar notebooks

Si se quiere revisar o exportar la narrativa completa del TP:

```bash
.venv/bin/jupyter notebook
```

Los notebooks principales son:

- `analisis_conectividad_subte_caba.ipynb`
- `analisis_mercado_y_atraccion_subte_caba.ipynb`

## Resultados actuales relevantes

Con la configuracion actual del repositorio:

- la correlacion entre demanda potencial y molinetes es baja, lo que refuerza que la poblacion residencial cercana no alcanza para explicar por si sola los flujos observados.
- la correlacion por atraccion urbana mejora en rangos, pero sigue siendo insuficiente para capturar completamente transbordos, cabeceras y centralidades funcionales.
- estaciones como `Plaza de Mayo`, `Retiro`, `Leandro N. Alem` y `Constitucion` quedan mejor representadas con el dataset 2024 y muestran con claridad la diferencia entre mercado residencial, atraccion urbana y uso efectivo de la red.

## Dependencias

Las dependencias declaradas en `requirements.txt` son:

- `pandas`
- `networkx`
- `matplotlib`
- `shapely`
- `nbformat`
- `nbclient`
- `ipykernel`

## Estado del repositorio

El proyecto esta preparado para seguir iterando desde los notebooks o desde los scripts. Antes de publicar resultados finales conviene mantener sincronizados:

- el codigo en `*.py`
- los notebooks ejecutados
- las figuras exportadas en `figures/`
- los CSV de salida en `outputs/`

Con eso, el repositorio queda en condiciones de compartirse, revisarse y reproducirse de punta a punta.
