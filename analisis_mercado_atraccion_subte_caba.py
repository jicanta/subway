from __future__ import annotations

import json
import math
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import transform

from analisis_conectividad_subte_caba import (
    LINE_COLORS,
    apply_plot_style,
    build_graph,
    build_positions,
    build_summary,
    compute_metrics,
    draw_metric_map,
    load_data,
)


BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_DATA_DIR = BASE_DIR / "external_data"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"

TURNSTILE_URL = "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/sbase/subte-viajes-molinetes/BaseUnificadaEstaciones.csv"
CENSUS_URL = "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/direccion-general-de-estadisticas-y-censos/informacion-censal-por-radio/caba_radios_censales.geojson"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DEFAULT_DEMAND_RADIUS_M = 800
DEFAULT_POI_RADIUS_M = 400
DEFAULT_DECAY_MINUTES = 15.0
PRE_LOCKDOWN_END = pd.Timestamp("2020-03-19")
EARTH_RADIUS_M = 6_371_000
LATITUDE_REFERENCE_RAD = math.radians(-34.6037)

TURNSTILE_NAME_ALIASES = {
    ("A", "flores"): "San Jose de Flores",
    ("B", "los incas"): "De los Incas",
    ("B", "rosas"): "Juan Manuel de Rosas",
    ("C", "mariano moreno"): "Moreno",
    ("D", "callao"): "Callao",
    ("D", "pueyrredon"): "Pueyrredon",
    ("E", "general belgrano"): "Belgrano",
    ("E", "independencia"): "Independencia",
    ("E", "pza de los virreyes"): "Plaza de los Virreyes",
    ("E", "retiro e"): "Retiro",
    ("E", "urquiza"): "General Urquiza",
    ("H", "facultad derecho"): "Facultad de Derecho",
    ("H", "humberto i"): "Humberto 1o",
    ("H", "lasheras"): "Las Heras",
    ("H", "patricios"): "Parque Patricios",
}

POI_CATEGORY_FILTERS = {
    "educacion": [
        ("amenity", "school"),
        ("amenity", "college"),
        ("amenity", "university"),
        ("amenity", "kindergarten"),
        ("amenity", "library"),
    ],
    "salud": [
        ("amenity", "hospital"),
        ("amenity", "clinic"),
        ("amenity", "doctors"),
        ("amenity", "dentist"),
        ("amenity", "pharmacy"),
    ],
    "comercio": [
        ("shop", "supermarket"),
        ("shop", "convenience"),
        ("shop", "bakery"),
        ("shop", "department_store"),
        ("shop", "mall"),
        ("shop", "clothes"),
        ("shop", "books"),
        ("shop", "shoes"),
    ],
}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_station_key(linea: str, nombre: str) -> tuple[str, str]:
    line_key = str(linea).strip()
    normalized_name = normalize_text(nombre)
    alias = TURNSTILE_NAME_ALIASES.get((line_key, normalized_name), nombre)
    return line_key, normalize_text(alias)


def download_if_missing(url: str, destination: Path) -> Path:
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        return destination

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())
    return destination


def post_json(url: str, payload: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        ["curl", "-sS", "-X", "POST", url, "--data-urlencode", f"data={payload['data']}"] ,
        capture_output=True,
        check=True,
        text=True,
        timeout=180,
    )
    return json.loads(result.stdout)


def lonlat_to_xy(lon: float, lat: float) -> tuple[float, float]:
    x = math.radians(lon) * EARTH_RADIUS_M * math.cos(LATITUDE_REFERENCE_RAD)
    y = math.radians(lat) * EARTH_RADIUS_M
    return x, y


def geometry_to_xy(geom):
    return transform(lambda lon, lat, z=None: lonlat_to_xy(lon, lat), geom)


def load_station_context() -> dict[str, object]:
    estaciones, conexiones, posiciones, diagrama = load_data()
    grafo = build_graph(estaciones, conexiones)
    _, pos_diag = build_positions(posiciones, diagrama)
    summary = build_summary(grafo, compute_metrics(grafo))

    station_frame = estaciones.merge(posiciones[["id", "lat", "long"]], on="id", how="left")
    station_frame["linea"] = station_frame["linea"].astype(str).str.strip()
    station_frame["lat"] = station_frame["lat"].str.replace(",", ".", regex=False).astype(float)
    station_frame["long"] = station_frame["long"].str.replace(",", ".", regex=False).astype(float)
    station_frame[["x_m", "y_m"]] = station_frame.apply(
        lambda row: pd.Series(lonlat_to_xy(row["long"], row["lat"])), axis=1
    )
    station_frame["point_m"] = station_frame.apply(lambda row: Point(row["x_m"], row["y_m"]), axis=1)
    station_frame["nombre_norm"] = station_frame["nombre"].map(normalize_text)
    station_frame["clave_estacion"] = station_frame.apply(lambda row: f"{row['nombre']} ({row['linea']})", axis=1)

    return {
        "graph": grafo,
        "positions": pos_diag,
        "summary": summary,
        "stations": station_frame,
    }


def load_turnstiles(pre_lockdown_only: bool = True) -> pd.DataFrame:
    path = download_if_missing(TURNSTILE_URL, EXTERNAL_DATA_DIR / "BaseUnificadaEstaciones.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [column.strip().lower() for column in frame.columns]
    frame["fecha"] = pd.to_datetime(frame["fecha"], dayfirst=True, errors="coerce")
    frame["linea"] = frame["linea"].astype(str).str.strip()
    frame["estacion"] = frame["estacion"].astype(str).str.strip()
    frame["cantidad"] = pd.to_numeric(frame["cantidad"], errors="coerce").fillna(0)

    if pre_lockdown_only:
        frame = frame[frame["fecha"] <= PRE_LOCKDOWN_END].copy()

    daily = frame.groupby(["fecha", "linea", "estacion"], as_index=False)["cantidad"].sum()
    station_daily = (
        daily.groupby(["linea", "estacion"], as_index=False)
        .agg(
            pasajeros_diarios_promedio=("cantidad", "mean"),
            pasajeros_totales_periodo=("cantidad", "sum"),
            dias_observados=("fecha", "nunique"),
        )
        .sort_values(["linea", "estacion"])
    )
    station_daily[["linea", "nombre_norm"]] = station_daily.apply(
        lambda row: pd.Series(normalize_station_key(row["linea"], row["estacion"])), axis=1
    )
    return station_daily


def build_turnstile_diagnostics(stations: pd.DataFrame, turnstiles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    network_keys = stations[["linea", "nombre_norm", "nombre"]].drop_duplicates().copy()
    turnstile_keys = turnstiles[["linea", "nombre_norm", "estacion"]].drop_duplicates().copy()

    missing_turnstiles = network_keys.merge(
        turnstile_keys[["linea", "nombre_norm"]], on=["linea", "nombre_norm"], how="left", indicator=True
    )
    missing_turnstiles = (
        missing_turnstiles[missing_turnstiles["_merge"] == "left_only"][ ["linea", "nombre"] ]
        .rename(columns={"nombre": "estacion_red"})
        .sort_values(["linea", "estacion_red"])
        .reset_index(drop=True)
    )

    missing_network = turnstile_keys.merge(
        network_keys[["linea", "nombre_norm"]], on=["linea", "nombre_norm"], how="left", indicator=True
    )
    missing_network = (
        missing_network[missing_network["_merge"] == "left_only"][ ["linea", "estacion"] ]
        .rename(columns={"estacion": "estacion_molinetes"})
        .sort_values(["linea", "estacion_molinetes"])
        .reset_index(drop=True)
    )

    return {
        "stations_without_turnstiles": missing_turnstiles,
        "turnstiles_without_network_station": missing_network,
    }


def load_census_radios() -> pd.DataFrame:
    path = download_if_missing(CENSUS_URL, EXTERNAL_DATA_DIR / "caba_radios_censales.geojson")
    payload = json.loads(path.read_text(encoding="utf-8"))

    records = []
    for feature in payload["features"]:
        properties = feature["properties"]
        population = int(properties.get("TOTAL_POB") or 0)
        geometry = geometry_to_xy(shape(feature["geometry"]))
        records.append(
            {
                "radio_id": str(properties.get("ID")),
                "population": population,
                "geometry_m": geometry,
            }
        )

    return pd.DataFrame(records)


def estimate_station_demand(stations: pd.DataFrame, radios: pd.DataFrame, radius_m: int) -> pd.DataFrame:
    rows = []
    for station in stations.itertuples():
        selected = radios[radios["geometry_m"].map(lambda geometry: geometry.distance(station.point_m) <= radius_m)]
        rows.append(
            {
                "id": station.id,
                "demanda_potencial": int(selected["population"].sum()),
                "radios_cubiertos": int(len(selected)),
            }
        )
    return pd.DataFrame(rows)


def overpass_bbox(stations: pd.DataFrame, margin_deg: float = 0.03) -> tuple[float, float, float, float]:
    south = stations["lat"].min() - margin_deg
    north = stations["lat"].max() + margin_deg
    west = stations["long"].min() - margin_deg
    east = stations["long"].max() + margin_deg
    return south, west, north, east


def build_overpass_query(filters: list[tuple[str, str]], bbox: tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    clauses = []
    for key, value in filters:
        clauses.append(f'node["{key}"="{value}"]({south},{west},{north},{east});')
        clauses.append(f'way["{key}"="{value}"]({south},{west},{north},{east});')
        clauses.append(f'relation["{key}"="{value}"]({south},{west},{north},{east});')
    joined = "\n".join(clauses)
    return f"[out:json][timeout:120];({joined});out center;"


def load_poi_category(category: str, stations: pd.DataFrame) -> pd.DataFrame:
    cache_path = EXTERNAL_DATA_DIR / f"osm_{category}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        query = build_overpass_query(POI_CATEGORY_FILTERS[category], overpass_bbox(stations))
        payload = post_json(OVERPASS_URL, {"data": query})
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

    rows = []
    for element in payload.get("elements", []):
        lat = element.get("lat")
        lon = element.get("lon")
        if lat is None or lon is None:
            center = element.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")
        if lat is None or lon is None:
            continue

        x_m, y_m = lonlat_to_xy(float(lon), float(lat))
        rows.append(
            {
                "osm_id": f"{element['type']}/{element['id']}",
                "categoria": category,
                "lat": float(lat),
                "long": float(lon),
                "x_m": x_m,
                "y_m": y_m,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["osm_id", "categoria", "lat", "long", "x_m", "y_m"])
    return pd.DataFrame(rows).drop_duplicates(subset=["osm_id"]).reset_index(drop=True)


def count_poi_near_stations(stations: pd.DataFrame, poi: pd.DataFrame, radius_m: int, output_column: str) -> pd.DataFrame:
    rows = []
    for station in stations.itertuples():
        if poi.empty:
            count = 0
        else:
            distance_sq = (poi["x_m"] - station.x_m) ** 2 + (poi["y_m"] - station.y_m) ** 2
            count = int((distance_sq <= radius_m**2).sum())
        rows.append({"id": station.id, output_column: count})
    return pd.DataFrame(rows)


def build_path_model(grafo: nx.Graph) -> dict[str, object]:
    nodes = list(grafo.nodes())
    node_index = {node: index for index, node in enumerate(nodes)}
    pair_index: list[tuple[int, int]] = []
    distances: list[float] = []
    participation_rows: list[np.ndarray] = []

    for source in nodes:
        source_lengths = nx.single_source_dijkstra_path_length(grafo, source, weight="tiempo")
        for target in nodes:
            if source == target:
                continue

            pair_index.append((source, target))
            distances.append(float(source_lengths[target]))

            contribution = np.zeros(len(nodes), dtype=float)
            paths = list(nx.all_shortest_paths(grafo, source, target, weight="tiempo"))
            share = 1 / len(paths)
            for path in paths:
                for node in path[1:-1]:
                    contribution[node_index[node]] += share
            participation_rows.append(contribution)

    participation = pd.DataFrame(
        np.vstack(participation_rows),
        index=pd.MultiIndex.from_tuples(pair_index, names=["origen", "destino"]),
        columns=nodes,
    )
    distance_series = pd.Series(distances, index=participation.index, name="tiempo")
    return {
        "nodes": nodes,
        "participation": participation,
        "distances": distance_series,
    }


def uniform_od_matrix(nodes: list[int]) -> pd.DataFrame:
    node_count = len(nodes)
    values = np.full((node_count, node_count), 1 / (node_count * (node_count - 1)), dtype=float)
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=nodes, columns=nodes)


def gravity_od_matrix(
    station_metrics: pd.DataFrame,
    distance_series: pd.Series,
    origin_column: str,
    destination_column: str,
    decay_minutes: float,
) -> pd.DataFrame:
    node_order = station_metrics["id"].tolist()
    indexed = station_metrics.set_index("id")
    origin = indexed.loc[node_order, origin_column].astype(float).to_numpy()[:, None]
    destination = indexed.loc[node_order, destination_column].astype(float).to_numpy()[None, :]
    distance_matrix = distance_series.unstack().loc[node_order, node_order].to_numpy(dtype=float)

    weights = origin * destination * np.exp(-distance_matrix / decay_minutes)
    np.fill_diagonal(weights, 0.0)
    total = weights.sum()
    if total == 0:
        raise ValueError(f"No se pudo normalizar la matriz OD para {origin_column} y {destination_column}.")
    weights /= total
    return pd.DataFrame(weights, index=node_order, columns=node_order)


def evaluate_od_model(path_model: dict[str, object], od_matrix: pd.DataFrame) -> dict[str, object]:
    pair_weights = od_matrix.stack().reindex(path_model["distances"].index).fillna(0.0)
    mean_time = float((pair_weights * path_model["distances"]).sum())
    betweenness_values = path_model["participation"].to_numpy().T @ pair_weights.to_numpy()
    betweenness = dict(zip(path_model["participation"].columns.tolist(), betweenness_values, strict=True))
    return {
        "tiempo_medio": mean_time,
        "betweenness": betweenness,
    }


def draw_scatter_comparison(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    output: Path,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
) -> None:
    apply_plot_style()
    plotted = frame.dropna(subset=[x_column, y_column]).copy()
    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    colors = [LINE_COLORS[linea] for linea in plotted["linea"]]
    ax.scatter(plotted[x_column], plotted[y_column], c=colors, alpha=0.82, s=55, edgecolors="white", linewidths=0.7)
    ax.set_title(f"{title}\n{subtitle}", loc="left")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(color="#DEE2E6", linewidth=0.8)
    ax.set_axisbelow(True)

    top_labels = plotted.nlargest(6, y_column)
    for row in top_labels.itertuples():
        ax.annotate(
            f"{row.estacion} ({row.linea})",
            (getattr(row, x_column), getattr(row, y_column)),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).dropna()
    if valid.empty:
        return float("nan")
    ranked = valid.rank(method="average")
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1], method="pearson"))


def build_station_metrics_table(
    summary: pd.DataFrame,
    stations: pd.DataFrame,
    demand: pd.DataFrame,
    turnstiles: pd.DataFrame,
    poi_counts: dict[str, pd.DataFrame],
    model_results: dict[str, dict[str, object]],
) -> pd.DataFrame:
    frame = summary.merge(stations[["id", "lat", "long", "clave_estacion", "nombre_norm"]], on="id", how="left")
    frame = frame.merge(demand, on="id", how="left")
    frame = frame.merge(turnstiles[["linea", "nombre_norm", "pasajeros_diarios_promedio", "pasajeros_totales_periodo", "dias_observados"]], on=["linea", "nombre_norm"], how="left")

    for column, poi_frame in poi_counts.items():
        frame = frame.merge(poi_frame, on="id", how="left")

    frame[["educacion", "salud", "comercio"]] = frame[["educacion", "salud", "comercio"]].fillna(0).astype(int)
    frame["atraccion_total"] = frame[["educacion", "salud", "comercio"]].sum(axis=1)
    frame["betweenness_uniforme"] = frame["id"].map(model_results["uniforme"]["betweenness"])
    frame["betweenness_demanda"] = frame["id"].map(model_results["demanda"]["betweenness"])
    frame["betweenness_atraccion"] = frame["id"].map(model_results["atraccion"]["betweenness"])
    frame["cambio_demanda_vs_uniforme"] = frame["betweenness_demanda"] - frame["betweenness_uniforme"]
    frame["cambio_atraccion_vs_demanda"] = frame["betweenness_atraccion"] - frame["betweenness_demanda"]
    return frame


def build_model_summary(model_results: dict[str, dict[str, object]]) -> pd.DataFrame:
    baseline = model_results["uniforme"]["tiempo_medio"]
    rows = []
    labels = {
        "uniforme": "Sin peso",
        "demanda": "Demanda potencial",
        "atraccion": "Demanda y atraccion",
    }
    for key in ["uniforme", "demanda", "atraccion"]:
        rows.append(
            {
                "escenario": labels[key],
                "tiempo_medio": model_results[key]["tiempo_medio"],
                "delta_vs_sin_peso_pct": (model_results[key]["tiempo_medio"] - baseline) / baseline * 100,
            }
        )
    return pd.DataFrame(rows)


def generate_outputs(
    demand_radius_m: int = DEFAULT_DEMAND_RADIUS_M,
    poi_radius_m: int = DEFAULT_POI_RADIUS_M,
    decay_minutes: float = DEFAULT_DECAY_MINUTES,
) -> dict[str, object]:
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    EXTERNAL_DATA_DIR.mkdir(exist_ok=True)

    station_context = load_station_context()
    grafo = station_context["graph"]
    pos_diag = station_context["positions"]
    base_summary = station_context["summary"]
    stations = station_context["stations"]

    turnstiles = load_turnstiles(pre_lockdown_only=True)
    diagnostics = build_turnstile_diagnostics(stations, turnstiles)
    radios = load_census_radios()
    demand = estimate_station_demand(stations, radios, radius_m=demand_radius_m)

    poi_frames = {category: load_poi_category(category, stations) for category in POI_CATEGORY_FILTERS}
    poi_counts = {
        category: count_poi_near_stations(stations, poi_frame, radius_m=poi_radius_m, output_column=category)
        for category, poi_frame in poi_frames.items()
    }

    model_input = stations[["id", "nombre", "nombre_norm", "linea"]].merge(demand, on="id", how="left")
    for category, frame in poi_counts.items():
        model_input = model_input.merge(frame, on="id", how="left")
    model_input[["educacion", "salud", "comercio"]] = model_input[["educacion", "salud", "comercio"]].fillna(0).astype(int)
    model_input["atraccion_total"] = model_input[["educacion", "salud", "comercio"]].sum(axis=1)

    path_model = build_path_model(grafo)
    od_uniform = uniform_od_matrix(path_model["nodes"])
    od_demand = gravity_od_matrix(model_input, path_model["distances"], "demanda_potencial", "demanda_potencial", decay_minutes)
    od_attraction = gravity_od_matrix(model_input, path_model["distances"], "demanda_potencial", "atraccion_total", decay_minutes)

    model_results = {
        "uniforme": evaluate_od_model(path_model, od_uniform),
        "demanda": evaluate_od_model(path_model, od_demand),
        "atraccion": evaluate_od_model(path_model, od_attraction),
    }

    station_metrics = build_station_metrics_table(base_summary, stations, demand, turnstiles, poi_counts, model_results)
    model_summary = build_model_summary(model_results)

    demand_map_summary = station_metrics[["id", "linea", "estacion", "demanda_potencial"]].copy()
    demand_values = dict(zip(station_metrics["id"], station_metrics["demanda_potencial"], strict=True))
    draw_metric_map(
        grafo,
        pos_diag,
        demand_values,
        demand_map_summary,
        metric="demanda_potencial",
        title="Demanda potencial por estacion",
        subtitle=f"Poblacion total en radios censales a menos de {demand_radius_m} m de cada estacion.",
        output=FIGURES_DIR / "demanda_potencial_por_estacion.png",
    )

    for category, title in {
        "educacion": "Oferta educativa cercana",
        "salud": "Oferta de salud cercana",
        "comercio": "Oferta comercial cercana",
    }.items():
        category_summary = station_metrics[["id", "linea", "estacion", category]].copy()
        category_values = dict(zip(station_metrics["id"], station_metrics[category], strict=True))
        draw_metric_map(
            grafo,
            pos_diag,
            category_values,
            category_summary,
            metric=category,
            title=title,
            subtitle=f"Cantidad de puntos de interes a menos de {poi_radius_m} m de cada estacion.",
            output=FIGURES_DIR / f"poi_{category}_por_estacion.png",
        )

    attraction_summary = station_metrics[["id", "linea", "estacion", "atraccion_total"]].copy()
    attraction_values = dict(zip(station_metrics["id"], station_metrics["atraccion_total"], strict=True))
    draw_metric_map(
        grafo,
        pos_diag,
        attraction_values,
        attraction_summary,
        metric="atraccion_total",
        title="Atraccion total por estacion",
        subtitle=f"Suma de educacion, salud y comercio a menos de {poi_radius_m} m.",
        output=FIGURES_DIR / "atraccion_total_por_estacion.png",
    )

    draw_scatter_comparison(
        station_metrics,
        x_column="demanda_potencial",
        y_column="pasajeros_diarios_promedio",
        output=FIGURES_DIR / "demanda_vs_molinetes.png",
        title="Demanda potencial vs usuarios observados",
        subtitle="Promedio diario por estacion usando el tramo previo al ASPO de 2020.",
        x_label="Demanda potencial",
        y_label="Pasajeros diarios promedio",
    )
    draw_scatter_comparison(
        station_metrics,
        x_column="atraccion_total",
        y_column="pasajeros_diarios_promedio",
        output=FIGURES_DIR / "atraccion_vs_molinetes.png",
        title="Atraccion de POI vs usuarios observados",
        subtitle="La atraccion combina educacion, salud y comercio cercanos.",
        x_label="Atraccion total",
        y_label="Pasajeros diarios promedio",
    )

    matched_for_ridership = station_metrics.dropna(subset=["pasajeros_diarios_promedio"]).copy()
    diagnostics_summary = pd.DataFrame(
        [
            {"indicador": "estaciones_red", "valor": int(len(stations))},
            {"indicador": "estaciones_con_molinetes_matcheadas", "valor": int(len(matched_for_ridership))},
            {"indicador": "estaciones_red_sin_molinetes", "valor": int(len(diagnostics["stations_without_turnstiles"]))},
            {"indicador": "estaciones_molinetes_sin_red", "valor": int(len(diagnostics["turnstiles_without_network_station"]))},
            {"indicador": "radios_censales", "valor": int(len(radios))},
            {"indicador": "poi_educacion_descargados", "valor": int(len(poi_frames["educacion"]))},
            {"indicador": "poi_salud_descargados", "valor": int(len(poi_frames["salud"]))},
            {"indicador": "poi_comercio_descargados", "valor": int(len(poi_frames["comercio"]))},
        ]
    )

    correlation_summary = pd.DataFrame(
        [
            {
                "comparacion": "demanda_vs_molinetes",
                "pearson": matched_for_ridership["demanda_potencial"].corr(matched_for_ridership["pasajeros_diarios_promedio"], method="pearson"),
                "spearman": spearman_correlation(matched_for_ridership["demanda_potencial"], matched_for_ridership["pasajeros_diarios_promedio"]),
            },
            {
                "comparacion": "atraccion_vs_molinetes",
                "pearson": matched_for_ridership["atraccion_total"].corr(matched_for_ridership["pasajeros_diarios_promedio"], method="pearson"),
                "spearman": spearman_correlation(matched_for_ridership["atraccion_total"], matched_for_ridership["pasajeros_diarios_promedio"]),
            },
        ]
    )

    station_metrics.to_csv(OUTPUT_DIR / "estaciones_mercado_atraccion.csv", index=False)
    model_summary.to_csv(OUTPUT_DIR / "resumen_modelos_od.csv", index=False)
    diagnostics_summary.to_csv(OUTPUT_DIR / "diagnosticos_mercado_atraccion.csv", index=False)
    correlation_summary.to_csv(OUTPUT_DIR / "correlaciones_mercado_atraccion.csv", index=False)
    od_demand.to_csv(OUTPUT_DIR / "matriz_od_demanda.csv", index=True)
    od_attraction.to_csv(OUTPUT_DIR / "matriz_od_atraccion.csv", index=True)
    diagnostics["stations_without_turnstiles"].to_csv(OUTPUT_DIR / "estaciones_red_sin_molinetes.csv", index=False)
    diagnostics["turnstiles_without_network_station"].to_csv(OUTPUT_DIR / "estaciones_molinetes_sin_red.csv", index=False)

    return {
        "config": pd.DataFrame(
            [
                {"parametro": "radio_demanda_m", "valor": demand_radius_m},
                {"parametro": "radio_poi_m", "valor": poi_radius_m},
                {"parametro": "d0_minutos", "valor": decay_minutes},
                {"parametro": "molinetes_periodo", "valor": "2020-01-01 a 2020-03-19"},
            ]
        ),
        "diagnostics": diagnostics_summary,
        "mapping_diagnostics": diagnostics,
        "correlations": correlation_summary,
        "model_summary": model_summary,
        "station_metrics": station_metrics,
        "top_demand": station_metrics.sort_values("demanda_potencial", ascending=False)[
            ["estacion", "linea", "demanda_potencial", "pasajeros_diarios_promedio", "radios_cubiertos"]
        ].head(12),
        "top_attraction": station_metrics.sort_values("atraccion_total", ascending=False)[
            ["estacion", "linea", "atraccion_total", "educacion", "salud", "comercio", "pasajeros_diarios_promedio"]
        ].head(12),
        "top_betweenness_demand": station_metrics.sort_values("betweenness_demanda", ascending=False)[
            ["estacion", "linea", "betweenness_uniforme", "betweenness_demanda", "cambio_demanda_vs_uniforme"]
        ].head(12),
        "top_betweenness_attraction": station_metrics.sort_values("betweenness_atraccion", ascending=False)[
            ["estacion", "linea", "betweenness_demanda", "betweenness_atraccion", "cambio_atraccion_vs_demanda"]
        ].head(12),
    }
