from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from analisis_conectividad_subte_caba import draw_metric_map
from analisis_mercado_atraccion_subte_caba import (
    FIGURES_DIR,
    OUTPUT_DIR,
    build_path_model,
    evaluate_od_model,
    generate_outputs as generate_market_outputs,
    load_station_context,
    normalize_text,
    uniform_od_matrix,
)


BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_DATA_DIR = BASE_DIR / "external_data"
BID_DATA_DIR = EXTERNAL_DATA_DIR / "bid_sube_od"

BID_ETAPAS_URL = "https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/resultados/etapas.csv"
BID_PARADAS_URL = "https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/paradas.csv"
BID_LINEAS_RAMALES_URL = "https://media.githubusercontent.com/media/EL-BID/Matriz-Origen-Destino-Transporte-Publico/main/data/lineas_ramales.csv"

BID_SUBTE_RAMAL_TO_LINE = {
    32: "A",
    33: "B",
    269: "C",
    54: "D",
    148: "E",
    220: "H",
}

BID_STATION_NAME_ALIASES = {
    ("A", "plaza de miserere"): "Plaza Miserere",
    ("A", "rio de janeiro"): "Rio de Janeiro",
    ("A", "saenz pena"): "Saenz Pena",
    ("A", "saenz peaa"): "Saenz Pena",
    ("B", "c pellegrini"): "Carlos Pellegrini",
    ("B", "de los incas pque chas"): "De los Incas",
    ("B", "malabia osvaldo pugliese"): "Malabia",
    ("B", "pasteur amia"): "Pasteur",
    ("B", "tronador villa ortuzar"): "Tronador",
    ("C", "av de mayo"): "Avenida de Mayo",
    ("C", "constitucion"): "Constitucion",
    ("C", "san martin"): "General San Martin",
    ("D", "9 de julio"): "9 de Julio",
    ("D", "aguero"): "Aguero",
    ("D", "congreso de tucuman"): "Congreso de Tucuman",
    ("D", "jose hernandez"): "Jose Hernandez",
    ("D", "r scalabrini ortiz"): "Scalabrini Ortiz",
    ("E", "av la plata"): "Avenida La Plata",
    ("E", "bolivar"): "Bolivar",
    ("E", "entre rios rodolfo walsh"): "Entre Rios",
    ("E", "jose maria moreno"): "Jose Maria Moreno",
    ("E", "plaza de los virreyes eva peron"): "Plaza de los Virreyes",
    ("E", "urquiza"): "General Urquiza",
    ("H", "cordoba"): "Cordoba",
    ("H", "facultad de derecho julieta lanteri"): "Facultad de Derecho",
    ("H", "humberto 1"): "Humberto 1o",
    ("H", "inclan"): "Inclan",
    ("H", "once 30 de diciembre"): "Once",
}


def resolve_bid_source(filename: str, remote_url: str) -> str | Path:
    local_path = BID_DATA_DIR / filename
    if local_path.exists():
        return local_path
    return remote_url


def canonical_bid_station_name(linea: str, nombre: str) -> str:
    normalized_name = normalize_text(nombre)
    return BID_STATION_NAME_ALIASES.get((linea, normalized_name), str(nombre).strip())


def build_bid_station_lookup(stations: pd.DataFrame) -> tuple[dict[int, int], dict[str, pd.DataFrame], pd.DataFrame]:
    network_keys = stations[["id", "linea", "nombre", "nombre_norm"]].copy()
    network_lookup = dict(zip(zip(network_keys["linea"], network_keys["nombre_norm"], strict=True), network_keys["id"], strict=True))

    bid_stops = pd.read_csv(
        resolve_bid_source("paradas.csv", BID_PARADAS_URL),
        usecols=["id", "id_ramal", "nombre_estacion"],
    )
    bid_stops = bid_stops[bid_stops["id_ramal"].isin(BID_SUBTE_RAMAL_TO_LINE)].copy()
    bid_stops["linea"] = bid_stops["id_ramal"].astype(int).map(BID_SUBTE_RAMAL_TO_LINE)
    bid_stops["estacion_canonica"] = bid_stops.apply(
        lambda row: canonical_bid_station_name(row["linea"], row["nombre_estacion"]), axis=1
    )
    bid_stops["nombre_norm"] = bid_stops["estacion_canonica"].map(normalize_text)
    bid_stops["station_id"] = bid_stops.apply(
        lambda row: network_lookup.get((row["linea"], row["nombre_norm"])), axis=1
    )

    unmatched_bid_stops = (
        bid_stops[bid_stops["station_id"].isna()][["linea", "nombre_estacion", "estacion_canonica"]]
        .drop_duplicates()
        .sort_values(["linea", "nombre_estacion"])
        .reset_index(drop=True)
    )

    unused_network_stations = (
        network_keys.merge(
            bid_stops[["linea", "nombre_norm"]].drop_duplicates(),
            on=["linea", "nombre_norm"],
            how="left",
            indicator=True,
        )
    )
    unused_network_stations = (
        unused_network_stations[unused_network_stations["_merge"] == "left_only"][["linea", "nombre"]]
        .rename(columns={"nombre": "estacion_red"})
        .sort_values(["linea", "estacion_red"])
        .reset_index(drop=True)
    )

    lookup = {
        int(stop_id): int(station_id)
        for stop_id, station_id in bid_stops[["id", "station_id"]].itertuples(index=False, name=None)
        if pd.notna(station_id)
    }

    source_config = pd.DataFrame(
        [
            {"parametro": "bid_etapas_fuente", "valor": str(resolve_bid_source("etapas.csv", BID_ETAPAS_URL))},
            {"parametro": "bid_paradas_fuente", "valor": str(resolve_bid_source("paradas.csv", BID_PARADAS_URL))},
            {
                "parametro": "bid_lineas_ramales_fuente",
                "valor": str(resolve_bid_source("lineas_ramales.csv", BID_LINEAS_RAMALES_URL)),
            },
        ]
    )

    return lookup, {
        "bid_stops_without_network_station": unmatched_bid_stops,
        "network_stations_without_bid_stop": unused_network_stations,
    }, source_config


def build_sube_stage_od(stations: pd.DataFrame, stop_lookup: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    line_by_station_id = dict(zip(stations["id"], stations["linea"], strict=True))
    name_by_station_id = dict(zip(stations["id"], stations["nombre"], strict=True))

    pair_counter: Counter[tuple[int, int]] = Counter()
    total_subte_stages = 0
    subte_stages_with_complete_od = 0
    subte_stages_mapped_to_network = 0
    self_loop_stages = 0
    cross_line_stages = 0

    etapas_source = resolve_bid_source("etapas.csv", BID_ETAPAS_URL)
    chunk_iterator = pd.read_csv(
        etapas_source,
        usecols=["modo", "parada_id_o", "parada_id_d"],
        chunksize=300_000,
    )

    for chunk in chunk_iterator:
        subte = chunk[chunk["modo"] == "SUB"].copy()
        total_subte_stages += len(subte)

        subte = subte[subte["parada_id_o"].notna() & subte["parada_id_d"].notna()].copy()
        subte_stages_with_complete_od += len(subte)

        subte["origen_id"] = subte["parada_id_o"].map(stop_lookup)
        subte["destino_id"] = subte["parada_id_d"].map(stop_lookup)
        subte = subte[subte["origen_id"].notna() & subte["destino_id"].notna()].copy()
        subte_stages_mapped_to_network += len(subte)

        subte["origen_id"] = subte["origen_id"].astype(int)
        subte["destino_id"] = subte["destino_id"].astype(int)

        self_loop_mask = subte["origen_id"] == subte["destino_id"]
        self_loop_stages += int(self_loop_mask.sum())
        subte = subte[~self_loop_mask].copy()

        cross_line_stages += int(subte["origen_id"].map(line_by_station_id).ne(subte["destino_id"].map(line_by_station_id)).sum())
        pair_counter.update(subte[["origen_id", "destino_id"]].itertuples(index=False, name=None))

    node_order = stations["id"].tolist()
    od_sube = pd.DataFrame(0.0, index=node_order, columns=node_order)
    for (origin_id, destination_id), count in pair_counter.items():
        od_sube.loc[origin_id, destination_id] = float(count)

    total_weight = float(od_sube.to_numpy().sum())
    if total_weight == 0:
        raise ValueError("No se pudo construir la matriz OD realista con etapas SUBE del subte.")
    od_sube /= total_weight

    diagnostics = pd.DataFrame(
        [
            {"indicador": "subte_stages_total", "valor": int(total_subte_stages)},
            {"indicador": "subte_stages_complete_od", "valor": int(subte_stages_with_complete_od)},
            {"indicador": "subte_stages_mapped_to_network", "valor": int(subte_stages_mapped_to_network)},
            {"indicador": "subte_stages_same_station_dropped", "valor": int(self_loop_stages)},
            {"indicador": "subte_stages_used_in_pdd", "valor": int(sum(pair_counter.values()))},
            {"indicador": "subte_unique_station_pairs", "valor": int(len(pair_counter))},
            {
                "indicador": "subte_cross_line_share",
                "valor": float(cross_line_stages / sum(pair_counter.values())) if pair_counter else 0.0,
            },
        ]
    )

    top_pairs = pd.DataFrame(
        [
            {
                "origen": name_by_station_id[origin_id],
                "linea_origen": line_by_station_id[origin_id],
                "destino": name_by_station_id[destination_id],
                "linea_destino": line_by_station_id[destination_id],
                "viajes_estimados": count,
                "probabilidad": count / sum(pair_counter.values()),
            }
            for (origin_id, destination_id), count in pair_counter.most_common(25)
        ]
    )

    return od_sube, diagnostics, top_pairs


def build_model_summary_with_sube(base_summary: pd.DataFrame, sube_time: float) -> pd.DataFrame:
    summary = base_summary.copy()
    baseline = float(summary.loc[summary["escenario"] == "Sin peso", "tiempo_medio"].iloc[0])
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "escenario": "SUBE realista",
                        "tiempo_medio": sube_time,
                        "delta_vs_sin_peso_pct": (sube_time - baseline) / baseline * 100,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return summary


def generate_outputs() -> dict[str, object]:
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    BID_DATA_DIR.mkdir(parents=True, exist_ok=True)

    market_results = generate_market_outputs()
    station_context = load_station_context()
    graph = station_context["graph"]
    positions = station_context["positions"]
    stations = station_context["stations"]
    path_model = build_path_model(graph)

    stop_lookup, mapping_diagnostics, source_config = build_bid_station_lookup(stations)
    od_sube, sube_diagnostics, top_pairs = build_sube_stage_od(stations, stop_lookup)
    sube_results = evaluate_od_model(path_model, od_sube)

    station_metrics = market_results["station_metrics"].copy()
    station_metrics["betweenness_sube"] = station_metrics["id"].map(sube_results["betweenness"])
    station_metrics["cambio_sube_vs_atraccion"] = station_metrics["betweenness_sube"] - station_metrics["betweenness_atraccion"]

    model_summary = build_model_summary_with_sube(market_results["model_summary"], sube_results["tiempo_medio"])

    sube_map_summary = station_metrics[["id", "linea", "estacion", "betweenness_sube"]].copy()
    sube_values = dict(zip(station_metrics["id"], station_metrics["betweenness_sube"], strict=True))
    draw_metric_map(
        graph,
        positions,
        sube_values,
        sube_map_summary,
        metric="betweenness_sube",
        title="Intermediacion con flujos SUBE realistas",
        subtitle="Betweenness recalculada usando etapas SUBE observadas en el subte.",
        output=FIGURES_DIR / "betweenness_sube_realista.png",
    )

    diagnostics = pd.concat(
        [
            source_config.rename(columns={"parametro": "indicador", "valor": "valor"}),
            pd.DataFrame(
                [
                    {
                        "indicador": "bid_subte_stops_without_network_station",
                        "valor": int(len(mapping_diagnostics["bid_stops_without_network_station"])),
                    },
                    {
                        "indicador": "network_stations_without_bid_stop",
                        "valor": int(len(mapping_diagnostics["network_stations_without_bid_stop"])),
                    },
                ]
            ),
            sube_diagnostics,
        ],
        ignore_index=True,
    )

    station_metrics.to_csv(OUTPUT_DIR / "estaciones_modelo_sube.csv", index=False)
    model_summary.to_csv(OUTPUT_DIR / "resumen_modelos_od_con_sube.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "diagnosticos_sube.csv", index=False)
    mapping_diagnostics["bid_stops_without_network_station"].to_csv(
        OUTPUT_DIR / "bid_stops_sin_estacion_red.csv", index=False
    )
    mapping_diagnostics["network_stations_without_bid_stop"].to_csv(
        OUTPUT_DIR / "estaciones_red_sin_bid_stop.csv", index=False
    )
    top_pairs.to_csv(OUTPUT_DIR / "top_pares_od_sube.csv", index=False)
    od_sube.to_csv(OUTPUT_DIR / "matriz_od_sube.csv", index=True)

    return {
        "config": source_config,
        "diagnostics": diagnostics,
        "mapping_diagnostics": mapping_diagnostics,
        "model_summary": model_summary,
        "station_metrics": station_metrics,
        "top_pairs": top_pairs,
        "top_betweenness_sube": station_metrics.sort_values("betweenness_sube", ascending=False)[
            ["estacion", "linea", "betweenness_atraccion", "betweenness_sube", "cambio_sube_vs_atraccion"]
        ].head(12),
        "top_gain_sube_vs_atraccion": station_metrics.sort_values("cambio_sube_vs_atraccion", ascending=False)[
            ["estacion", "linea", "betweenness_atraccion", "betweenness_sube", "cambio_sube_vs_atraccion"]
        ].head(12),
        "top_loss_sube_vs_atraccion": station_metrics.sort_values("cambio_sube_vs_atraccion")[
            ["estacion", "linea", "betweenness_atraccion", "betweenness_sube", "cambio_sube_vs_atraccion"]
        ].head(12),
        "od_sube": od_sube,
    }


if __name__ == "__main__":
    generate_outputs()
