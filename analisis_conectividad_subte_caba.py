from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import networkx as nx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"

LINE_COLORS = {
    "A": "#1098F7",
    "B": "#E03131",
    "C": "#1C4E80",
    "D": "#2F9E44",
    "E": "#7B2CBF",
    "F": "#C2185B",
    "G": "#FF8C00",
    "H": "#F08C00",
    "I": "#F8A5C2",
}
TRANSFER_COLOR = "#495057"
BACKGROUND = "#F8F9FA"

LABEL_OFFSETS = {
    "9 de Julio": (0.0, -0.85, "center", "top"),
    "Bolívar": (0.55, 0.1, "left", "center"),
    "Catedral": (0.55, -0.35, "left", "top"),
    "Diagonal Norte": (0.55, 0.6, "left", "bottom"),
    "Avenida de Mayo": (-0.6, -0.55, "right", "top"),
    "Carlos Pellegrini": (0.75, -0.15, "left", "top"),
    "Corrientes": (0.75, 0.55, "left", "bottom"),
    "Pueyrredón": (0.7, 0.45, "left", "bottom"),
    "Once": (-0.55, 0.6, "right", "bottom"),
    "Santa Fe": (0.65, -0.45, "left", "top"),
    "Venezuela": (-0.75, 0.0, "right", "center"),
    "Humberto 1º": (-0.75, -0.15, "right", "top"),
    "Jujuy": (-0.75, 0.25, "right", "bottom"),
    "Pichincha": (-0.75, -0.05, "right", "center"),
    "Independencia": (-0.75, 0.0, "right", "center"),
    "Plaza Miserere": (0.0, 0.75, "center", "bottom"),
}

GROUP_LABEL_OFFSETS = {
    frozenset({"Perú", "Catedral", "Bolívar"}): (0.95, 0.1, "left", "center"),
    frozenset({"Lima", "Avenida de Mayo"}): (-0.9, -0.55, "right", "top"),
    frozenset({"Plaza Miserere", "Once"}): (0.0, 0.85, "center", "bottom"),
    frozenset({"Carlos Pellegrini", "Diagonal Norte", "9 de Julio"}): (0.0, -1.0, "center", "top"),
    frozenset({"Pueyrredón", "Corrientes"}): (0.95, 0.55, "left", "bottom"),
    frozenset({"Pueyrredón", "Santa Fe"}): (0.95, -0.45, "left", "top"),
    frozenset({"Jujuy", "Humberto 1º"}): (-0.9, 0.0, "right", "center"),
    frozenset({"Leandro N. Alem", "Correo Central"}): (0.0, 0.8, "center", "bottom"),
    frozenset({"Independencia"}): (-0.9, 0.0, "right", "center"),
    frozenset({"Retiro"}): (0.75, 0.45, "left", "bottom"),
}

FUTURE_LINES = ("F", "G", "I")
FUTURE_TRANSFER_TIME = 5
APPROX_TRANSFER_TIME = 6
FUTURE_SEGMENT_TIME = 2

FUTURE_STATIONS = [
    {
        "id": 101,
        "nombre": "Brandsen y Montes de Oca",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "GCBA 2025",
        "confianza": "alta",
        "pos_diag": (-3.8, 4.2),
    },
    {
        "id": 102,
        "nombre": "Constitución",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "GCBA 2025",
        "confianza": "alta",
        "pos_diag": (-3.1, 6.0),
    },
    {
        "id": 103,
        "nombre": "Cochabamba y Entre Ríos",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "GCBA 2025",
        "confianza": "alta",
        "pos_diag": (-4.2, 6.8),
    },
    {
        "id": 104,
        "nombre": "Chile y Avenida Entre Ríos",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "GCBA 2025",
        "confianza": "alta",
        "pos_diag": (-4.8, 8.4),
    },
    {
        "id": 105,
        "nombre": "Congreso",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "GCBA 2025",
        "confianza": "alta",
        "pos_diag": (-5.0, 10.0),
    },
    {
        "id": 106,
        "nombre": "Tucumán y Callao",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "GCBA 2025",
        "confianza": "alta",
        "pos_diag": (-5.2, 13.1),
    },
    {
        "id": 107,
        "nombre": "Santa Fe/Pizzurno",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "Completo con fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-5.6, 15.0),
    },
    {
        "id": 108,
        "nombre": "Recoleta",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "Completo con fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-6.2, 16.2),
    },
    {
        "id": 109,
        "nombre": "Pueyrredón/Hospital Rivadavia",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "Completo con fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-7.0, 17.2),
    },
    {
        "id": 110,
        "nombre": "Parque Las Heras/Coronel Díaz",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "Completo con fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-8.8, 18.6),
    },
    {
        "id": 111,
        "nombre": "Plaza Italia/Ecoparque",
        "linea": "F",
        "etapa": "tramo principal",
        "fuente": "Completo con fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.8, 20.0),
    },
    {
        "id": 201,
        "nombre": "Retiro",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (0.0, 15.0),
    },
    {
        "id": 202,
        "nombre": "Cerrito",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-1.6, 14.8),
    },
    {
        "id": 203,
        "nombre": "Paraná",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-3.2, 14.6),
    },
    {
        "id": 204,
        "nombre": "Callao",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-4.8, 14.5),
    },
    {
        "id": 205,
        "nombre": "Pueyrredón",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-7.0, 15.5),
    },
    {
        "id": 206,
        "nombre": "Jean Jaurès",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-8.2, 14.6),
    },
    {
        "id": 207,
        "nombre": "Mario Bravo",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-9.2, 13.7),
    },
    {
        "id": 208,
        "nombre": "Gascón",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.0, 12.8),
    },
    {
        "id": 209,
        "nombre": "Estado de Israel",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.8, 11.8),
    },
    {
        "id": 210,
        "nombre": "Parque Centenario",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.4, 10.0),
    },
    {
        "id": 211,
        "nombre": "Cid Campeador",
        "linea": "G",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.1, 8.3),
    },
    {
        "id": 301,
        "nombre": "Directorio",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-12.2, 0.0),
    },
    {
        "id": 302,
        "nombre": "Pedro Goyena",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-12.5, 3.2),
    },
    {
        "id": 303,
        "nombre": "Rivadavia",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-12.9, 8.8),
    },
    {
        "id": 304,
        "nombre": "Aranguren",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-12.0, 10.2),
    },
    {
        "id": 305,
        "nombre": "Díaz Vélez",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.6, 8.4),
    },
    {
        "id": 306,
        "nombre": "Warnes",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.4, 10.2),
    },
    {
        "id": 307,
        "nombre": "Corrientes",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.7, 12.0),
    },
    {
        "id": 308,
        "nombre": "Córdoba",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.7, 14.3),
    },
    {
        "id": 309,
        "nombre": "Costa Rica",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.7, 16.6),
    },
    {
        "id": 310,
        "nombre": "Serrano",
        "linea": "I",
        "etapa": "primera etapa",
        "fuente": "Ley 670 y fuentes secundarias",
        "confianza": "media",
        "pos_diag": (-10.3, 19.0),
    },
]

FUTURE_LINE_SEGMENTS = {
    "F": [(101, 102), (102, 103), (103, 104), (104, 105), (105, 106), (106, 107), (107, 108), (108, 109), (109, 110), (110, 111)],
    "G": [(201, 202), (202, 203), (203, 204), (204, 205), (205, 206), (206, 207), (207, 208), (208, 209), (209, 210), (210, 211)],
    "I": [(301, 302), (302, 303), (303, 304), (304, 305), (305, 306), (306, 307), (307, 308), (308, 309), (309, 310)],
}

FUTURE_TRANSFERS = [
    {"from": ("F", "Constitución"), "to": ("C", "Constitución"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "alta"},
    {"from": ("F", "Cochabamba y Entre Ríos"), "to": ("E", "Entre Ríos"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "media"},
    {"from": ("F", "Congreso"), "to": ("A", "Congreso"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "alta"},
    {"from": ("F", "Tucumán y Callao"), "to": ("B", "Callao"), "tiempo": APPROX_TRANSFER_TIME, "confianza": "media"},
    {"from": ("F", "Tucumán y Callao"), "to": ("D", "Callao"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "alta"},
    {"from": ("F", "Pueyrredón/Hospital Rivadavia"), "to": ("H", "Santa Fe"), "tiempo": APPROX_TRANSFER_TIME, "confianza": "media"},
    {"from": ("F", "Plaza Italia/Ecoparque"), "to": ("D", "Plaza Italia"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "media"},
    {"from": ("G", "Retiro"), "to": ("C", "Retiro"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "alta"},
    {"from": ("G", "Retiro"), "to": ("E", "Retiro"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "alta"},
    {"from": ("G", "Pueyrredón"), "to": ("D", "Pueyrredón"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "alta"},
    {"from": ("G", "Pueyrredón"), "to": ("H", "Santa Fe"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "alta"},
    {"from": ("G", "Estado de Israel"), "to": ("B", "Ángel Gallardo"), "tiempo": APPROX_TRANSFER_TIME, "confianza": "media"},
    {"from": ("I", "Directorio"), "to": ("E", "Emilio Mitre"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "media"},
    {"from": ("I", "Rivadavia"), "to": ("A", "Primera Junta"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "alta"},
    {"from": ("I", "Corrientes"), "to": ("B", "Ángel Gallardo"), "tiempo": APPROX_TRANSFER_TIME, "confianza": "media"},
    {"from": ("I", "Serrano"), "to": ("D", "Scalabrini Ortiz"), "tiempo": APPROX_TRANSFER_TIME, "confianza": "media"},
    {"from": ("G", "Callao"), "to": ("F", "Tucumán y Callao"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "media"},
    {"from": ("G", "Cid Campeador"), "to": ("I", "Díaz Vélez"), "tiempo": FUTURE_TRANSFER_TIME, "confianza": "media"},
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estaciones = pd.read_csv(DATA_DIR / "estaciones.txt")
    conexiones = pd.read_csv(DATA_DIR / "conexiones.txt")
    posiciones = pd.read_csv(DATA_DIR / "estaciones_posicion.txt")
    diagrama = pd.read_csv(DATA_DIR / "estaciones_posicion_diagrama.txt")
    return estaciones, conexiones, posiciones, diagrama


def build_graph(estaciones: pd.DataFrame, conexiones: pd.DataFrame) -> nx.Graph:
    grafo = nx.from_pandas_edgelist(
        conexiones,
        source="source",
        target="target",
        edge_attr=True,
        create_using=nx.Graph(),
    )
    grafo.add_nodes_from(estaciones["id"])
    nx.set_node_attributes(grafo, estaciones.set_index("id").to_dict("index"))
    nx.set_edge_attributes(
        grafo,
        {edge: 1 / grafo.edges[edge]["tiempo"] for edge in grafo.edges()},
        name="invtiempo",
    )
    return grafo


def build_positions(
    posiciones: pd.DataFrame, diagrama: pd.DataFrame
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    pos_geo = {
        row["id"]: (float(row["long"].replace(",", ".")), float(row["lat"].replace(",", ".")))
        for _, row in posiciones.iterrows()
    }
    pos_diag = {row["id"]: (-row["y"], row["x"]) for _, row in diagrama.iterrows()}
    return pos_geo, pos_diag


def future_station_frame(selected_lines: tuple[str, ...] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(FUTURE_STATIONS)
    if selected_lines is None:
        return frame.copy()
    return frame[frame["linea"].isin(selected_lines)].copy()


def future_transfer_frame(selected_lines: tuple[str, ...] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(FUTURE_TRANSFERS)
    if selected_lines is None:
        return frame.copy()

    def include_transfer(row: pd.Series) -> bool:
        from_line = row["from"][0]
        to_line = row["to"][0]
        if from_line not in selected_lines:
            return False
        if to_line in FUTURE_LINES and to_line not in selected_lines:
            return False
        return True

    return frame[frame.apply(include_transfer, axis=1)].copy()


def node_lookup(grafo: nx.Graph) -> dict[tuple[str, str], int]:
    return {(grafo.nodes[node]["linea"], grafo.nodes[node]["nombre"]): node for node in grafo.nodes()}


def build_future_graph(base_graph: nx.Graph, selected_lines: tuple[str, ...]) -> nx.Graph:
    grafo = base_graph.copy()
    future_stations = future_station_frame(selected_lines)

    for station in future_stations.to_dict("records"):
        station_attrs = station.copy()
        station_attrs.pop("pos_diag")
        station_attrs["futuro"] = True
        grafo.add_node(station["id"], **station_attrs)

    for line in selected_lines:
        for source, target in FUTURE_LINE_SEGMENTS[line]:
            grafo.add_edge(
                source,
                target,
                tipo=line,
                tiempo=FUTURE_SEGMENT_TIME,
                invtiempo=1 / FUTURE_SEGMENT_TIME,
                futuro=True,
                confianza="media" if line in {"G", "I"} else "alta",
            )

    lookup = node_lookup(grafo)
    for transfer in future_transfer_frame(selected_lines).to_dict("records"):
        source_key = transfer["from"]
        target_key = transfer["to"]
        if source_key not in lookup or target_key not in lookup:
            continue
        tiempo = int(transfer["tiempo"])
        grafo.add_edge(
            lookup[source_key],
            lookup[target_key],
            tipo="transbordo",
            tiempo=tiempo,
            invtiempo=1 / tiempo,
            futuro=True,
            confianza=transfer["confianza"],
        )

    return grafo


def build_future_positions(base_pos_diag: dict[int, tuple[float, float]], selected_lines: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    pos = dict(base_pos_diag)
    for station in future_station_frame(selected_lines).to_dict("records"):
        pos[station["id"]] = station["pos_diag"]
    return pos


def validate_inputs(
    estaciones: pd.DataFrame,
    conexiones: pd.DataFrame,
    posiciones: pd.DataFrame,
    diagrama: pd.DataFrame,
    grafo: nx.Graph,
) -> dict[str, int | bool]:
    station_ids = set(estaciones["id"])
    edge_station_ids = set(conexiones["source"]) | set(conexiones["target"])
    geo_ids = set(posiciones["id"])
    diagram_ids = set(diagrama["id"])

    checks = {
        "estaciones": len(estaciones),
        "conexiones": len(conexiones),
        "nodos_grafo": grafo.number_of_nodes(),
        "grafo_conexo": nx.is_connected(grafo),
        "ids_sin_conexion": len(station_ids - edge_station_ids),
        "ids_sin_pos_geo": len(station_ids - geo_ids),
        "ids_sin_pos_diag": len(station_ids - diagram_ids),
    }
    return checks


def compute_metrics(grafo: nx.Graph) -> dict[str, dict[int, float]]:
    return {
        "grado": {node: float(value) for node, value in grafo.degree()},
        "betweenness": nx.betweenness_centrality(grafo, weight="tiempo"),
        "closeness": nx.closeness_centrality(grafo, distance="tiempo"),
        "autovector": nx.eigenvector_centrality(grafo, weight="invtiempo", max_iter=10000),
        "excentricidad": nx.eccentricity(grafo, weight="tiempo"),
    }


def distance_metrics_on_nodes(grafo: nx.Graph, demand_nodes: list[int]) -> dict[str, float]:
    total_distance = 0.0
    total_inverse_distance = 0.0
    pair_count = 0
    diameter = 0.0
    eccentricities = []

    for index, source in enumerate(demand_nodes):
        lengths = nx.single_source_dijkstra_path_length(grafo, source, weight="tiempo")
        relevant_distances = [lengths[target] for target in demand_nodes if target != source]
        eccentricities.append(max(relevant_distances))

        for target in demand_nodes[index + 1 :]:
            distance = float(lengths[target])
            total_distance += distance
            total_inverse_distance += 1 / distance
            pair_count += 1
            diameter = max(diameter, distance)

    return {
        "tiempo_medio": total_distance / pair_count,
        "eficiencia": total_inverse_distance / pair_count,
        "diametro": diameter,
        "radio": min(eccentricities),
    }


def demand_betweenness(grafo: nx.Graph, demand_nodes: list[int]) -> dict[int, float]:
    return nx.betweenness_centrality_subset(
        grafo,
        sources=demand_nodes,
        targets=demand_nodes,
        weight="tiempo",
        normalized=False,
    )


def normalize_demand_betweenness(values: dict[int, float], demand_node_count: int) -> dict[int, float]:
    scale = 2 / ((demand_node_count - 1) * (demand_node_count - 2))
    return {node: value * scale for node, value in values.items()}


def build_summary(grafo: nx.Graph, metrics: dict[str, dict[int, float]]) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "id": list(grafo.nodes()),
            "estacion": [grafo.nodes[node]["nombre"] for node in grafo.nodes()],
            "linea": [grafo.nodes[node]["linea"] for node in grafo.nodes()],
            "grado": [metrics["grado"][node] for node in grafo.nodes()],
            "betweenness": [metrics["betweenness"][node] for node in grafo.nodes()],
            "closeness": [metrics["closeness"][node] for node in grafo.nodes()],
            "autovector": [metrics["autovector"][node] for node in grafo.nodes()],
            "excentricidad": [metrics["excentricidad"][node] for node in grafo.nodes()],
        }
    )

    for metric in ["grado", "betweenness", "closeness", "autovector"]:
        summary[f"r_{metric}"] = summary[metric].rank(ascending=False, method="min")
        summary[f"r_{metric}_periferia"] = summary[metric].rank(ascending=True, method="min")

    summary["r_excentricidad"] = summary["excentricidad"].rank(ascending=True, method="min")
    summary["r_excentricidad_periferia"] = summary["excentricidad"].rank(ascending=False, method="min")

    summary["score_central"] = summary[
        ["r_grado", "r_betweenness", "r_closeness", "r_autovector", "r_excentricidad"]
    ].mean(axis=1)
    summary["score_periferico"] = summary[
        [
            "r_grado_periferia",
            "r_betweenness_periferia",
            "r_closeness_periferia",
            "r_autovector_periferia",
            "r_excentricidad_periferia",
        ]
    ].mean(axis=1)

    return summary.sort_values(["score_central", "betweenness", "closeness"], ascending=[True, False, False])


def build_line_summary(estaciones: pd.DataFrame, grafo: nx.Graph) -> pd.DataFrame:
    estaciones_por_linea = estaciones.groupby("linea").size().rename("estaciones")
    transbordos = {
        linea: sum(
            1
            for node in grafo.nodes()
            if grafo.nodes[node]["linea"] == linea and is_transfer_station(grafo, node)
        )
        for linea in estaciones_por_linea.index
    }
    extremos = {
        linea: sum(
            1
            for node in grafo.nodes()
            if grafo.nodes[node]["linea"] == linea and same_line_degree(grafo, node, linea) == 1
        )
        for linea in estaciones_por_linea.index
    }

    summary = pd.DataFrame({
        "linea": estaciones_por_linea.index,
        "estaciones": estaciones_por_linea.values,
        "estaciones_con_transbordo": [transbordos[linea] for linea in estaciones_por_linea.index],
        "cabeceras": [extremos[linea] for linea in estaciones_por_linea.index],
    })
    return summary


def is_transfer_station(grafo: nx.Graph, node: int) -> bool:
    return any(grafo.edges[node, neighbor]["tipo"] == "transbordo" for neighbor in grafo.neighbors(node))


def same_line_degree(grafo: nx.Graph, node: int, linea: str) -> int:
    return sum(1 for neighbor in grafo.neighbors(node) if grafo.edges[node, neighbor]["tipo"] == linea)


def important_nodes(summary: pd.DataFrame, top_n: int = 8) -> set[int]:
    centrales = summary.nsmallest(top_n, "score_central")["id"]
    transferencias = summary.loc[summary["grado"] >= 3, "id"]
    return set(centrales) | set(transferencias)


def transfer_components(grafo: nx.Graph) -> list[set[int]]:
    transfer_subgraph = nx.Graph(
        (source, target)
        for source, target, data in grafo.edges(data=True)
        if data["tipo"] == "transbordo"
    )
    return [set(component) for component in nx.connected_components(transfer_subgraph)]


def transfer_edges_and_nodes(grafo: nx.Graph) -> tuple[list[tuple[int, int]], list[int]]:
    edges = []
    nodes = set()
    for source, target, data in grafo.edges(data=True):
        if data["tipo"] == "transbordo":
            edges.append((source, target))
            nodes.add(source)
            nodes.add(target)
    return edges, sorted(nodes)


def build_label_groups(grafo: nx.Graph, label_nodes: set[int]) -> list[set[int]]:
    groups = []
    grouped_nodes = set()

    for component in transfer_components(grafo):
        if component & label_nodes:
            groups.append(component)
            grouped_nodes.update(component)

    for node in label_nodes - grouped_nodes:
        groups.append({node})

    return groups


def group_names(grafo: nx.Graph, nodes: set[int]) -> frozenset[str]:
    return frozenset(grafo.nodes[node]["nombre"] for node in nodes)


def label_anchor(name: str, x: float, y: float) -> tuple[float, float, str, str]:
    if name in LABEL_OFFSETS:
        dx, dy, ha, va = LABEL_OFFSETS[name]
        return x + dx, y + dy, ha, va

    if x >= 0 and y >= 0:
        return x + 0.45, y + 0.35, "left", "bottom"
    if x >= 0 and y < 0:
        return x + 0.45, y - 0.35, "left", "top"
    if x < 0 and y >= 0:
        return x - 0.45, y + 0.35, "right", "bottom"
    return x - 0.45, y - 0.35, "right", "top"


def group_label_anchor(grafo: nx.Graph, pos: dict[int, tuple[float, float]], nodes: set[int]) -> tuple[float, float, str, str]:
    xs = [pos[node][0] for node in nodes]
    ys = [pos[node][1] for node in nodes]
    x = sum(xs) / len(xs)
    y = sum(ys) / len(ys)
    names = group_names(grafo, nodes)

    if names in GROUP_LABEL_OFFSETS:
        dx, dy, ha, va = GROUP_LABEL_OFFSETS[names]
        return x + dx, y + dy, ha, va

    if len(names) == 1:
        only_name = next(iter(names))
        return label_anchor(only_name, x, y)

    return label_anchor("", x, y)


def node_sort_key(grafo: nx.Graph, pos: dict[int, tuple[float, float]], node: int) -> tuple[float, float, str, str]:
    x, y = pos[node]
    return (x, y, grafo.nodes[node]["linea"], grafo.nodes[node]["nombre"])


def line_letters_box(lines: list[str]) -> list[TextArea]:
    parts: list[TextArea] = [TextArea(" (", textprops={"fontsize": 8, "color": TRANSFER_COLOR})]
    for index, line in enumerate(lines):
        parts.append(
            TextArea(
                line,
                textprops={"fontsize": 8, "color": LINE_COLORS[line], "fontweight": "bold"},
            )
        )
        if index < len(lines) - 1:
            parts.append(TextArea("/", textprops={"fontsize": 8, "color": TRANSFER_COLOR}))
    parts.append(TextArea(")", textprops={"fontsize": 8, "color": TRANSFER_COLOR}))
    return parts


def label_box(grafo: nx.Graph, pos: dict[int, tuple[float, float]], nodes: set[int]) -> HPacker:
    ordered_nodes = sorted(nodes, key=lambda node: node_sort_key(grafo, pos, node))
    grouped_names: dict[str, list[str]] = {}
    for node in ordered_nodes:
        name = grafo.nodes[node]["nombre"]
        line = grafo.nodes[node]["linea"]
        grouped_names.setdefault(name, [])
        if line not in grouped_names[name]:
            grouped_names[name].append(line)

    unique_names = list(grouped_names)
    parts: list[TextArea] = []

    if len(unique_names) == 1:
        lines = grouped_names[unique_names[0]]
        parts.append(TextArea(unique_names[0], textprops={"fontsize": 8, "color": "#212529", "fontweight": "bold"}))
        if len(lines) > 1:
            parts.extend(line_letters_box(lines))
        return HPacker(children=parts, align="center", pad=0, sep=0)

    for index, name in enumerate(unique_names):
        lines = grouped_names[name]
        if len(lines) == 1:
            parts.append(
                TextArea(
                    name,
                    textprops={"fontsize": 8, "color": LINE_COLORS[lines[0]], "fontweight": "bold"},
                )
            )
        else:
            parts.append(TextArea(name, textprops={"fontsize": 8, "color": "#212529", "fontweight": "bold"}))
            parts.extend(line_letters_box(lines))
        if index < len(unique_names) - 1:
            parts.append(TextArea(" / ", textprops={"fontsize": 8, "color": TRANSFER_COLOR}))

    return HPacker(children=parts, align="center", pad=0, sep=0)


def box_alignment(ha: str, va: str) -> tuple[float, float]:
    align_x = {"left": 0.0, "center": 0.5, "right": 1.0}[ha]
    align_y = {"bottom": 0.0, "center": 0.5, "top": 1.0}[va]
    return align_x, align_y


def draw_labels(ax: plt.Axes, grafo: nx.Graph, pos: dict[int, tuple[float, float]], label_nodes: set[int], alpha: float) -> None:
    for nodes in build_label_groups(grafo, label_nodes):
        lx, ly, ha, va = group_label_anchor(grafo, pos, nodes)
        annotation = AnnotationBbox(
            label_box(grafo, pos, nodes),
            (lx, ly),
            xycoords="data",
            frameon=True,
            box_alignment=box_alignment(ha, va),
            bboxprops={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": alpha},
        )
        ax.add_artist(annotation)


def draw_transfer_overlay(
    ax: plt.Axes,
    grafo: nx.Graph,
    pos: dict[int, tuple[float, float]],
    node_sizes: dict[int, float] | None = None,
) -> None:
    transfer_edges, transfer_nodes = transfer_edges_and_nodes(grafo)
    if transfer_edges:
        nx.draw_networkx_edges(
            grafo,
            pos,
            edgelist=transfer_edges,
            edge_color="white",
            width=5.4,
            alpha=0.95,
            ax=ax,
        )
        nx.draw_networkx_edges(
            grafo,
            pos,
            edgelist=transfer_edges,
            edge_color=TRANSFER_COLOR,
            width=2.8,
            style="dashed",
            alpha=0.98,
            ax=ax,
        )

    if transfer_nodes:
        if node_sizes is None:
            sizes = [330 for _ in transfer_nodes]
        else:
            sizes = [node_sizes[node] + 120 for node in transfer_nodes]
        nx.draw_networkx_nodes(
            grafo,
            pos,
            nodelist=transfer_nodes,
            node_size=sizes,
            node_color="none",
            edgecolors=TRANSFER_COLOR,
            linewidths=2.0,
            ax=ax,
        )


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def draw_network_overview(
    grafo: nx.Graph,
    pos: dict[int, tuple[float, float]],
    output: Path,
    title: str,
    subtitle: str,
    label_nodes: set[int],
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 9))

    line_edges = {linea: [] for linea in LINE_COLORS}
    transfer_edges = []
    for source, target, data in grafo.edges(data=True):
        if data["tipo"] == "transbordo":
            transfer_edges.append((source, target))
        else:
            line_edges[data["tipo"]].append((source, target))

    for linea, edges in line_edges.items():
        if edges:
            nx.draw_networkx_edges(
                grafo,
                pos,
                edgelist=edges,
                edge_color=LINE_COLORS[linea],
                width=2.5,
                alpha=0.9,
                ax=ax,
            )

    node_sizes = {node: 260 if node in label_nodes else 120 for node in grafo.nodes()}
    node_colors = [LINE_COLORS[grafo.nodes[node]["linea"]] for node in grafo.nodes()]
    nx.draw_networkx_nodes(
        grafo,
        pos,
        node_size=[node_sizes[node] for node in grafo.nodes()],
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.8,
        ax=ax,
    )

    draw_transfer_overlay(ax, grafo, pos, node_sizes=node_sizes)

    draw_labels(ax, grafo, pos, label_nodes, alpha=0.82)

    present_lines = sorted({grafo.nodes[node]["linea"] for node in grafo.nodes()})
    legend_items = [
        Line2D([0], [0], color=LINE_COLORS[linea], lw=3, label=f"Linea {linea}")
        for linea in present_lines
    ]
    legend_items.append(Line2D([0], [0], color=TRANSFER_COLOR, lw=2.8, linestyle="--", label="Enlace de transbordo"))
    legend_items.append(
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=TRANSFER_COLOR,
            markeredgewidth=2,
            linestyle="none",
            label="Estacion con transbordo",
        )
    )
    ax.legend(handles=legend_items, loc="upper left", frameon=False, ncol=2)
    ax.set_title(f"{title}\n{subtitle}", loc="left")
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.03)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scale_node_sizes(values: dict[int, float], min_size: float = 180, max_size: float = 1200) -> dict[int, float]:
    series = pd.Series(values, dtype=float)
    if series.nunique() == 1:
        return {node: (min_size + max_size) / 2 for node in values}
    scaled = (series - series.min()) / (series.max() - series.min())
    return (min_size + scaled * (max_size - min_size)).to_dict()


def draw_metric_map(
    grafo: nx.Graph,
    pos: dict[int, tuple[float, float]],
    values: dict[int, float],
    summary: pd.DataFrame,
    metric: str,
    title: str,
    subtitle: str,
    output: Path,
    top_n: int = 8,
) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    top_nodes = set(summary.nlargest(top_n, metric)["id"])
    sizes = scale_node_sizes(values)
    node_values = [values[node] for node in grafo.nodes()]

    nx.draw_networkx_edges(grafo, pos, edge_color="#CED4DA", width=1.6, ax=ax)
    scatter = nx.draw_networkx_nodes(
        grafo,
        pos,
        node_size=[sizes[node] for node in grafo.nodes()],
        node_color=node_values,
        cmap="YlOrRd",
        edgecolors="white",
        linewidths=0.9,
        ax=ax,
    )

    draw_transfer_overlay(ax, grafo, pos, node_sizes=sizes)

    draw_labels(ax, grafo, pos, top_nodes, alpha=0.86)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric.capitalize())
    ax.set_title(f"{title}\n{subtitle}", loc="left")
    ax.axis("off")
    fig.subplots_adjust(left=0.04, right=0.93, top=0.9, bottom=0.04)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_ranking_bars(
    ranking: pd.DataFrame,
    score_column: str,
    output: Path,
    title: str,
    subtitle: str,
) -> None:
    apply_plot_style()
    ordered = ranking.iloc[::-1].copy()
    colors = [LINE_COLORS[linea] for linea in ordered["linea"]]
    labels = [f"{estacion} ({linea})" for estacion, linea in zip(ordered["estacion"], ordered["linea"], strict=True)]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(labels, ordered[score_column], color=colors, alpha=0.95)
    ax.set_title(f"{title}\n{subtitle}", loc="left")
    ax.set_xlabel(score_column.replace("_", " ").capitalize())
    ax.grid(axis="x", color="#DEE2E6", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, ordered[score_column], strict=True):
        ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_line_summary(line_summary: pd.DataFrame, output: Path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    colors = [LINE_COLORS[linea] for linea in line_summary["linea"]]
    axes[0].bar(line_summary["linea"], line_summary["estaciones"], color=colors)
    axes[0].set_title("Cantidad de estaciones por linea", loc="left")
    axes[0].set_ylabel("Estaciones")
    axes[0].grid(axis="y", color="#DEE2E6", linewidth=0.8)
    axes[0].set_axisbelow(True)

    axes[1].bar(line_summary["linea"], line_summary["estaciones_con_transbordo"], color=colors)
    axes[1].set_title("Estaciones con transbordo por linea", loc="left")
    axes[1].set_ylabel("Estaciones")
    axes[1].grid(axis="y", color="#DEE2E6", linewidth=0.8)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def future_key_nodes(grafo: nx.Graph, summary: pd.DataFrame) -> set[int]:
    future_summary = summary[summary["linea"].isin(FUTURE_LINES)]
    terminals = {
        node
        for node in grafo.nodes()
        if grafo.nodes[node].get("futuro") and same_line_degree(grafo, node, grafo.nodes[node]["linea"]) == 1
    }
    transfers = {node for node in grafo.nodes() if grafo.nodes[node].get("futuro") and is_transfer_station(grafo, node)}
    central = set(future_summary.sort_values(["betweenness", "closeness"], ascending=[False, False]).head(8)["id"])
    return terminals | transfers | central


def draw_future_scenario_comparison(scenarios: pd.DataFrame, output: Path) -> None:
    apply_plot_style()
    compare = scenarios[scenarios["escenario"].isin(FUTURE_LINES)].copy()
    colors = [LINE_COLORS[linea] for linea in compare["escenario"]]
    metrics = [
        ("mejora_tiempo_pct", "Reduccion del tiempo medio entre estaciones actuales", "% vs red actual"),
        ("mejora_eficiencia_pct", "Ganancia de eficiencia global", "% vs red actual"),
        ("caida_betweenness_hubs_pct", "Redistribucion de flujos en hubs actuales", "% de caida en betweenness"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (column, title, ylabel) in zip(axes, metrics, strict=True):
        bars = ax.bar(compare["escenario"], compare[column], color=colors, alpha=0.95)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DEE2E6", linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, compare[column], strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_hub_relief(hub_relief: pd.DataFrame, output: Path, top_n: int = 6) -> None:
    apply_plot_style()
    selected = hub_relief.sort_values("betweenness_actual", ascending=False).head(top_n).iloc[::-1].copy()
    labels = [f"{row.estacion} ({row.linea})" for row in selected.itertuples()]
    width = 0.22
    offsets = {"F": -width, "G": 0.0, "I": width}
    y_positions = list(range(len(selected)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for line in FUTURE_LINES:
        values = selected[f"caida_{line}_pct"]
        bars = ax.barh(
            [y + offsets[line] for y in y_positions],
            values,
            height=0.2,
            color=LINE_COLORS[line],
            alpha=0.95,
            label=f"Linea {line}",
        )
        for bar, value in zip(bars, values, strict=True):
            if value >= 0:
                text_x = value + 0.04
                ha = "left"
            else:
                text_x = value - 0.04
                ha = "right"
            ax.text(text_x, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha=ha, fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Caida porcentual de betweenness")
    ax.set_title("Hubs actuales mas descargados por cada linea futura", loc="left")
    ax.grid(axis="x", color="#DEE2E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#ADB5BD", linewidth=1)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_future_analysis(
    base_graph: nx.Graph,
    base_pos_diag: dict[int, tuple[float, float]],
    base_summary: pd.DataFrame,
) -> dict[str, object]:
    scenario_definitions = {
        "F": ("F",),
        "G": ("G",),
        "I": ("I",),
        "F+G+I": FUTURE_LINES,
    }
    demand_nodes = sorted(base_graph.nodes())
    base_distance_metrics = distance_metrics_on_nodes(base_graph, demand_nodes)
    base_subset_betweenness = normalize_demand_betweenness(demand_betweenness(base_graph, demand_nodes), len(demand_nodes))
    hub_ids = list(base_summary.sort_values("betweenness", ascending=False).head(8)["id"])
    base_hub_total = sum(base_subset_betweenness[node] for node in hub_ids)

    scenario_rows = []
    hub_relief_rows = []
    scenario_graphs: dict[str, nx.Graph] = {}
    scenario_summaries: dict[str, pd.DataFrame] = {}
    scenario_positions: dict[str, dict[int, tuple[float, float]]] = {}
    scenario_subsets: dict[str, dict[int, float]] = {}

    for label, lines in scenario_definitions.items():
        graph = build_future_graph(base_graph, lines)
        summary = build_summary(graph, compute_metrics(graph))
        pos_diag = build_future_positions(base_pos_diag, lines)
        distance_metrics = distance_metrics_on_nodes(graph, demand_nodes)
        subset = normalize_demand_betweenness(demand_betweenness(graph, demand_nodes), len(demand_nodes))
        future_nodes = [node for node in graph.nodes() if graph.nodes[node].get("futuro")]
        hub_total = sum(subset[node] for node in hub_ids)

        scenario_rows.append(
            {
                "escenario": label,
                "estaciones_nuevas": len(future_nodes),
                "transbordos_nuevos": sum(
                    1
                    for _, _, data in graph.edges(data=True)
                    if data.get("futuro") and data["tipo"] == "transbordo"
                ),
                "tiempo_medio": distance_metrics["tiempo_medio"],
                "mejora_tiempo_pct": (base_distance_metrics["tiempo_medio"] - distance_metrics["tiempo_medio"]) / base_distance_metrics["tiempo_medio"] * 100,
                "eficiencia": distance_metrics["eficiencia"],
                "mejora_eficiencia_pct": (distance_metrics["eficiencia"] - base_distance_metrics["eficiencia"]) / base_distance_metrics["eficiencia"] * 100,
                "diametro": distance_metrics["diametro"],
                "radio": distance_metrics["radio"],
                "caida_betweenness_hubs_pct": (base_hub_total - hub_total) / base_hub_total * 100,
                "betweenness_nuevas_estaciones": sum(subset[node] for node in future_nodes),
            }
        )

        for hub_id in hub_ids:
            current_value = base_subset_betweenness[hub_id]
            scenario_value = subset[hub_id]
            hub_relief_rows.append(
                {
                    "id": hub_id,
                    "estacion": base_graph.nodes[hub_id]["nombre"],
                    "linea": base_graph.nodes[hub_id]["linea"],
                    "escenario": label,
                    "betweenness_actual": current_value,
                    "betweenness_escenario": scenario_value,
                    "caida_pct": (current_value - scenario_value) / current_value * 100 if current_value else 0.0,
                }
            )

        scenario_graphs[label] = graph
        scenario_summaries[label] = summary
        scenario_positions[label] = pos_diag
        scenario_subsets[label] = subset

    scenario_table = pd.DataFrame(scenario_rows)
    scenario_table["escenario"] = pd.Categorical(scenario_table["escenario"], categories=list(scenario_definitions), ordered=True)
    scenario_table = scenario_table.sort_values("escenario").reset_index(drop=True)

    hub_relief = pd.DataFrame(hub_relief_rows)
    hub_relief_compare = (
        hub_relief[hub_relief["escenario"].isin(FUTURE_LINES)]
        .pivot_table(index=["id", "estacion", "linea", "betweenness_actual"], columns="escenario", values="caida_pct")
        .reset_index()
        .rename(columns={line: f"caida_{line}_pct" for line in FUTURE_LINES})
        .fillna(0.0)
    )
    hub_relief_compare.columns.name = None

    future_summary = scenario_summaries["F+G+I"].copy()
    future_summary["betweenness_demanda_actual"] = future_summary["id"].map(scenario_subsets["F+G+I"])
    future_stations_summary = future_summary[future_summary["linea"].isin(FUTURE_LINES)].copy()

    line_centrality_rows = []
    for line in FUTURE_LINES:
        line_data = future_stations_summary[future_stations_summary["linea"] == line].copy()
        top_betweenness_station = line_data.sort_values(["betweenness_demanda_actual", "closeness"], ascending=[False, False]).iloc[0]["estacion"]
        top_closeness_station = line_data.sort_values("closeness", ascending=False).iloc[0]["estacion"]
        line_centrality_rows.append(
            {
                "linea": line,
                "estaciones_modeladas": len(line_data),
                "betweenness_demanda_actual_total": line_data["betweenness_demanda_actual"].sum(),
                "betweenness_demanda_actual_media": line_data["betweenness_demanda_actual"].mean(),
                "closeness_media": line_data["closeness"].mean(),
                "top_estacion_betweenness": top_betweenness_station,
                "top_estacion_closeness": top_closeness_station,
            }
        )
    line_centrality = pd.DataFrame(line_centrality_rows).sort_values("betweenness_demanda_actual_total", ascending=False)

    top_future_stations = future_stations_summary[
        ["estacion", "linea", "betweenness_demanda_actual", "betweenness", "closeness", "score_central"]
    ].sort_values(["betweenness_demanda_actual", "closeness"], ascending=[False, False]).head(12)

    future_station_model = future_station_frame().drop(columns="pos_diag").copy()
    future_transfer_model = future_transfer_frame().copy()
    future_transfer_model[["linea_origen", "estacion_origen"]] = pd.DataFrame(future_transfer_model["from"].tolist(), index=future_transfer_model.index)
    future_transfer_model[["linea_destino", "estacion_destino"]] = pd.DataFrame(future_transfer_model["to"].tolist(), index=future_transfer_model.index)
    future_transfer_model = future_transfer_model.drop(columns=["from", "to"])

    full_future_graph = scenario_graphs["F+G+I"]
    full_future_pos = scenario_positions["F+G+I"]
    draw_network_overview(
        full_future_graph,
        full_future_pos,
        FIGURES_DIR / "red_subte_caba_futuro_esquematico.png",
        title="Escenario futuro con las lineas F, G e I",
        subtitle="La traza combina la red actual con F oficial 2025 completada con fuentes secundarias, y G/I segun la primera etapa de los planes historicos.",
        label_nodes=future_key_nodes(full_future_graph, future_summary),
    )
    draw_future_scenario_comparison(scenario_table, FIGURES_DIR / "comparacion_lineas_futuras.png")
    draw_hub_relief(hub_relief_compare, FIGURES_DIR / "alivio_hubs_futuros.png")

    future_station_model.to_csv(OUTPUT_DIR / "modelo_estaciones_futuras.csv", index=False)
    future_transfer_model.to_csv(OUTPUT_DIR / "modelo_transbordos_futuros.csv", index=False)
    scenario_table.to_csv(OUTPUT_DIR / "escenarios_futuros_resumen.csv", index=False)
    hub_relief_compare.to_csv(OUTPUT_DIR / "alivio_hubs_futuros.csv", index=False)
    line_centrality.to_csv(OUTPUT_DIR / "centralidad_lineas_futuras.csv", index=False)
    top_future_stations.to_csv(OUTPUT_DIR / "top_estaciones_futuras.csv", index=False)

    compare_lines = scenario_table[scenario_table["escenario"].isin(FUTURE_LINES)].copy()
    winner_distance = compare_lines.sort_values(["mejora_tiempo_pct", "mejora_eficiencia_pct"], ascending=[False, False]).iloc[0]["escenario"]
    winner_redistribution = compare_lines.sort_values(["caida_betweenness_hubs_pct", "betweenness_nuevas_estaciones"], ascending=[False, False]).iloc[0]["escenario"]

    return {
        "scenario_table": scenario_table,
        "hub_relief": hub_relief_compare,
        "line_centrality": line_centrality,
        "top_future_stations": top_future_stations,
        "future_station_model": future_station_model,
        "future_transfer_model": future_transfer_model,
        "winners": {"distance": winner_distance, "redistribution": winner_redistribution},
    }


def build_key_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "estacion",
        "linea",
        "grado",
        "betweenness",
        "closeness",
        "autovector",
        "excentricidad",
        "score_central",
        "score_periferico",
    ]
    top_central = summary.nsmallest(10, "score_central")[columns].copy()
    top_peripheral = summary.nsmallest(10, "score_periferico")[columns].copy()
    return top_central, top_peripheral


def network_stats(grafo: nx.Graph) -> dict[str, int]:
    return {
        "nodos": grafo.number_of_nodes(),
        "aristas": grafo.number_of_edges(),
        "transbordos": sum(1 for _, _, data in grafo.edges(data=True) if data["tipo"] == "transbordo"),
        "radio": nx.radius(grafo, weight="tiempo"),
        "diametro": nx.diameter(grafo, weight="tiempo"),
    }


def generate_outputs() -> dict[str, object]:
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    estaciones, conexiones, posiciones, diagrama = load_data()
    grafo = build_graph(estaciones, conexiones)
    pos_geo, pos_diag = build_positions(posiciones, diagrama)
    checks = validate_inputs(estaciones, conexiones, posiciones, diagrama, grafo)
    metrics = compute_metrics(grafo)
    summary = build_summary(grafo, metrics)
    line_summary = build_line_summary(estaciones, grafo)
    central_table, peripheral_table = build_key_tables(summary)
    key_nodes = important_nodes(summary)
    future_analysis = build_future_analysis(grafo, pos_diag, summary)

    draw_network_overview(
        grafo,
        pos_geo,
        FIGURES_DIR / "red_subte_caba_mapa_geografico.png",
        title="Red de subte de CABA sobre el mapa",
        subtitle="Colores por linea y etiquetas solo en estaciones clave para evitar saturacion visual.",
        label_nodes=key_nodes,
    )
    draw_network_overview(
        grafo,
        pos_diag,
        FIGURES_DIR / "red_subte_caba_mapa_esquematico.png",
        title="Diagrama esquematico de la red",
        subtitle="La lectura topologica es mas clara: las conexiones de transbordo aparecen en linea punteada.",
        label_nodes=key_nodes,
    )
    draw_metric_map(
        grafo,
        pos_diag,
        metrics["betweenness"],
        summary,
        metric="betweenness",
        title="Estaciones puente segun intermediacion",
        subtitle="Cuanto mayor el nodo, mas rutas minimas pasan por esa estacion.",
        output=FIGURES_DIR / "centralidad_estaciones_por_intermediacion.png",
    )
    draw_metric_map(
        grafo,
        pos_diag,
        metrics["closeness"],
        summary,
        metric="closeness",
        title="Estaciones con mejor acceso al resto de la red",
        subtitle="La cercania pondera el tiempo total necesario para llegar al resto de las estaciones.",
        output=FIGURES_DIR / "centralidad_estaciones_por_cercania.png",
    )
    draw_metric_map(
        grafo,
        pos_diag,
        metrics["excentricidad"],
        summary,
        metric="excentricidad",
        title="Estaciones mas perifericas segun excentricidad",
        subtitle="Valores altos indican mayor distancia minima al punto mas lejano de la red.",
        output=FIGURES_DIR / "centralidad_estaciones_por_excentricidad.png",
    )
    draw_ranking_bars(
        summary.nsmallest(10, "score_central"),
        score_column="score_central",
        output=FIGURES_DIR / "ranking_estaciones_centrales.png",
        title="Top 10 estaciones mas centrales",
        subtitle="Score agregado: combina grado, intermediacion, cercania, autovector y baja excentricidad.",
    )
    draw_ranking_bars(
        summary.nsmallest(10, "score_periferico"),
        score_column="score_periferico",
        output=FIGURES_DIR / "ranking_estaciones_perifericas.png",
        title="Top 10 estaciones mas perifericas",
        subtitle="Score agregado: premia baja conectividad y mayor distancia respecto del resto de la red.",
    )
    draw_line_summary(line_summary, FIGURES_DIR / "resumen_por_linea.png")

    summary.to_csv(OUTPUT_DIR / "caracterizacion_estaciones_subte.csv", index=False)
    central_table.to_csv(OUTPUT_DIR / "top_estaciones_centrales.csv", index=False)
    peripheral_table.to_csv(OUTPUT_DIR / "top_estaciones_perifericas.csv", index=False)
    line_summary.to_csv(OUTPUT_DIR / "resumen_por_linea.csv", index=False)

    return {
        "checks": checks,
        "stats": network_stats(grafo),
        "summary": summary,
        "line_summary": line_summary,
        "top_central": central_table,
        "top_peripheral": peripheral_table,
        "future": future_analysis,
    }


def print_rankings(top_central: pd.DataFrame, top_peripheral: pd.DataFrame) -> None:
    columns = ["estacion", "linea", "grado", "betweenness", "closeness", "autovector", "excentricidad"]
    print("\nTop 10 estaciones mas centrales (score agregado):")
    print(top_central[columns].to_string(index=False))
    print("\nTop 10 estaciones mas perifericas (score agregado):")
    print(top_peripheral[columns].to_string(index=False))


def main() -> None:
    results = generate_outputs()
    checks = results["checks"]
    stats = results["stats"]
    future = results["future"]
    print("Chequeos de consistencia:")
    for key, value in checks.items():
        print(f"- {key}: {value}")
    print(f"\nNodos: {stats['nodos']}")
    print(f"Aristas: {stats['aristas']}")
    print(f"Transbordos: {stats['transbordos']}")
    print(f"Radio de la red: {stats['radio']} min")
    print(f"Diametro de la red: {stats['diametro']} min")
    print_rankings(results["top_central"], results["top_peripheral"])
    print("\nGanadora por reduccion de distancias entre estaciones actuales:", future["winners"]["distance"])
    print("Ganadora por redistribucion de flujos en hubs actuales:", future["winners"]["redistribution"])


if __name__ == "__main__":
    main()
