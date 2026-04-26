from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"

COLORS_DICT = {
    "A": "tab:blue",
    "B": "tab:red",
    "C": "navy",
    "D": "tab:green",
    "E": "tab:purple",
    "H": "gold",
}


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
    )
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


def node_colors(grafo: nx.Graph) -> list[str]:
    return [COLORS_DICT[grafo.nodes[node]["linea"]] for node in grafo.nodes()]


def draw_network_map(grafo: nx.Graph, pos: dict[int, tuple[float, float]], output: Path) -> None:
    labels = nx.get_node_attributes(grafo, "nombre")
    colors = node_colors(grafo)
    widths = [grafo.edges[edge]["tiempo"] for edge in grafo.edges()]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(grafo, pos, node_size=250, node_color=colors, edgecolors="black", linewidths=0.4)
    nx.draw_networkx_edges(grafo, pos, edge_color="gray", width=[width * 0.6 for width in widths])
    nx.draw_networkx_labels(grafo, pos, labels=labels, font_size=6, font_color="black")
    plt.title("Ejercicio 1.a - Red de subte de CABA")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def draw_centrality_map(
    grafo: nx.Graph,
    pos: dict[int, tuple[float, float]],
    values: dict[int, float],
    title: str,
    output: Path,
    scale: float,
    base: float = 0.0,
) -> None:
    colors = node_colors(grafo)
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(
        grafo,
        pos,
        node_size=[base + values[node] * scale for node in grafo.nodes()],
        node_color=colors,
        edgecolors="black",
        linewidths=0.4,
    )
    nx.draw_networkx_edges(grafo, pos, edge_color="gray", width=1.5)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()


def build_summary(grafo: nx.Graph) -> pd.DataFrame:
    degree = dict(grafo.degree())
    betweenness = nx.betweenness_centrality(grafo, weight="tiempo")
    closeness = nx.closeness_centrality(grafo, distance="tiempo")
    eigenvector = nx.eigenvector_centrality(grafo, weight="invtiempo", max_iter=10000)
    eccentricity = nx.eccentricity(grafo, weight="tiempo")

    summary = pd.DataFrame(
        {
            "id": list(grafo.nodes()),
            "estacion": [grafo.nodes[node]["nombre"] for node in grafo.nodes()],
            "linea": [grafo.nodes[node]["linea"] for node in grafo.nodes()],
            "grado": [degree[node] for node in grafo.nodes()],
            "betweenness": [betweenness[node] for node in grafo.nodes()],
            "closeness": [closeness[node] for node in grafo.nodes()],
            "autovector": [eigenvector[node] for node in grafo.nodes()],
            "excentricidad": [eccentricity[node] for node in grafo.nodes()],
        }
    )

    summary["r_grado"] = summary["grado"].rank(ascending=False, method="min")
    summary["r_betweenness"] = summary["betweenness"].rank(ascending=False, method="min")
    summary["r_closeness"] = summary["closeness"].rank(ascending=False, method="min")
    summary["r_autovector"] = summary["autovector"].rank(ascending=False, method="min")
    summary["r_excentricidad"] = summary["excentricidad"].rank(ascending=True, method="min")
    summary["score_central"] = summary[
        ["r_grado", "r_betweenness", "r_closeness", "r_autovector", "r_excentricidad"]
    ].mean(axis=1)

    summary["r_grado_periferia"] = summary["grado"].rank(ascending=True, method="min")
    summary["r_betweenness_periferia"] = summary["betweenness"].rank(ascending=True, method="min")
    summary["r_closeness_periferia"] = summary["closeness"].rank(ascending=True, method="min")
    summary["r_autovector_periferia"] = summary["autovector"].rank(ascending=True, method="min")
    summary["r_excentricidad_periferia"] = summary["excentricidad"].rank(ascending=False, method="min")
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


def print_rankings(summary: pd.DataFrame) -> None:
    columns = ["estacion", "linea", "grado", "betweenness", "closeness", "autovector", "excentricidad"]

    print("\nTop 10 estaciones mas centrales (score agregado):")
    print(summary[columns].head(10).to_string(index=False))

    print("\nTop 10 estaciones mas perifericas (score agregado):")
    print(
        summary.sort_values(["score_periferico", "excentricidad", "closeness"], ascending=[True, False, True])[columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 por betweenness:")
    print(summary.sort_values("betweenness", ascending=False)[["estacion", "linea", "betweenness"]].head(10).to_string(index=False))

    print("\nTop 10 por closeness:")
    print(summary.sort_values("closeness", ascending=False)[["estacion", "linea", "closeness"]].head(10).to_string(index=False))

    print("\nTop 10 por excentricidad (mas perifericas):")
    print(summary.sort_values("excentricidad", ascending=False)[["estacion", "linea", "excentricidad"]].head(10).to_string(index=False))


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    estaciones, conexiones, posiciones, diagrama = load_data()
    grafo = build_graph(estaciones, conexiones)
    pos_geo, pos_diag = build_positions(posiciones, diagrama)

    draw_network_map(grafo, pos_geo, FIGURES_DIR / "1a_red_subte_mapa.png")
    draw_network_map(grafo, pos_diag, FIGURES_DIR / "1a_red_subte_diagrama.png")

    degree = dict(grafo.degree())
    betweenness = nx.betweenness_centrality(grafo, weight="tiempo")
    closeness = nx.closeness_centrality(grafo, distance="tiempo")
    eigenvector = nx.eigenvector_centrality(grafo, weight="invtiempo", max_iter=10000)
    eccentricity = nx.eccentricity(grafo, weight="tiempo")

    draw_centrality_map(
        grafo,
        pos_geo,
        {node: float(value) for node, value in degree.items()},
        "Ejercicio 1.b - Cantidad de conexiones",
        FIGURES_DIR / "1b_grado.png",
        scale=90,
        base=120,
    )
    draw_centrality_map(
        grafo,
        pos_geo,
        betweenness,
        "Ejercicio 1.b - Betweenness",
        FIGURES_DIR / "1b_betweenness.png",
        scale=7000,
        base=80,
    )
    draw_centrality_map(
        grafo,
        pos_geo,
        closeness,
        "Ejercicio 1.b - Closeness",
        FIGURES_DIR / "1b_closeness.png",
        scale=5000,
        base=80,
    )
    draw_centrality_map(
        grafo,
        pos_geo,
        eigenvector,
        "Ejercicio 1.b - Centralidad de autovector",
        FIGURES_DIR / "1b_autovector.png",
        scale=4000,
        base=80,
    )
    draw_centrality_map(
        grafo,
        pos_geo,
        eccentricity,
        "Ejercicio 1.b - Excentricidad",
        FIGURES_DIR / "1b_excentricidad.png",
        scale=20,
        base=40,
    )

    summary = build_summary(grafo)
    summary.to_csv(OUTPUT_DIR / "centralidad_estaciones.csv", index=False)
    print(f"Nodos: {grafo.number_of_nodes()}")
    print(f"Aristas: {grafo.number_of_edges()}")
    print(f"Transbordos: {sum(1 for edge in grafo.edges() if grafo.edges[edge]['tipo'] == 'transbordo')}")
    print(f"Radio de la red: {nx.radius(grafo, weight='tiempo')} min")
    print(f"Diametro de la red: {nx.diameter(grafo, weight='tiempo')} min")
    print_rankings(summary)


if __name__ == "__main__":
    main()
