import duckdb
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import heapq
import io

GRAPH = None
KD_TREE = None
NODES_LIST = []


def load_graph_data(db_path, id_rodada, horizonte, limit):
    """Carrega dados do DuckDB e monta o grafo."""
    global GRAPH, KD_TREE, NODES_LIST
    print("Iniciando carregamento do Grafo...")

    con = duckdb.connect(db_path, read_only=True)

    granularity = 100

    query = f"""
        SELECT 
            AVG(latitude) AS latitude,
            AVG(longitude) AS longitude,
            AVG(hs_metro) AS hs
        FROM DADOS_GRADE
        WHERE id_rodada = '{id_rodada}'
        AND horizonte_horas = {horizonte}
        GROUP BY 
            FLOOR(latitude * {granularity}),
            FLOOR(longitude * {granularity})
        LIMIT {limit}
    """

    df = con.execute(query).df()
    con.close()

    G = nx.Graph()

    for idx, row in df.iterrows():
        G.add_node(
            int(idx),
            lat=float(row['latitude']),
            lon=float(row['longitude']),
            hs=float(row['hs'])
        )

    node_ids = sorted(G.nodes())
    
    if len(node_ids) == 0:
        GRAPH = G
        KD_TREE = None
        NODES_LIST = []
        print("Aviso: Nenhum nó carregado no grafo.")
        return

    coords = np.array([[G.nodes[n]['lat'], G.nodes[n]['lon']] for n in node_ids])
    tree = cKDTree(coords)

    k_neighbors = 4
    max_distance = 0.15
    
    for i, node_id in enumerate(node_ids):
        distances, indices = tree.query(coords[i], k=k_neighbors + 1)
        
        for dist, idx in zip(distances[1:], indices[1:]):
            if dist > max_distance:
                continue
                
            neighbor_id = node_ids[idx]
            
            if not G.has_edge(node_id, neighbor_id):
                avg_hs = (G.nodes[node_id]['hs'] + G.nodes[neighbor_id]['hs']) / 2.0
                risk_factor = 1.0 + max(0, avg_hs)
                weight = dist * risk_factor
                
                G.add_edge(node_id, neighbor_id, weight=float(weight))

    GRAPH = G
    KD_TREE = tree
    NODES_LIST = node_ids

    print(f"Grafo carregado com {len(G.nodes)} nós e {len(G.edges)} conexões.")


def get_graph_markers(limit=10):
    if GRAPH is None:
        return []

    markers = []
    for i, (node_id, data) in enumerate(GRAPH.nodes(data=True)):
        if i >= limit:
            break

        markers.append({
            "lat": data['lat'],
            "lon": data['lon'],
            "hs": data['hs']
        })

    return markers


def find_nearest_node(lat, lon):
    if KD_TREE is None:
        raise ValueError("KD-Tree não inicializada")

    dist, idx = KD_TREE.query([lat, lon])
    return NODES_LIST[idx]


def dijkstra_manual(graph, start, end):

    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0.0

    previous = {}
    visited = set()

    heap = [(0.0, start)]

    while heap:
        current_dist, node = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)

        if node == end:
            break

        for neighbor in graph.neighbors(node):
            weight = graph.edges[node, neighbor]['weight']
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))

    if end not in previous and start != end:
        return None

    # Handle case where start == end
    if start == end:
        return [start]

    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()

    return path


def compute_shortest_path(lat_o, lon_o, lat_d, lon_d):
    """Calcula o menor caminho usando Dijkstra."""
    if GRAPH is None:
        raise ValueError("Grafo não foi carregado.")

    origin_node = find_nearest_node(lat_o, lon_o)
    dest_node = find_nearest_node(lat_d, lon_d)

    path = dijkstra_manual(GRAPH, origin_node, dest_node)

    if path is None:
        return None

    result = []
    for node in path:
        n = GRAPH.nodes[node]
        result.append({
            "node": int(node),
            "lat": float(n["lat"]),
            "lon": float(n["lon"]),
            "hs": float(n["hs"])
        })

    return result


def export_graph_image_buf(path_nodes=None, figsize=(10, 10), dpi=120):
    """Exporta PNG do grafo com o caminho destacado."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if GRAPH is None:
        raise ValueError("Grafo não carregado.")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Grafo de Rota com Caminho Destacado")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

    xs = [GRAPH.nodes[n]["lon"] for n in GRAPH.nodes]
    ys = [GRAPH.nodes[n]["lat"] for n in GRAPH.nodes]
    ax.scatter(xs, ys, s=10, color="gray", zorder=1)

    for u, v in GRAPH.edges():
        x1 = GRAPH.nodes[u]["lon"]
        y1 = GRAPH.nodes[u]["lat"]
        x2 = GRAPH.nodes[v]["lon"]
        y2 = GRAPH.nodes[v]["lat"]
        ax.plot([x1, x2], [y1, y2], color="lightblue", linewidth=0.5, zorder=0)

    if path_nodes and len(path_nodes) > 1:
        px = [GRAPH.nodes[n]["lon"] for n in path_nodes]
        py = [GRAPH.nodes[n]["lat"] for n in path_nodes]
        ax.plot(px, py, color="red", linewidth=3, label="Caminho")
        ax.scatter(px, py, color="red", s=40, zorder=3)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return buf


def generate_graph_png(lat_o, lon_o, lat_d, lon_d):
    """Gera PNG do grafo com caminho calculado."""
    path = compute_shortest_path(lat_o, lon_o, lat_d, lon_d)

    if path is None:
        return None

    ids = [p["node"] for p in path]

    return export_graph_image_buf(path_nodes=ids)
