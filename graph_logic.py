import duckdb
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


# Variáveis globais para guardar o mapa na memória
GRAPH = None
KD_TREE = None
NODES_LIST = []


def load_graph_data(db_path, id_rodada, horizonte, limit):
    """
    Lê os dados do DuckDB e monta o Grafo de riscos.
    """
    global GRAPH, KD_TREE, NODES_LIST
    
    print("Iniciando carregamento do Grafo...")
    
    con = duckdb.connect(db_path, read_only=True)
    
    # granularity controla o espaçamento dos pontos:
    # quanto maior o valor, maior o bloco espacial (menos pontos no retorno).
    # Ex.: granularity=10 → blocos de ~0.1° (~11 km)
    #     granularity=20 → blocos de ~0.05° (~5 km)
    granularity = 10 
    
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
    
    # Puxa os dados para um DataFrame
    df = con.execute(query).df()
    con.close()
    
    G = nx.Graph()
    
    # Adiciona os nós
    for idx, row in df.iterrows():
        G.add_node(idx, lat=row['latitude'], lon=row['longitude'], hs=row['hs'])
        
    # Prepara estrutura para encontrar vizinhos rápidos
    node_ids = list(G.nodes)
    coords = np.array([[G.nodes[n]['lat'], G.nodes[n]['lon']] for n in node_ids])
    tree = cKDTree(coords)
    
    # Conecta os pontos próximos
    radius = 0.15 
    pairs = tree.query_pairs(r=radius)
    
    for i, j in pairs:
        u = node_ids[i]
        v = node_ids[j]
        
        dist = np.hypot(G.nodes[u]['lat'] - G.nodes[v]['lat'], 
                        G.nodes[u]['lon'] - G.nodes[v]['lon'])
        
        # Lógica de Risco 
        risk_factor = 1.0 + max(0, G.nodes[u]['hs']) 
        weight = dist * risk_factor
        
        G.add_edge(u, v, weight=weight)
        
    GRAPH = G
    KD_TREE = tree
    NODES_LIST = node_ids
    print(f"Grafo carregado com {len(G.nodes)} nós e {len(G.edges)} conexões.")

def get_graph_markers(limit=10):
    """
    Retorna os marcadores diretamente da memória (Grafo)
    """
    if GRAPH is None:
        print("Erro: Grafo não inicializado")
        return []

    markers = []
    # Itera sobre os nós que já estão na memória
    for i, (node_id, data) in enumerate(GRAPH.nodes(data=True)):
        if i >= limit:
            break
        markers.append({
            "lat": data['lat'], 
            "lon": data['lon'], 
            "hs": data['hs']
        })
    
    return markers
