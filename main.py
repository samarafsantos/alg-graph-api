from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from graph_logic import load_graph_data, compute_shortest_path, get_graph_markers, generate_graph_png
import duckdb

PATH_DATABASE = "./maritime_db.duckdb"
ID_RODADA_ALVO = "2025111700"
HORIZONTE_ALVO = 0
NODE_LIMIT = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_graph_data(PATH_DATABASE, ID_RODADA_ALVO, HORIZONTE_ALVO, NODE_LIMIT)
    except Exception as e:
        print(f"Erro ao carregar o grafo: {e}")
    
    yield
    
    print("A desligar API...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/markers")
def get_markers(limit: int = 10):
    """Retorna marcadores contendo lat, lon, hs."""

    markers = get_graph_markers(limit)

    return {"count": len(markers), "items": markers}

@app.get("/tables")
def list_tables():
    con = duckdb.connect(PATH_DATABASE, read_only=True)
    rows = con.execute("SHOW TABLES").fetchall()
    con.close()

    return {"tables": [r[0] for r in rows]}

@app.get("/schema/{tabela}")
def get_schema(table: str):
    con = duckdb.connect(PATH_DATABASE, read_only=True)

    query = f"DESCRIBE {table}"
    rows = con.execute(query).fetchall()
    con.close()

    return {
        "table": table,
        "schema": [
            {"column": col, "tyle": type}
            for (col, type, *_rest) in rows
        ]
    }

@app.get("/path")
def get_path(lat_o: float, lon_o: float, lat_d: float, lon_d: float):
    """Retorna o melhor caminho entre origem e destino."""
    result = compute_shortest_path(lat_o, lon_o, lat_d, lon_d)

    if result is None:
        return {
            "error": "Nenhum caminho disponível entre os pontos."
        }

    return {
        "origin": {"lat": lat_o, "lon": lon_o},
        "destination": {"lat": lat_d, "lon": lon_d},
        "path": result,
        "num_points": len(result)
    }

@app.get("/graph")
def get_graph_png(lat_o: float, lon_o: float, lat_d: float, lon_d: float):
    """Retorna PNG do grafo com o caminho destacado."""
    buf = generate_graph_png(lat_o, lon_o, lat_d, lon_d)

    if buf is None:
        return JSONResponse(
            {"error": "Nenhum caminho disponível entre os pontos."},
            status_code=404
        )

    return StreamingResponse(buf, media_type="image/png")

@app.get("/debug/nearest")
def debug_nearest(lat: float, lon: float):
    """Retorna o nó mais próximo de uma coordenada."""
    import graph_logic

    kd = graph_logic.KD_TREE
    nodes = graph_logic.NODES_LIST

    if kd is None:
        return {"error": "KD_TREE não foi inicializada."}

    if not nodes:
        return {"error": "NODES_LIST está vazio."}

    dist, idx = kd.query([lat, lon])
    node_id = nodes[idx]

    node = graph_logic.GRAPH.nodes[node_id]

    return {
        "input": {"lat": lat, "lon": lon},
        "nearest_node": {
            "node": int(node_id),
            "lat": float(node["lat"]),
            "lon": float(node["lon"]),
            "hs": float(node["hs"])
        },
        "distance": float(dist)
    }

@app.get("/debug/edges")
def debug_edges(limit: int = 20):
    """Lista as conexões do grafo."""
    import graph_logic

    G = graph_logic.GRAPH
    if G is None:
        return {"error": "Grafo não carregado."}

    edges = []
    count = 0

    for u, v in G.edges():
        if count >= limit:
            break

        nu = G.nodes[u]
        nv = G.nodes[v]

        edges.append({
            "origin": {
                "node": int(u),
                "lat": float(nu["lat"]),
                "lon": float(nu["lon"])
            },
            "destination": {
                "node": int(v),
                "lat": float(nv["lat"]),
                "lon": float(nv["lon"])
            }
        })

        count += 1

    return {
        "total_edges": len(G.edges),
        "returned": len(edges),
        "edges": edges
    }
