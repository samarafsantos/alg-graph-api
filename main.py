from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import duckdb


from graph_logic import load_graph_data, get_graph_markers

# Variáveis de Configuração
PATH_DATABASE = "./maritime_db.duckdb"
ID_RODADA_ALVO = "2025111700"
HORIZONTE_ALVO = 0
NODE_LIMIT = 1000

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Evento de Inicialização ---
# (aparentemente deprecado)
@app.on_event("startup")
async def startup_event():
    try:
        load_graph_data(PATH_DATABASE, ID_RODADA_ALVO, HORIZONTE_ALVO, NODE_LIMIT)
    except Exception as e:
        print(f"Erro ao carregar o grafo: {e}")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/markers")
def get_markers(limit: int = 10):
    """
    Retorna 'limit' marcadores contendo lat, lon, hs.
    """

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
def get_path(origin: str, destination: str):
    """
    TODO: Rota que retornará o melhor caminho
    """
    return {
        "origin": origin,
        "destination": destination,
        "path": []
    }