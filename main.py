from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import duckdb

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

@app.get("/")
async def root():
    return {"message": "Hello World"}

PATH_DATABASE = "./maritime_db.duckdb"

ID_RODADA_ALVO = "2025111700"
HORIZONTE_ALVO = 0

@app.get("/markers")
def get_markers(limit: int = 10):
    """
    Retorna 'limit' marcadores contendo lat, lon, hs.
    """

    con = duckdb.connect(PATH_DATABASE, read_only=True)

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
        WHERE id_rodada = '{ID_RODADA_ALVO}'
        AND horizonte_horas = {HORIZONTE_ALVO}

        GROUP BY 
            FLOOR(latitude * {granularity}),
            FLOOR(longitude * {granularity})

        LIMIT {limit}
    """

    results = con.execute(query).fetchall()
    con.close()

    markers = [
        {"lat": lat, "lon": lon, "hs": hs}
        for (lat, lon, hs) in results
    ]

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