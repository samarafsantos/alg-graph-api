# Maritime Route Planning API

API FastAPI para planejamento de rotas marítimas usando algoritmo de Dijkstra. Calcula caminhos otimizados considerando altura de ondas (hs) e distância geográfica.

## Características

- **Algoritmo de Dijkstra** implementado manualmente para cálculo de rotas
- **Pesagem baseada em risco**: combina distância geográfica com altura de ondas
- **KD-Tree** para busca eficiente de nós mais próximos
- **Visualização**: geração de imagens PNG mostrando o grafo e a rota calculada
- **Dados marítimos**: integração com DuckDB contendo dados de grade marítima

## Estrutura do Projeto

```
alg-graph-api/
├── main.py # API FastAPI com endpoints
├── graph_logic.py # Lógica do grafo e algoritmo de Dijkstra
├── maritime_db.duckdb # Banco de dados com dados marítimos
├── requirements.txt # Dependências Python
└── README.MD # Este arquivo
```

## Instalação

1. Criar e ativar ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate # macOS/Linux

# ou

.venv\Scripts\activate # Windows
```

2. Instalar dependências:

```bash
pip install -r requirements.txt
```

## Executar

```bash
uvicorn main:app --reload
```

A API estará disponível em `http://127.0.0.1:8000`

## Endpoints

### `GET /`

Endpoint raiz, retorna mensagem de boas-vindas.

### `GET /path`

Calcula o melhor caminho entre dois pontos.

**Parâmetros:**

- `lat_o`: latitude de origem
- `lon_o`: longitude de origem
- `lat_d`: latitude de destino
- `lon_d`: longitude de destino

**Exemplo:**
```
http://127.0.0.1:8000/path?lat_o=-24.9375&lon_o=-46.0625&lat_d=-24.75&lon_d=-46.1875
```

**Resposta:**
```json
{
"origin": {"lat": -24.9375, "lon": -46.0625},
"destination": {"lat": -24.75, "lon": -46.1875},
"path": [
{"node": 694, "lat": -24.9375, "lon": -46.0625, "hs": 1.39},
{"node": 274, "lat": -24.75, "lon": -46.125, "hs": 1.3},
{"node": 273, "lat": -24.75, "lon": -46.1875, "hs": 1.28}
],
"num_points": 3
}
```

### `GET /graph`

Retorna visualização PNG do grafo com o caminho destacado em vermelho.

**Parâmetros:** Mesmos do `/path`

**Exemplo:**
```
http://127.0.0.1:8000/graph?lat_o=-24.9375&lon_o=-46.0625&lat_d=-24.75&lon_d=-46.1875
```

### `GET /markers`

Retorna lista de marcadores do grafo (nós com coordenadas e altura de ondas).

**Parâmetros:**

- `limit`: número de marcadores a retornar (padrão: 10)

### `GET /debug/nearest`

Retorna o nó mais próximo de uma coordenada específica.

**Parâmetros:**

- `lat`: latitude
- `lon`: longitude

### `GET /debug/edges`

Lista as conexões do grafo.

**Parâmetros:**

- `limit`: número de arestas a retornar (padrão: 20)

## Algoritmo

### Cálculo de Peso das Arestas

```python
weight = distance * (1.0 + avg_wave_height)
```

Onde:

- `distance`: distância euclidiana entre os nós
- `avg_wave_height`: média da altura de ondas (hs) dos dois nós

### Conexão de Nós

- Cada nó conecta-se aos seus **4 vizinhos mais próximos**
- Distância máxima de conexão: **0.15 graus** (~16km)
- Garante conectividade mantendo grafo esparso

### Parâmetros Configuráveis

Em `main.py`:

- `NODE_LIMIT = 1000`: número de nós a carregar do banco
- `ID_RODADA_ALVO`: identificador da rodada de dados
- `HORIZONTE_ALVO`: horizonte temporal (horas)

Em `graph_logic.py`:

- `granularity = 100`: agrupamento de coordenadas
- `k_neighbors = 4`: número de vizinhos por nó
- `max_distance = 0.15`: distância máxima de conexão

## Documentação da API

Acesse `http://127.0.0.1:8000/docs` após iniciar o servidor para interface interativa Swagger.
