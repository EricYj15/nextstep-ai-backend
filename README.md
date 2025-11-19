# NextStep AI Backend

Plataforma acadêmica de recrutamento e requalificação construída com FastAPI para demonstrar o uso de tabelas hash, heaps, grafos e algoritmos gulosos.

## Requisitos

- Python 3.10+
- Pip para instalar as dependências listadas em `requirements.txt`

## Como executar

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

A API ficará disponível em `http://127.0.0.1:8000` com documentação interativa em `/docs`.

## Endpoints principais

- `GET /health` — verificação simples.
- `GET /candidates`, `GET /jobs`, `GET /courses` — consultas aos dados mockados.
- `POST /rank-candidates` — usa heapq para priorizar candidatos com maior interseção de skills.
- `GET /recommend-jobs/{candidate_id}` — explora o grafo candidato-skill-vaga via BFS e sugere oportunidades próximas.
- `POST /generate-study-plan` — aplica algoritmo guloso para montar plano de estudos com cursos disponíveis.

### Exemplos rápidos

- `/rank-candidates`
	```json
	{
		"job_id": "job-1",
		"candidate_ids": ["cand-1", "cand-2", "cand-3"]
	}
	```
- `/generate-study-plan`
	```json
	{
		"candidate_id": "cand-2",
		"job_id": "job-1"
	}
	```

## Estruturas de dados destacadas

- **Tabela hash**: classe `Database` usa `dict` para armazenar candidatos, vagas, skills e cursos com acesso O(1).
- **Heap**: endpoint `/rank-candidates` usa `heapq` como max-heap invertendo o score.
- **Grafo**: classe `NetworkGraph` cria um grafo não direcionado e aplica BFS no endpoint de recomendações.
- **Algoritmo guloso**: `/generate-study-plan` seleciona iterativamente o curso que cobre mais skills faltantes.
