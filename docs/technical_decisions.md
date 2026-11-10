# Architecture Decision Records (ADRs) — Customer Insights Copilot

## ADR 01: Decoupling Streamlit Frontend from Core Logic via FastAPI

### Context
In the original implementation, `app.py` directly invoked file parsers, spaCy pipelines, VADER sentiment, and report generators. This tight coupling made it impossible to expose REST APIs, run headless batch jobs, integrate external services, or scale frontend and backend services independently.

### Decision
We introduced a dedicated FastAPI backend (`app/api/`) and service layer (`app/services/`). 
Streamlit remains the primary user dashboard but communicates via standard service interfaces and API contracts.

### Consequences
- **Positive:** Enables headless REST API usage, programmatic SDK integration, asynchronous jobs, and automated CI testing with `TestClient`.
- **Negative:** Slightly increased architectural footprint and requirement to manage two runtime processes in production (or unified container).

---

## ADR 02: Selection of FAISS and Sentence-Transformers for Vector Store and RAG

### Context
Customers need to ask natural-language questions across thousands of unstructured reviews (e.g., "What are the most common complaints about battery life?"). Cloud-only vector databases (Pinecone, Weaviate Cloud) violate the project requirement to support 100% offline, local execution without paid subscriptions or external API requirements.

### Decision
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (fast, lightweight, CPU-efficient, 384 dimensions).
- **Vector Store:** `faiss-cpu` with L2-normalized Inner Product (`IndexFlatIP`) index to compute exact cosine similarities locally without external daemons.
- **Persistence:** Local disk serialization (`index.faiss` + review metadata pickle).

### Consequences
- **Positive:** Works 100% offline; zero recurring API costs; sub-millisecond similarity search on local CPU.
- **Negative:** Large datasets (100k+ reviews) require index partitioning (IVF/HNSW) if RAM is constrained.

---

## ADR 03: Dual-Mode Sentiment Pipeline (VADER+spaCy vs. DeBERTa ABSA)

### Context
Transformer-based models (DeBERTa-v3 ABSA) deliver high accuracy on complex linguistic constructs and aspect-conditioned sentiment, but are computationally heavy and slow on CPU without GPU acceleration. Lightweight rule-based methods (VADER + spaCy dependency parsing) run in milliseconds per review.

### Decision
Maintain a dual-inference strategy:
1. **Default:** Lightweight VADER + spaCy clause extraction for instantaneous batch processing.
2. **Advanced:** DeBERTa-v3 ABSA (`yangheng/deberta-v3-base-absa-v1.1`) or custom fine-tuned weights for deep aspect-level accuracy when activated by the user.

### Consequences
- **Positive:** Immediate responsiveness for everyday workloads and resource-constrained environments (like free Streamlit Community Cloud) while maintaining enterprise-grade accuracy capabilities when GPU is present.

---

## ADR 04: Grounded LLM Generation and Strict Anti-Hallucination Controls

### Context
LLMs frequently hallucinate review counts, fabricate statistics, and are vulnerable to prompt injection embedded within customer reviews.

### Decision
1. **Deterministic Statistical Computation:** All percentages, review counts, aspect distributions, and top issue rankings are computed strictly in Python code (`pandas` / `numpy`), never estimated by LLMs.
2. **Untrusted Data Boundary:** Reviews passed into LLM prompts are treated as untrusted strings, encapsulated within structured XML/bracket blocks with explicit instructions prohibiting instruction execution.
3. **Evidence Citations:** Generated answers must cite concrete review IDs (`[ID:xxxx]`).
4. **Insufficient Evidence Fallback:** If similarity scores are low or evidence is lacking, the system deterministically outputs: *"Insufficient evidence in the available customer reviews to answer this question reliably."*
5. **Rule-Based Fallback:** If no LLM API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) is configured, the system gracefully defaults to an extractive summarization engine.

---

## ADR 05: Data Layer and Dataset Isolation

### Context
Users upload multiple independent datasets (e.g. "Smartwatch 2026", "Wireless Earbuds"). Reviews and vector indices must not leak across different datasets or products.

### Decision
- Maintain dataset metadata and review records with explicit `dataset_id` partitioning in SQLite (expandable to PostgreSQL via standard SQLAlchemy / SQL models).
- FAISS search supports dataset filtering and dataset-level deletion to allow safe multi-product management.

---

## ADR 06: Hybrid Async Job Strategy (In-Memory + Celery/Redis)

### Context
Processing large datasets (thousands of reviews) synchronously risks HTTP request timeouts. However, requiring Redis and Celery for simple local development creates unnecessary setup friction.

### Decision
- Provide a built-in background task runner using Python's standard `ThreadPoolExecutor` and in-memory status tracking for lightweight local execution.
- Provide a pluggable Celery + Redis worker configuration via Docker Compose profiles (`--profile async`) for scalable enterprise deployment.
