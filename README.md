# 🧠 Autonomous Reasoning System

A modular AI framework that simulates autonomous reasoning, reflection, and memory in an agentic system.  
Designed to explore how artificial cognition can evolve from reactive chat to **self-directed reasoning**.

---

# 🧩 Why This Architecture Is Unique

Most agent frameworks in the current ecosystem depend on heavy stacks of abstractions (LangChain, LlamaIndex, CrewAI, AutoGen), hundreds of indirect dependencies, and thousands of lines of glue code.  
This project takes a very different approach.

---

## 1. **Minimal Code, Maximum Functionality**

The entire agent — including memory, retrieval, knowledge graph, planning, and UI — is implemented in **~720 lines of core logic**.

- No large frameworks  
- No decorator chains  
- No hidden “magic” behavior  

Everything is explicit, transparent, and easy to audit.

---

## 2. **Deterministic Retrieval Instead of Blind LLM Guesswork**

The system combines:

- **Exact-match deterministic search**  
- **HNSW vector similarity**  
- **Runtime knowledge-graph lookups**

This hybrid retrieval ensures responses remain grounded, stable, and reproducible — even when running on small LLMs like **Gemma 3:1B**.

---

## 3. **True Long-Term Memory with Structure, Not Strings**

Memory isn’t just text blobs.  
Each stored item can include:

- JSON metadata  
- Extracted knowledge-graph triples  
- Importance scores  
- Timestamps  
- Memory types  

Older memories can be summarized or decayed cleanly, enabling multi-session continuity without unbounded growth.

---

## 4. **Knowledge Graph Extraction and Querying Built In**

The system automatically extracts semantic triples via the LLM and stores them in DuckDB.  
Retrieval then uses these triples to answer fact-based queries with deterministic correctness.

This bridges the gap between **vector RAG** and **symbolic reasoning** — in only a few dozen lines of code.

---

## 5. **A Real Planner, Not a Prompt Hack**

The planner:

- Decomposes goals into steps  
- Executes each step with contextual awareness  
- Maintains a shared workspace  
- Updates a persistent `plans` table  
- Stores final summaries  

Most lightweight agents skip structured reasoning entirely.  
This architecture implements it cleanly and predictably.

---

## 6. **Seamless Persistence Without External Services**

All system state — memory, vectors, triples, plans — is stored in **DuckDB**, a lightweight embedded database.

No need for:

- Postgres  
- Milvus  
- Chroma  
- Redis  
- External vector databases  

The entire agent is self-contained.

---

## 7. **Works on Small Models by Design**

Where most frameworks assume a GPT-4 class model, this architecture explicitly compensates for small local models through:

- Strict fact-mode prompts  
- Hybrid retrieval  
- Controlled context construction  
- Grounded KG lookups  
- Rule-based intent handling  

This makes the system genuinely useful on **CPU-only machines** with compact models.

---

## 8. **Built for Single-User, Single-Instance Reliability**

Instead of over-engineering multi-user concurrency, this architecture optimizes for:

- **One instance = one user**  
- Minimal concurrency risk  
- Consistent state  
- Predictable behavior  
- Straightforward deployment  

Complexity is removed where possible instead of layered endlessly.

---

## 9. **Fully Understandable End-to-End**

Every subsystem — Memory, Retrieval, Brain, Planner, Reflection, UI — is small enough to:

- Read  
- Understand  
- Reason about  
- Modify  

…without digging through undocumented internals.

This makes the system:

- Easy to maintain  
- Easy to extend  
- Easy to reason about  
- Very difficult to “get lost in”  

The design trades maximal simplicity for maximal control.

---

# 🧠 In Short

This architecture delivers the full behavior of an intelligent agent — planning, memory, retrieval, knowledge graph reasoning, and document RAG — **without the usual complexity, dependency bloat, or hidden machinery**.

It is one of the smallest fully functional agent frameworks available, and its transparent, deterministic design makes it **practical, auditable, and extensible**.

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Geo**  
SQL Developer • Robotics Student • AI Systems Engineer-in-Training
