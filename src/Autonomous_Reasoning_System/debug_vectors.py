import duckdb
from fastembed import TextEmbedding

# Initialize
print("Loading model...")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
con = duckdb.connect("data/memory.duckdb")

# 1. Check Content
print("\n--- 1. Database Content ---")
memories = con.execute("SELECT id, text FROM memory").fetchall()
for m in memories:
    print(f"[{m[0][:4]}] {m[1]}")

# 2. Check Vector Search Scores
query = "When is Cornelia's birthday?"
print(f"\n--- 2. Vector Scores for: '{query}' ---")
query_vec = list(embedder.embed([query]))[0].tolist()

# We run the query with threshold 0.0 to see EVERYTHING
try:
    # Load VSS
    con.execute("INSTALL vss; LOAD vss;")
    results = con.execute(f"""
        SELECT m.text, (1 - list_cosine_similarity(v.embedding, ?::FLOAT[384])) as score
        FROM vectors v
        JOIN memory m ON v.id = m.id
        ORDER BY score DESC
    """, (query_vec,)).fetchall()

    for r in results:
        print(f"Score: {r[1]:.4f} | Text: {r[0]}")
except Exception as e:
    print(f"Vector search failed: {e}")