<!-- Source: implementation-plan.md lines 284-430, verbatim. Do not edit. Index: 00_INDEX.md -->

## 4. Embeddings: what they are and how GridSense should use them

### 4.1 What an embedding is

An embedding model maps a piece of text to a fixed-length vector of real
numbers:

`f(text) -> [x1, x2, ..., xd]`

Here, `d` is the model's embedding dimension. The dimension is
determined by the model architecture and checkpoint; it is **not chosen
arbitrarily for each document**.

For example, a sentence embedding might have hundreds of numeric values.
Those values are learned representations, not manually assigned keyword
counts. Texts that the model has learned to represent as semantically
similar often have vectors that score highly under a similarity measure.

An embedding is not a guarantee of truth. It is a retrieval
representation that helps find candidate passages.

### 4.2 Dense embedding and similarity

For vectors `q` (query) and `d` (document), cosine similarity is:

`cos(q, d) = (q · d) / (||q|| ||d||)`

-   `q · d` is the dot product.
-   `||q||` and `||d||` are vector magnitudes.
-   Cosine similarity measures the angle between vectors.
-   For L2-normalized vectors, dot product and cosine similarity are
    equivalent.

Do not interpret cosine similarity as a probability that a passage is
relevant. A score of 0.8 does not mean "80% relevant." Thresholds must
be validated on the application's own queries.

### 4.3 Why not create embeddings by assigning token IDs directly?

Token IDs are categorical identifiers, not semantic coordinates. IDs 100
and 101 do not necessarily mean those tokens are semantically close. A
learned embedding table maps token IDs to vectors whose geometry is
optimized during model training. A sentence-embedding model then
combines contextual token representations into a useful sentence or
passage vector.

For GridSense, use a pretrained embedding model first. Training
embeddings from scratch would require a suitable objective,
positive/negative examples, training data, and evaluation; it is not
needed for the initial system.

### 4.4 Embedding model selection

Start with a small, locally runnable model and benchmark alternatives on
the actual corpus and query set.

Candidate starting point: - `sentence-transformers/all-MiniLM-L6-v2`

Compare against one or two current small embedding candidates after
checking their model cards, license, task suitability, language support,
dimension, memory needs, and local inference speed. Do not select a
model only because a leaderboard ranks it highly.

For this project, evaluation should include: - Technical language such
as "μPMU", "bus 45", "feeder", "relay", "isolation", and "SOP". - Exact
identifiers, document section numbers, and acronyms. - Paraphrased
operator questions. - Queries that combine semantic intent and exact
terminology. - Queries whose answer is not present in the knowledge
base.

The last category is important: a good system must sometimes say that no
adequate evidence was found.

**[Rev 2] Input-length limit.** `all-MiniLM-L6-v2` truncates input at
256 word pieces and was trained on sequences of at most 128. Anything
beyond the limit is silently dropped, not embedded. Before any chunk-size
experiment, check `model.max_seq_length` and log, per chunk, whether the
tokenized length exceeds it. Two valid options:

-   Keep MiniLM and use parent-child chunking (§6.3): embed child chunks
    of ≤ 200--256 tokens and return the parent for context.
-   Compare against a model with a longer input window (for example a
    512-token bge/e5 small model, or a long-context model such as
    nomic-embed-text, which also runs through Ollama). Confirm each
    window on the model card at implementation time.

Never compare chunk sizes larger than the model's input window: the
"large" setting would only measure truncation.

### 4.5 Query and document encoding

Some embedding models are trained for asymmetric retrieval: a short
query is matched against longer passages. When supported by the selected
model, use its query/document-specific encoding methods, such as
`encode_query` and `encode_document`, rather than assuming the same
encoding pathway is optimal for both.

Document embeddings are created during ingestion. Query embeddings are
created at request time using the **same compatible model version and
vector space**. Never query vectors produced by an unrelated model
against stored vectors without re-embedding the corpus.

**[Rev 2] Prefixes.** Several retrieval models expect fixed text
prefixes: e5 models use `query: ` / `passage: `, bge models use a query
instruction, nomic-embed-text uses `search_query: ` /
`search_document: `. Omitting them measurably lowers retrieval quality.
MiniLM needs none. Store the prefix convention in the collection
manifest (§7.2) and apply it in one function, so ingestion and query
time cannot drift apart.

### 4.6 Baseline embedding implementation

Illustrative baseline code:

``` python
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME, device="cpu")

passages = [
    "If a critical sensor is unavailable, follow the approved fallback procedure.",
    "The operator must verify the applicable isolation procedure before action.",
]

vectors = model.encode(
    passages,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print(vectors.shape)  # (number_of_passages, model_dimension)
```

Implementation notes: - Record the exact model name and immutable
revision where possible. - Record package versions and preprocessing
configuration. - Store the vector dimension in the collection
configuration. - Normalize vectors consistently if using dot product as
the similarity metric. - Do not mix vectors from different embedding
model versions in one logical index unless the design explicitly
supports separate named vectors. - Batch document encoding to avoid
unnecessary memory pressure. - Begin with CPU embedding inference;
benchmark GPU use before consuming VRAM needed by the STGNN. - Avoid
loading several large models on the 8 GB GPU simultaneously.

------------------------------------------------------------------------

