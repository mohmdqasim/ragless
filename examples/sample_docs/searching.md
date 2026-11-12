# Searching

Call `Retriever.search()` with a query string. The `mode` argument picks
which engines run: `"hybrid"` fuses every configured engine with Reciprocal
Rank Fusion, or you can request a single engine like `"bm25"` or `"fuzzy"`.

Every result keeps the source file path and the line range it came from, so
you can jump straight to the matching text.
