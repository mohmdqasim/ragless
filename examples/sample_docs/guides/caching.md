# Caching embeddings

When semantic search is enabled with a `cache_dir`, embeddings are written to
disk as plain `.npy` (the vectors) and `.json` (metadata: embedder name,
dimension, the source file's mtime/size, and the exact chunk ids they cover)
sidecar files. There is no database or manifest to keep in sync.

On the next run, each source file's signature is checked against its cached
entry. Only files that changed are re-embedded; everything else is loaded
straight from disk.
