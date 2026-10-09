# Reranking

First-stage retrieval has to be fast, so it compares a query vector with passage vectors
that were computed in advance. The model that produced them, a bi-encoder, never sees the
query and the passage together.

A cross-encoder reads the query and one passage in a single forward pass and outputs a
relevance score. Every query token can attend to every passage token, which makes the
score more accurate, but nothing can be precomputed: scoring N passages costs N forward
passes.

Reranking combines the two. The first stage returns a few dozen candidates and the
cross-encoder rescores only those. Quality improves most where the first stage found the
right passage but ranked it too low. Reranking cannot recover a passage that the first
stage missed, so the number of candidates bounds what it can achieve.

The cost is latency. On a CPU a small cross-encoder adds tens to hundreds of milliseconds
per query, growing with the number and length of the candidates.
