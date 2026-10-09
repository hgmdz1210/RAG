# Hybrid search

Hybrid search runs two retrievers over the same collection and merges their results.

A dense retriever embeds the query and every passage as vectors and ranks passages by
cosine similarity. It matches paraphrases: a question about "heart attack" can find a
passage that only says "myocardial infarction". It is weaker on rare tokens such as
product codes, gene names or error numbers, which the embedding model may never have
seen often enough to place precisely.

A sparse retriever such as BM25 ranks passages by the query terms they contain. A term
counts for more when it is rare in the collection (inverse document frequency) and when
it occurs several times in the passage, with diminishing returns. BM25 is exact on rare
tokens and blind to synonyms.

The two rankings are merged with reciprocal rank fusion. A passage at rank r in one list
receives 1 / (k + r) from that list, where k is a constant, commonly 60, and its fused
score is the sum over the lists. Because only ranks are used, cosine similarities and
BM25 scores never need to be normalised to a common scale.
