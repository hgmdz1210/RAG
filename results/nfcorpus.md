Dataset `nfcorpus`, test split: 323 queries over 3,633 documents (6,566 chunks). Means with 95% bootstrap confidence intervals.

| Configuration | ndcg@10 | recall@10 | mrr@10 | hit@1 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| dense | 0.348 (0.314–0.384) | 0.165 (0.141–0.192) | 0.536 (0.491–0.585) | 0.446 (0.393–0.501) | 12.6 | 33.4 |
| bm25 | 0.323 (0.291–0.359) | 0.155 (0.129–0.183) | 0.530 (0.484–0.581) | 0.455 (0.406–0.511) | 4.4 | 27.3 |
| hybrid | 0.365 (0.333–0.402) | 0.178 (0.151–0.207) | 0.573 (0.528–0.621) | 0.486 (0.433–0.542) | 13.9 | 19.2 |
| hybrid+rerank | 0.364 (0.330–0.400) | 0.172 (0.147–0.200) | 0.590 (0.544–0.638) | 0.514 (0.461–0.567) | 2140.5 | 2591.0 |

Paired differences in ndcg@10, computed query by query:

| Comparison | Δ ndcg@10 | 95% CI |
|---|---|---|
| bm25 vs dense | -0.026 | -0.046 to -0.005 |
| hybrid vs dense | +0.017 | +0.006 to +0.029 |
| hybrid+rerank vs dense | +0.015 | -0.001 to +0.031 |
| hybrid+rerank vs hybrid | -0.001 | -0.014 to +0.010 |
