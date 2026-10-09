Dataset `scifact`, test split: 300 queries over 5,183 documents (8,645 chunks). Means with 95% bootstrap confidence intervals.

| Configuration | ndcg@10 | recall@10 | mrr@10 | hit@1 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| dense | 0.712 (0.667–0.753) | 0.825 (0.782–0.865) | 0.684 (0.635–0.728) | 0.610 (0.550–0.663) | 13.2 | 18.2 |
| bm25 | 0.679 (0.633–0.722) | 0.804 (0.757–0.846) | 0.646 (0.598–0.692) | 0.557 (0.500–0.610) | 2.9 | 3.6 |
| hybrid | 0.730 (0.688–0.771) | 0.852 (0.810–0.887) | 0.698 (0.651–0.742) | 0.607 (0.547–0.660) | 15.0 | 22.4 |
| hybrid+rerank | 0.710 (0.666–0.749) | 0.833 (0.792–0.872) | 0.680 (0.633–0.724) | 0.593 (0.537–0.647) | 2241.3 | 2772.5 |

Paired differences in ndcg@10, computed query by query:

| Comparison | Δ ndcg@10 | 95% CI |
|---|---|---|
| bm25 vs dense | -0.033 | -0.068 to -0.000 |
| hybrid vs dense | +0.018 | -0.000 to +0.037 |
| hybrid+rerank vs dense | -0.002 | -0.032 to +0.028 |
| hybrid+rerank vs hybrid | -0.021 | -0.047 to +0.007 |
