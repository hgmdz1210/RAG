# Evaluating retrieval

A retrieval benchmark needs a corpus, a set of queries, and relevance judgments that say
which documents answer each query. With judgments made by people, a metric measures the
retriever rather than the habits of whatever generated the questions.

Recall@k is the fraction of a query's relevant documents found in the top k results.
Reciprocal rank is one divided by the rank of the first relevant result, and its mean
over queries is MRR. nDCG@k sums a gain for each relevant result, discounted by the
logarithm of its rank, and divides by the best achievable sum, so it rewards placing
relevant documents near the top.

A mean over a few hundred queries is an estimate. Bootstrapping gives it an interval:
resample the queries with replacement many times, recompute the mean each time, and take
the central 95% of the results. To compare two systems, bootstrap the per-query
differences instead. The same queries are hard for both systems, so the paired interval
is much tighter than the two separate ones, and a gap is credible when that interval
excludes zero.
