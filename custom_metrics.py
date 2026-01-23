from pytrec_eval import RelevanceEvaluator

class MetricsEvaluator:
  def __init__(self, results_df):
    k_values = [1, 3, 5, 10]
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    qrels = {}
    result = {}
    for query_id, group in results_df.groupby("query-id"):
      qrels[str(query_id)] = {}
      result[str(query_id)] = {}
      for _, row in group.iterrows():
        corpus_id = row["corpus-id"]
        qrels[str(query_id)][str(corpus_id)] = int(row["human-score"])
        result[str(query_id)][str(corpus_id)] = int(row["llm-score"])
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(result)
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    
    MRR = {}
    
    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    
    for query_id, doc_scores in result.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"]/len(qrels), 5)

    self.all_metrics = ndcg | _map | recall | precision | MRR
  def __getitem__(self, key):
     return self.all_metrics[key]

