import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, ndcg_score
import pandas as pd

def nrmse(y_true, y_pred, scale=2):
    """Нормализованная RMSE"""
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    return rmse / scale if scale != 0 else rmse

def nmae(y_true, y_pred, scale=2):
    """Нормализованная MAE"""
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / scale if scale != 0 else mae

def rbo(y_true, y_pred, p=0.9):
    """Rank Biased Overlap"""
    S = y_true
    T = y_pred
    S_len = len(S)
    T_len = len(T)
    k = min(len(S), len(T))

    if not S_len and not T_len:
        return 1
    if not S_len or not T_len:
        return 0

    A, AO = [0] * k, [0] * k
    if p == 1.0:
        weights = [1.0 for _ in range(k)]
    else:
        weights = [1.0 * (1 - p) * p**d for d in range(k)]

    S_running, T_running = {S[0]: True}, {T[0]: True}
    A[0] = 1 if S[0] == T[0] else 0
    AO[0] = weights[0] if S[0] == T[0] else 0

    for d in range(1, k):
        tmp = 0
        if S[d] in T_running:
            tmp += 1
        if T[d] in S_running:
            tmp += 1
        if S[d] == T[d]:
            tmp += 1
        A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

        if p == 1.0:
            AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
        else:
            AO[d] = AO[d - 1] + weights[d] * A[d]

        S_running[S[d]] = True
        T_running[T[d]] = True
    return AO[-1] if AO[-1] < 1 else 1.0

def metric_for_all_queries(queries_data: pd.DataFrame, metric, exclude_len=0):
    """Вычисляет метрику для каждого запроса и усредняет"""
    n = 0
    result_sum = 0
    results_by_query = {}
    
    for query_id, group in queries_data.groupby("query-id"):
        if len(group) <= exclude_len:
            continue
        
        # Получаем оценки (не ранги!)
        llm_scores = group["llm-score"].to_numpy(int)  # Оценки LLM
        human_scores = group["human-score"].to_numpy(int)  # Человеческие оценки
        
        # Вычисляем метрику
        score = metric(llm_scores, human_scores, query_id)
        
        if np.isnan(score):
            score = 0
        
        results_by_query[query_id] = score
        result_sum += score
        n += 1
    
    if n == 0:
        return 0, {}
    
    return result_sum / n

def calc_ndcg(results_df, k=10):
    """Правильный расчет NDCG"""
    
    def ndcg_metric(llm_scores, human_scores, query_id=None):
        """
        llm_scores: оценки от модели (int)
        human_scores: истинные оценки (int)
        
        Важно: NDCG ожидает, что llm_scores - это скоры для сортировки,
        а human_scores - истинные релевантности
        """
        # Если все оценки одинаковые, возвращаем 1 или 0
        if len(np.unique(human_scores)) <= 1:
            return 1.0 if len(llm_scores) > 0 else 0.0
        
        # NDCG ожидает, что llm_scores будут использоваться для сортировки
        # human_scores - истинные релевантности
        try:
            return ndcg_score([human_scores], [llm_scores], k=min(k, len(llm_scores)))
        except:
            # Fallback для проблемных случаев
            return 0.0
    
    return metric_for_all_queries(results_df, ndcg_metric, exclude_len=1)

def calc_ndcg_explicit(results_df, k=10):
    """Альтернативная версия NDCG с явным ранжированием"""
    
    def ndcg_explicit_metric(llm_scores, human_scores, query_id=None):
        # Получаем ранги на основе оценок
        llm_ranks = np.argsort(-llm_scores)  # Индексы от высокого к низкому
        human_ranks = np.argsort(-human_scores)
        
        # Сортируем человеческие оценки по рангам LLM
        human_sorted_by_llm = human_scores[llm_ranks]
        
        # Сортируем человеческие оценки идеально
        human_sorted_ideal = np.sort(human_scores)[::-1]
        
        # Рассчитываем DCG и IDCG
        def dcg(scores, k):
            k = min(k, len(scores))
            return sum((2**scores[i] - 1) / np.log2(i + 2) for i in range(k))
        
        dcg_val = dcg(human_sorted_by_llm, k)
        idcg_val = dcg(human_sorted_ideal, k)
        
        return dcg_val / idcg_val if idcg_val > 0 else 0.0
    
    return metric_for_all_queries(results_df, ndcg_explicit_metric, exclude_len=1)

def calc_nrmse(results_df):
    """Нормализованная RMSE по всем оценкам"""
    llm = results_df["llm-score"].to_numpy(int)
    human = results_df["human-score"].to_numpy(int)
    
    # Масштаб для нормализации (max - min оценок)
    scale = human.max() - human.min() if len(human) > 0 else 2
    
    return nrmse(human, llm)

def calc_nmae(results_df):
    """Нормализованная MAE по всем оценкам"""
    llm = results_df["llm-score"].to_numpy(int)
    human = results_df["human-score"].to_numpy(int)
    
    # Масштаб для нормализации (max - min оценок)
    scale = human.max() - human.min() if len(human) > 0 else 2
    
    return nmae(human, llm)

def kendall_metric(llm_scores, human_scores, query_id=None):
        # Получаем ранги из оценок
        # llm_ranks = np.argsort(-llm_scores)
        # human_ranks = np.argsort(-human_scores)
        
        # Вычисляем корреляцию между рангами
        corr, _ = kendalltau(llm_scores, human_scores)
        return corr if not np.isnan(corr) else 0.0

def calc_kendalltau(results_df):
    """Коэффициент корреляции Кендалла между рангами"""

    return metric_for_all_queries(results_df, kendall_metric, exclude_len=1)

def calc_spearman(results_df):
    """Коэффициент корреляции Спирмена между рангами"""
    
    def spearman_metric(llm_scores, human_scores, query_id=None):
        llm_ranks = np.argsort(-llm_scores)
        human_ranks = np.argsort(-human_scores)
        
        corr, _ = spearmanr(llm_ranks, human_ranks)
        return corr if not np.isnan(corr) else 0.0
    
    return metric_for_all_queries(results_df, spearman_metric, exclude_len=1)

def calc_rbo(results_df, p=0.9):
    """Rank Biased Overlap между рангами"""
    
    def rbo_metric(llm_scores, human_scores, query_id=None):
        llm_ranks = np.argsort(-llm_scores)
        human_ranks = np.argsort(-human_scores)
        
        return rbo(llm_ranks, human_ranks, p=p)
    
    return metric_for_all_queries(results_df, rbo_metric, exclude_len=1)

def cohen_kappa_metric(human, llm):
        return cohen_kappa_score(human, llm, weights="quadratic")
def calc_kappa(results_df):
    llm = results_df["llm-score"].to_numpy(int)
    human = results_df["human-score"].to_numpy(int)
    return cohen_kappa_metric(human, llm)
    
    

def precision_at_k_class2(df, k=5):
    """Как часто в топ-k есть документы с human=2"""
    
    def prec_metric(llm_scores, human_scores, query_id=None):
        # Индексы документов с human=2
        human2_indices = np.where(human_scores == 2)[0]
        if len(human2_indices) == 0:
            return 0  # В этом запросе нет human=2
        
        # Сортируем по llm score (убывание)
        sorted_idx = np.argsort(-llm_scores)
        top_k = sorted_idx[:min(k, len(sorted_idx))]
        
        # Сколько human=2 в топ-k
        human2_in_topk = sum(1 for idx in top_k if human_scores[idx] == 2)
        return human2_in_topk / len(top_k)
    
    return metric_for_all_queries(df, prec_metric, exclude_len=1)

def recall_at_k_class2(df, k=5):
    """Какая доля human=2 попадает в топ-k"""
    
    def recall_metric(llm_scores, human_scores, query_id=None):
        # Индексы документов с human=2
        human2_indices = np.where(human_scores == 2)[0]
        if len(human2_indices) == 0:
            return 0  # В этом запросе нет human=2
        
        # Сортируем по llm score (убывание)
        sorted_idx = np.argsort(-llm_scores)
        top_k = sorted_idx[:min(k, len(sorted_idx))]
        
        # Сколько human=2 в топ-k
        human2_in_topk = sum(1 for idx in top_k if human_scores[idx] == 2)
        
        # RECALL: доля найденных human=2 от общего числа human=2
        return human2_in_topk / len(human2_indices) if len(human2_indices) > 0 else 0
    
    return metric_for_all_queries(df, recall_metric, exclude_len=1)