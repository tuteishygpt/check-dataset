"""Model comparison utilities."""
import utils


def select_best_model_result(model_results: dict, similarity_threshold: int = 90):
    """
    Параўноўвае вынікі ўсіх мадэлей і выбірае лепшы вынік.
    
    Args:
        model_results: Слоўнік {model_name: {"hyp_text": ..., "score": ..., "norm_ref": ..., "norm_hyp": ...}}
        similarity_threshold: Парог для вызначэння карэктнасці
    
    Returns:
        Tuple (best_model_name, best_result_dict)
    """
    if not model_results:
        return None, None
    
    best_model = None
    best_result = None
    best_score = -1
    
    for model_name, result in model_results.items():
        score = result.get('score', 0)
        if score > best_score:
            best_score = score
            best_model = model_name
            best_result = result
    
    return best_model, best_result


def compare_two_models(model_results: dict, model1: str, model2: str):
    """
    Параўноўвае вынікі двух канкрэтных мадэлей.
    
    Args:
        model_results: Слоўнік з вынікамі ўсіх мадэлей
        model1: Назва першай мадэлі
        model2: Назва другой мадэлі
    
    Returns:
        Слоўнік з параўнаннем: {"model1": ..., "model2": ..., "winner": ..., "score_diff": ...}
    """
    result1 = model_results.get(model1)
    result2 = model_results.get(model2)
    
    if not result1 and not result2:
        return {"error": "Абедзве мадэлі не знойдзены"}
    if not result1:
        return {"winner": model2, "model1": None, "model2": result2, "score_diff": None}
    if not result2:
        return {"winner": model1, "model1": result1, "model2": None, "score_diff": None}
    
    score1 = result1.get('score', 0)
    score2 = result2.get('score', 0)
    score_diff = abs(score1 - score2)
    
    if score1 > score2:
        winner = model1
    elif score2 > score1:
        winner = model2
    else:
        winner = "tie"
    
    return {
        "model1": {"name": model1, **result1},
        "model2": {"name": model2, **result2},
        "winner": winner,
        "score_diff": score_diff
    }


def get_all_model_comparison(record: dict):
    """
    Атрымлівае поўнае параўнанне вынікаў усіх мадэлей для запісу.
    
    Args:
        record: Запіс з global_results
    
    Returns:
        Слоўнік з усімі параўнаннямі і лепшым вынікам
    """
    model_results = record.get('model_results', {})
    
    if not model_results:
        return {
            "models_count": 0,
            "best_model": None,
            "comparisons": [],
            "all_scores": {}
        }
    
    best_model, best_result = select_best_model_result(model_results)
    
    # Стварыць усе парныя параўнанні
    models = list(model_results.keys())
    comparisons = []
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            comparison = compare_two_models(model_results, models[i], models[j])
            comparisons.append(comparison)
    
    # Сабраць усе скоры
    all_scores = {model: result.get('score', 0) for model, result in model_results.items()}
    
    return {
        "models_count": len(models),
        "best_model": best_model,
        "best_result": best_result,
        "comparisons": comparisons,
        "all_scores": all_scores
    }


def find_best_model_pair(record: dict, ref_text: str):
    """
    Знаходзіць пару крыніц з найлепшым супадзеннем ПАМІЖ САБОЙ.
    
    Args:
        record: Запіс з global_results
        ref_text: Арыгінальны тэкст для параўнання
    
    Returns:
        Слоўнік з інфармацыяй пра лепшую пару
    """
    model_results = record.get('model_results', {})
    
    if not model_results or len(model_results) < 2:
        return None
    
    # Параўнаць усе пары крыніц паміж сабой і знайсці найлепшае супадзенне
    sources = list(model_results.items())
    best_pair = None
    best_pair_similarity = -1
    
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            m1_name, m1_result = sources[i]
            m2_name, m2_result = sources[j]
            
            m1_hyp = m1_result.get('hyp_text', '')
            m2_hyp = m2_result.get('hyp_text', '')
            
            # Вылічыць падабенства паміж двума гіпотэзамі
            pair_similarity, _, _ = utils.calculate_similarity(m1_hyp, m2_hyp)
            
            if pair_similarity > best_pair_similarity:
                best_pair_similarity = pair_similarity
                m1_ref_score = m1_result.get('score', 0)
                m2_ref_score = m2_result.get('score', 0)
                
                # Выбраць лепшы тэкст (той, што мае вышэйшы скор з арыгіналам)
                if m1_ref_score >= m2_ref_score:
                    best_hyp = m1_hyp
                    best_model = m1_name
                    best_score = m1_ref_score
                else:
                    best_hyp = m2_hyp
                    best_model = m2_name
                    best_score = m2_ref_score
                
                best_pair = {
                    "model1": m1_name,
                    "model2": m2_name,
                    "model1_hyp": m1_hyp,
                    "model2_hyp": m2_hyp,
                    "model1_score": m1_ref_score,
                    "model2_score": m2_ref_score,
                    "pair_similarity": pair_similarity,
                    "best_hyp": best_hyp,
                    "best_model": best_model,
                    "best_score": best_score
                }
    
    return best_pair
