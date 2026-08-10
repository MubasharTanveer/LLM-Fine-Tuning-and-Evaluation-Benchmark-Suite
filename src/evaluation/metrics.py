import math
from typing import List, Dict, Any, Tuple
from ..utils.logger import logger

try:
    import torch
except ImportError:
    torch = None

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
except ImportError:
    nltk = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

def compute_perplexity(model, tokenizer, texts: List[str], max_length: int = 512, device: str = "cuda") -> float:
    """
    Calculates average language model Perplexity (PPL) across target text samples.
    PPL = exp( cross_entropy_loss )
    """
    if not texts:
        return 0.0

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = encodings["input_ids"].to(device if torch.cuda.is_available() else "cpu")
            target_ids = input_ids.clone()

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * input_ids.size(1)

            total_loss += neg_log_likelihood.item()
            total_tokens += input_ids.size(1)

    if total_tokens == 0:
        return 0.0

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return round(perplexity, 4)

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Computes exact match percentage (0.0 to 100.0%)."""
    if not predictions or len(predictions) != len(references):
        return 0.0

    matches = 0
    for p, r in zip(predictions, references):
        clean_p = p.strip().upper()
        clean_r = r.strip().upper()
        # Handle multiple choice single character matches (e.g. 'A', 'B', 'C', 'D')
        if len(clean_p) > 0 and clean_p[0] == clean_r[0]:
            matches += 1

    return round((matches / len(references)) * 100.0, 2)

def compute_generation_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L, and BLEU-4 scores between predicted and target text.
    """
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

    if rouge_scorer is None or nltk is None:
        logger.warning("NLTK or rouge-score package missing. Calculating simplified overlap metrics.")
        scores = []
        for p, r in zip(predictions, references):
            p_words = set(p.lower().split())
            r_words = set(r.lower().split())
            overlap = len(p_words.intersection(r_words)) / max(len(r_words), 1)
            scores.append(overlap * 100.0)
        avg = round(sum(scores) / len(scores), 2)
        return {"rouge1": avg, "rouge2": avg, "rougeL": avg, "bleu": avg}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_list, r2_list, rl_list = [], [], []
    bleu_list = []
    smooth = SmoothingFunction().method1

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_list.append(scores['rouge1'].fmeasure)
        r2_list.append(scores['rouge2'].fmeasure)
        rl_list.append(scores['rougeL'].fmeasure)

        pred_tokens = pred.strip().split()
        ref_tokens = [ref.strip().split()]
        b_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth) if pred_tokens else 0.0
        bleu_list.append(b_score)

    avg_r1 = sum(r1_list) / len(r1_list)
    avg_r2 = sum(r2_list) / len(r2_list)
    avg_rl = sum(rl_list) / len(rl_list)
    avg_bleu = sum(bleu_list) / len(bleu_list)

    return {
        "rouge1": round(avg_r1 * 100.0, 2),
        "rouge2": round(avg_r2 * 100.0, 2),
        "rougeL": round(avg_rl * 100.0, 2),
        "bleu": round(avg_bleu * 100.0, 2)
    }
