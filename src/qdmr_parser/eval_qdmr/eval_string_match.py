import torch
from torchtext.data.metrics import bleu_score

from dataset_qdmr import remove_dataset_delimiter_from_source
from eval_qdmr.sari_hook import get_sari

import string


class StringMatch(object):

    def __init__(self):
        return

    def evaluate(self, questions, gold, predict, metrics=None, prepend_dataset_name=None):
        """
        :param predict: list
            List of lower-cased source questions
        :param gold: list
            List of cased gold decompositions
        :param predict: list
            List of lower-cased predicted decompositions
        :param metrics: list
            Subset of metrics to choose from e.g., only "bleu"
        :return: dict
            Dictionary with the exact string match & bleu_4 scores
        """
        glist = gold
        plist = predict
        count = 0
        exact_match = 0.0
        f1_score = 0.0
        bleu_score = 0.0
        sari_score = 0.0
        for p, g, q in zip(plist, glist, questions):
            p_str = format_prediction(p)
            g_str = format_prediction(g)
            q = remove_dataset_delimiter_from_source(q) if prepend_dataset_name else q
            question = _normalize_question(q)
            print("**** question: ", question) 
            print("**** g_str: ", g_str) 
            print("**** p_str: ", p_str) 
            count += 1
            exact_match += _compute_exact_match(p_str, g_str)
            f1_score += _compute_f1(p_str, g_str)
            sari_score += _compute_sari(p_str, g_str, question)
            bleu_score = torch.add(bleu_score, _compute_bleu(p_str, g_str))
        exact_match /= count
        f1_score /= count
        sari_score /= count
        bleu_score = torch.div(bleu_score, count)
        return {"sari_score": sari_score,
                "exact_match": exact_match,
                "bleu_4": bleu_score,
                "f1_score": f1_score}


def restore_oov(prediction):
    """
    Replace T5 SPM OOV character with `<`.
    Certain punctuation characters are mapped to the OOV symbol in T5's
    sentence-piece model. For Spider, this appears to only affect the `<` symbol,
    so it can be deterministically recovered by running this script.
    An alternative is to preprocess dataset to avoid OOV symbols for T5.
    """
    pred = prediction.replace(" ⁇ ", "<")
    return pred


def remove_t5_tokens(prediction):
    t5_special_tokens = ["</s>", "<pad>"]
    for tok in t5_special_tokens:
        prediction = prediction.replace(tok, "")
    return prediction.strip()


def format_prediction(prediction, no_split=None):
    pred = remove_t5_tokens(restore_oov(prediction))
    return _normalize_question(pred)


def _white_space_fix(text: str) -> str:
    return ' '.join(text.split())


def _lower(text: str) -> str:
    return text.lower()


def _normalize_question(question):
    """Lower text and remove punctuation, articles and extra whitespace."""
    model_tokens = ["<pad>", "</s>"]
    for tok in model_tokens:
        question = question.replace(tok, "").strip()
    parts = [_white_space_fix((_lower(token))) for token in question.split()]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized


def _compute_f1(predicted, gold):
    predicted_bag = set(predicted.split())
    gold_bag = set(gold.split())
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _compute_exact_match(predicted, gold):
    if predicted == gold:
        return 1.0
    return 0.0


def _compute_bleu(predicted_text, gold_text, n_gram=None):
    candidate_corpus = [predicted_text.split()]
    references_corpus = [[gold_text.split()]]
    # tensor
    return bleu_score(candidate_corpus=candidate_corpus,
                      references_corpus=references_corpus,
                      max_n=4)


def _compute_sari(predicted_text, gold_text, question):
    # evaluate using SARI
    source = question.split(" ")
    prediction = predicted_text.split(" ")
    targets = [gold_text.split(" ")]
    sari, keep, add, deletion = get_sari(source, prediction, targets)
    return sari[0]
