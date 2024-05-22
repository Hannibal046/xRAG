from transformers import StoppingCriteria
import transformers
from typing import List
import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter

def keyword_extraction_with_tfidf(documents,topk=1):
    """
    Documents: List[String]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  
  
    vectorizer = TfidfVectorizer()  
    tfidf_matrix = vectorizer.fit_transform(documents)  
    feature_names = vectorizer.get_feature_names_out()  
    ret = []  
    for doc_index, doc in enumerate(documents):  
        doc_tfidf_scores = tfidf_matrix.toarray()[doc_index]  
        keywords_with_scores = {feature_names[col]: doc_tfidf_scores[col] for col in range(len(feature_names))}  
        top_keywords = sorted(keywords_with_scores.items(), key=lambda item: item[1], reverse=True)[:topk]  
    
        keywords = []
        for keyword,_ in top_keywords:
            keywords.append(keyword)  
        ret.append(" ".join(keywords))

    return ret


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

## copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/cb22e5028a6e40f409a539cbdd87194fd5e2570c/lm_eval/models/utils.py#L248
def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    initial_decoder_input_length: int,
    batch_size: int,
    stop_sequences: List[str] = ['\n', '.', ','],
    ) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    from rouge import Rouge
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## file-level evaluation ... ### 
def eval_recall(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens



def eval_fact_checking(outputs,answers):

    tokenizer = SimpleTokenizer()

    results = []
    acc_count = 0
    answer_lengths = []
    for output,answer in zip(outputs,answers):

        if answer == "False":
            answer = ["refutes", "no", "false"]
        if answer == "True":
            answer = ["supports", "yes", "true"]
        assert answer == ["refutes", "no", "false"] or answer == ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            acc_count += 1
            results.append(1.0)
        else:
            results.append(0.0)
        
        answer_lengths.append(len(output.split()))

    acc = round(sum(results)/len(results),4)
    return acc,results


def eval_truthfulqa(outputs,answers):

    f1_scores = []
    rl_scores = []
    for output,answer in zip(outputs,answers):

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)

    return F1, RL, f1_scores,rl_scores

def get_exact_match_score(outputs,answers):
    import numpy as np
    assert len(outputs) == len(answers)
    if not isinstance(answers[0],list):
        answers = [[x] for x in answers]
    exact_match_scores = []
    answer_lengths = []
    for output,answer in zip(outputs,answers):
        if ems(output, answer): # EM evaluation
            exact_match_scores.append(1.0)
        else:
            exact_match_scores.append(0.0)
        
        answer_lengths.append(len(output.split()))

    em = round(sum(exact_match_scores)/len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em,exact_match_scores


def get_substring_match_score(outputs,answers):
    """
    outputs: [string1,string2]
    answers: [
                [string1_1,string1_2],
                [string2_1,string2_2]
             ]
    """
    import numpy as np
    assert len(outputs) == len(answers)
    if not isinstance(answers[0],list):
        answers = [[x] for x in answers]
    substring_match_scores = []
    answer_lengths = []
    for output,answer in zip(outputs,answers):
        if has_answer(answer,output): # EM evaluation
            substring_match_scores.append(1.0)
        else:
            substring_match_scores.append(0.0)
        
        answer_lengths.append(len(output.split()))

    substring_match = round(sum(substring_match_scores)/len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return substring_match,substring_match_scores


def eval_multiple_choice(generated_answers,answers):
    ret = []
    assert len(generated_answers) == len(answers)
    for g_answer,answer in zip(generated_answers,answers):
        ret.append(float(g_answer==answer))
    return round(sum(ret)/len(ret),3),ret


def get_unigram_f1(text: str, answers: list[str]) -> float:
    """Calculate unigram f1 score between the text and reference answers."""
    def _get_unigram_f1(text,answers):
        if isinstance(answers,str):
            answers = [answers]
        norm_pred = normalize_answer(text)
        norm_answers = [normalize_answer(ans) for ans in answers]
        common_tokens = [
            Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
        ]
        num_same = [sum(common.values()) for common in common_tokens]

        score_list = []
        for i, num in enumerate(num_same):
            if num == 0:
                score_list.append(0.0)
            else:
                p = 1.0 * num / len(norm_pred)
                r = 1.0 * num / len(norm_answers[i])
                f1 = 2 * p * r / (p + r)
                score_list.append(f1)
        return max(score_list)
    unigram_f1 = [_get_unigram_f1(t,a) for t,a in zip(text,answers)]
    
    return sum(unigram_f1)/len(unigram_f1),unigram_f1 
