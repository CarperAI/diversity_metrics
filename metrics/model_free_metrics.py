import torch
from torch.nn import functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk import ngrams
import gzip

def self_bleu(sentences):
    '''
    Calculates the Self-BLEU score for a collection of generated examples (https://arxiv.org/abs/1802.01886)
    :param sentences: List of generated examples
    :return:
    '''

    scores = []
    for i, hypothesis in enumerate(sentences):
        hypothesis_split = hypothesis.strip().split()

        references = [sentences[j].strip().split() for j in range(len(sentences)) if i != j]

        scores.append(sentence_bleu(references, hypothesis_split))

    return sum(scores) / len(scores)

def pairwise_ngram(n, x, y):
    '''
    Jaccard similarity using ngrams

    # common ngrams / # unique ngrams
    :param x:
    :param y:
    :param n:
    :return:
    '''
    x_ngrams = set(list(ngrams(x.lower().split(), n)))
    y_ngrams = set(list(ngrams(y.lower().split(), n)))
    intersect = x_ngrams.intersection(y_ngrams)
    union = x_ngrams.union(y_ngrams)

    if len(union) == 0:
        return 0

    else:
        return len(intersect) / len(union)

def ncd(x, y):
    '''
    Calculates the normalized compression distance using gzip
    https://arxiv.org/abs/cs/0312044
    https://ieeexplore.ieee.org/abstract/document/1362909
    :param x:
    :param y:
    :return:
    '''
    x_len = len(gzip.compress(x.encode()))
    y_len = len(gzip.compress(y.encode()))
    joint_len = len(gzip.compress("{} {}".format(x, y).encode()))

    min_len = min(x_len, y_len)
    max_len = max(x_len, y_len)

    return (joint_len - min_len) / max_len

def get_pairwise_ncd(generations):
    '''

    :param generations:
    :return:
    '''
    ncds = []
    for i, s in enumerate(generations):

        vals = [ncd(s,generations[j]) for j in range(len(generations)) if i != j]
        ncds.append(sum(vals) / len(vals))
    return ncds

def get_avg_cosine_sim(model, generations):
    '''
    Average pairwise cosine similarity
    Requires SentenceEmbedder
    :param model:
    :param generations:
    :return:
    '''
    embeds = model.embed_sentences(generations)
    tense_embeds = torch.Tensor(embeds).unsqueeze(0)
    sims = F.cosine_similarity(tense_embeds[..., None, :, :], tense_embeds[..., :, None, :], dim=-1)
    lower_triangle = torch.tril(sims, diagonal=-1)
    return lower_triangle.squeeze(0).sum() / (lower_triangle != 0).sum()

def get_pairwise_cosine_sim(model, x, y):

    return get_avg_cosine_sim(model, [x,y])


def avg_compression_ratio_full(sentences):
    '''
    Calculates the average change in compression distance in a leave-one-out setting. Here, we calculate the compression
    ratio of the list of examples minus the target examples, and then calculate the compression ratio including the target
    at the end
    :param sentences:
    :return:
    '''
    leave_one_out = [([sentences[j] for j in range(len(sentences)) if j != i], sentences[i]) for i in range(len(sentences))]

    ratio_diffs = []
    for x, y in leave_one_out:

        source = " ".join(x)
        initial_ratio = len(gzip.compress(source.encode())) / len(source.encode())

        target = " ".join(x + [y])
        final_ratio = len(gzip.compress(target.encode())) / len(target.encode())

        ratio_diffs.append(initial_ratio - final_ratio)

    return sum(ratio_diffs) / len(ratio_diffs)

def avg_compression_ratio_target(sentences):
    '''
    Calculates the average change in compression distance in a leave-one-out setting. Here we calculate the compression
    ratio of the target sentence by itself, and then calculate the compression ratio by prepending the other examples
    :param sentences:
    :return:
    '''
    leave_one_out = [([sentences[j] for j in range(len(sentences)) if j != i], sentences[i]) for i in
                     range(len(sentences))]

    ratio_diffs = []
    for x, y in leave_one_out:
        initial_ratio = len(gzip.compress(y.encode())) / len(y.encode())

        target = " ".join(x + [y])
        final_ratio = len(gzip.compress(target.encode())) / len(target.encode())

        ratio_diffs.append(initial_ratio - final_ratio)

    return sum(ratio_diffs) / len(ratio_diffs)

