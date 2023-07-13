from nltk.translate.bleu_score import sentence_bleu
import gzip

def self_bleu(sentences):
    '''
    Calculates the Self-BLEU score for a collection of generated sentences (https://arxiv.org/abs/1802.01886)
    :param sentences: List of generated sentences
    :return:
    '''

    scores = []
    for i, hypothesis in enumerate(sentences):
        hypothesis_split = hypothesis.strip().split()

        references = [sentences[j].strip().split() for j in range(len(sentences)) if i != j]

        scores.append(sentence_bleu(references, hypothesis_split))

    return sum(scores) / len(scores)

def n_gram_overlap(inputs, n):
    pass

def ncd(x, y):
    '''
    Calculates the normalized compression distnce using gzip
    https://arxiv.org/abs/cs/0312044
    
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

def avg_compression_ratio_full(sentences):
    '''

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

