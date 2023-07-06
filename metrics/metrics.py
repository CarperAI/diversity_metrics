from nltk.translate.bleu_score import sentence_bleu


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
