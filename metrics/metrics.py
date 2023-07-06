from nltk.translate.bleu_score import sentence_bleu
import torch

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

def emb_sim(ref_model, tokenizer, s1, s2):
    pass

def embed(ref_model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors = 'pt')
    with torch.no_grad():
        outputs = ref_model(inputs**, output_hidden_states = True)

    return outputs.hidden_states[-1]

