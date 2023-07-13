import abc
from sentence_transformers import SentenceTransformer

class Embedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, inputs):
        return NotImplementedError


class SentenceEmbedder(Embedder):

    @abc.abstractmethod
    def embed(self, sentence):
        return NotImplementedError

    @abc.abstractmethod
    def embed_sentences(self, sentences):
        return NotImplementedError

class WordEmbedder(Embedder):

    @abc.abstractmethod
    def embed(self, words):
        return NotImplementedError

class SBERTEmbedder(SentenceEmbedder):
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    def embed(self, sentence):
        return self.model.encode(sentence)

    def embed_sentences(self, sentences):

        return self.model.encode(sentences)

