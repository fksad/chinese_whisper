# -*- coding: utf-8 -*-
import os
from typing import Dict

from numpy import zeros, dtype, float32, fromstring, ascontiguousarray, ndarray

from app.common import project_root_path
from app.common.utils import smart_open


class KeyedVectors:
    def __init__(self, vocab_size: int = 0, vector_size: int = 0):
        self._word_vector_matrix: ndarray = zeros((vocab_size, vector_size), dtype=float32)
        self._word2index: Dict = {}

    def word_vec(self, word):
        if word in self._word2index:
            result = self._word_vector_matrix[self._word2index[word]]
            result.setflags(write=False)
            return result
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    @classmethod
    def load_from_binary_file(cls, model_file_path):
        with smart_open(model_file_path) as f:
            header = f.readline()
            vocab_size, vector_size = (int(word) for word in header.split())
            keyed_vectors = cls(vocab_size, vector_size)
            binary_len = dtype(float32).itemsize * vector_size
            for _ in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = b''.join(word).decode()
                weights = fromstring(f.read(binary_len), dtype=float32)
                keyed_vectors._add_word(word, weights)
        if len(keyed_vectors._word2index) != vocab_size:
            print(f"duplicate words detected, shrinking matrix size from {vocab_size} to {len(keyed_vectors._word2index)}")
            keyed_vectors._word_vector_matrix = ascontiguousarray(keyed_vectors._word_vector_matrix[: len(keyed_vectors._word2index)])
        assert (len(keyed_vectors._word2index), vector_size) == keyed_vectors._word_vector_matrix.shape
        print(f"loaded {keyed_vectors._word_vector_matrix.shape} matrix from {model_file_path}")
        return keyed_vectors

    def _add_word(self, word, weights):
        if word in self._word2index:
            print(f"duplicate word: '{word}'")
        else:
            word_id = len(self._word2index)
            self._word2index[word] = word_id
            self._word_vector_matrix[word_id] = weights


if __name__ == '__main__':
    _fin_wv_path = os.path.join(project_root_path, 'data', 'word2vec', 'news.w2v.bin.gz')
    KeyedVectors.load_from_binary_file(_fin_wv_path)
