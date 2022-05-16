# -*- coding:utf-8 -*-
import os

import numpy as np
from sklearn.neighbors import KDTree

from app.common import utils, project_root_path
from app.common.similarity import SimilarityComputer
from app.common.tokenizer import Tokenizer


class VectorKDTree:
    _tokenizer = Tokenizer()
    _similarity_computer = SimilarityComputer()

    def __init__(self, text_list):
        self._text_list = text_list
        self._build_kdtree()

    def neighbours(self, v, size=10):
        [distances], [points] = self._kdt.query(np.array([v]), k=size, return_distance=True)
        assert len(distances) == len(points), "distances and points should be in same shape."
        for point, distance in zip(points, distances):
            yield self._index_list[point], self._text_list[point], distance

    def vectorize_text(self, text):
        bow = None
        tokens = None
        word_list, tag_list = self._tokenizer.word_segment(text)
        if word_list:
            tokens = [word for word, tag in zip(word_list, tag_list) if tag.startswith(("n", "v"))]
        if tokens:
            token_vector = self._similarity_computer.vectorize(" ".join(tokens))
            if token_vector:
                bow = SimilarityComputer.sim_molecule(token_vector)
        return bow, tokens

    def _build_kdtree(self):
        self._index_list = []
        self._vector_list = []
        self._token_list = []
        for id_, text in self._text_list:
            bow, tokens = self.vectorize_text(text)
            if bow is not None:
                self._index_list.append(id_)
                self._vector_list.append(bow)
                self._token_list.append(tokens)
        if len(self._vector_list) > 0:
            self._kdt = KDTree(np.array(self._vector_list, dtype=float), leaf_size=30)


if __name__ == "__main__":
    print("test_get_neighbors")
    corpus_file_path = os.path.join(project_root_path, 'corpus', 'gfzq', 'gfzq.2017-08-25.visitor.less')
    text = "这个股票你看好吗"
    data = []
    with utils.smart_open(corpus_file_path) as f:
        for text_line in f.readlines():
            text_line = text_line.decode()
            content_list = text_line.split()
            id_ = content_list[0].strip()
            post = content_list[1].strip()
            data.append([id_, post])
    kd_tree = VectorKDTree(data)
    text_vector, _ = kd_tree.vectorize_text(text)
    for x, y, z in kd_tree.neighbours(text_vector, size=10):
        print(f"id: {x}, post: {y}, distance: {z}")
