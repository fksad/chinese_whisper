# -*- encode: utf-8 -*-
import os

import numpy as np

from app.common.tokenizer import Tokenizer
from app.common.word2vec import KeyedVectors
from app.common import project_root_path

STOPWORDS_FILE_LIST = [os.path.join(project_root_path, 'data/word2vec/news.stopwords.txt'),
                       os.path.join(project_root_path, 'app/resources/stopwords.utf8')]
W2V_MODEL_PATH = os.path.join(project_root_path, 'data/word2vec/news.w2v.bin.gz')


class SimilarityComputer:

    def __init__(self,
                 stopwords_file_path_list=None,
                 w2v_model_file_path=None):
        self._stopwords_file_path_list = stopwords_file_path_list or STOPWORDS_FILE_LIST
        self._w2v_model_file_path = w2v_model_file_path or W2V_MODEL_PATH
        self._load_stopwords()
        self._load_w2v()
        self._tokenizer = Tokenizer()

    def compare(self, s1, s2, seg=True):
        assert len(s1) > 0 and len(s2) > 0, "The length of s1 and s2 should > 0."
        if seg:
            s1, _ = self._tokenizer.word_segment(s1)
            s2, _ = self._tokenizer.word_segment(s2)
            s1 = ' '.join(s1)
            s2 = ' '.join(s2)

        similarity = self._similarity_distance(s1, s2)
        return similarity

    def _load_stopwords(self):
        self._stopwords = set()
        for stopword_file_path in self._stopwords_file_path_list:
            with open(stopword_file_path) as f:
                self._stopwords &= set([word.strip() for word in f.readlines()])

    def _load_w2v(self):
        if not os.path.exists(self._w2v_model_file_path):
            raise Exception(f"Model file {self._w2v_model_file_path} does not exist")
        self._w2v_model = KeyedVectors.load_from_binary_file(self._w2v_model_file_path)

    def _vectorize(self, sentence):
        vectors = []
        for word in sentence.split():
            if word not in self._stopwords:
                try:
                    vectors.append(self._w2v_model.word_vec(word))
                except KeyError as _:
                    pass
        return vectors

    def _jaccard_similarity(self, sentence1, sentence2, epi=.00001):
        s1_word_set = set()
        s2_word_set = set()
        for word in sentence1.split():
            if word in self._stopwords:
                continue
            s1_word_set.add(word)

        for word in sentence2.split():
            if word in self._stopwords:
                continue
            s2_word_set.add(word)
        intersection = s1_word_set & s2_word_set  # 交集
        union = s1_word_set | s2_word_set  # 并集
        similarity = len(intersection) / (len(union) + epi)
        return similarity

    def _similarity_distance(self, s1, s2):
        vector_s1 = self._vectorize(s1)
        vector_s2 = self._vectorize(s2)
        a = self._sim_molecule(vector_s1)
        b = self._sim_molecule(vector_s2)
        g = 1 / (np.linalg.norm(a - b) + 1)
        u = self._jaccard_similarity(s1, s2)
        r = g * (12 + abs(len(vector_s1) - len(vector_s2))) + u * 0.8
        r = min(r, 1.0)
        return round(r, 3)

    @staticmethod
    def _sim_molecule(x_):
        return np.sum(x_, axis=0)  # 将x 按列相加，得到长度 100 的数组


if __name__ == "__main__":
    similarity_computer = SimilarityComputer()
    sen1 = ["计算出每篇文章的关键词", "计算出每篇文章的关键词", "每篇文章的关键词", "当然你会好奇这里的IDF是什么", "我喜欢你"]
    sen2 = ["计算出每篇文章的关键词", "计算出每篇关键词", "计算出每篇文章", "IDF是什么", "手机怎么解锁？"]
    for x, y in zip(sen1, sen2):
        print(f"similarity of {x} and {y}：{similarity_computer.compare(x, y)}")
