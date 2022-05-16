# -*- coding: utf-8 -*-
import os
import re
from typing import List, Tuple, Iterable

import jieba
from jieba import posseg

from app.common import project_root_path


class Tokenizer:
    RESOURCES_PATH = os.path.join(project_root_path, "app", "resources")
    COMPANY_DICT_PATH = os.path.join(RESOURCES_PATH, "vocab.company.utf8")
    SF_DICT_PATH = os.path.join(RESOURCES_PATH, "vocab.security.utf8")
    SOUGOU_DICT_PATH = os.path.join(RESOURCES_PATH, "vocab.sougou.utf8")
    STOPWORD_DICT_PATH = os.path.join(RESOURCES_PATH, "stopwords.utf8")
    NAME_LIST_PATH = os.path.join(RESOURCES_PATH, "names.utf8")
    EMOJI_LIST_PATH = os.path.join(RESOURCES_PATH, "emoji.utf8")
    PUNCT_LIST_PATH = os.path.join(RESOURCES_PATH, 'punctuation.utf8')
    SF_STOPWORDS_PATH = os.path.join(RESOURCES_PATH, "stopwords.security.utf8")

    def __init__(self):
        self._load_jieba_corpus()
        self._load_corpus()
        self._tokenizer = posseg

    def _load_jieba_corpus(self):
        jieba.load_userdict(self.COMPANY_DICT_PATH)
        jieba.load_userdict(self.SOUGOU_DICT_PATH)
        jieba.load_userdict(self.SF_DICT_PATH)
        self._jieba_stopwords = set()
        with open(self.STOPWORD_DICT_PATH, "r") as f:
            for word in f:
                word = word.strip()
                if not word.startswith("#"):
                    self._jieba_stopwords.add(word)

    def _load_corpus(self):
        with open(self.NAME_LIST_PATH) as f:
            self._person_name_list = set([word.strip() for word in f.readlines()])
        with open(self.EMOJI_LIST_PATH) as f:
            self._emoji_list = set([word.strip() for word in f.readlines()])
        with open(self.PUNCT_LIST_PATH) as f:
            self._punct_list = set([word.strip() for word in f.readlines()])
        with open(self.COMPANY_DICT_PATH) as f:
            self._company_name_list = set([word.split()[0] for word in f.readlines()])
        with open(self.SF_DICT_PATH) as f:
            self._sf_stopwords_list = set([word.strip() for word in f.readlines()])

    @staticmethod
    def cut(sent: str, cut_all: bool = False, hmm: bool = True) -> List[str]:
        words = []
        for word in jieba.cut(sent.strip(), cut_all=cut_all, HMM=hmm):
            words.append(word)
        return words

    def word_segment(self, word: str, punct: bool = False, filter_stopword: bool = True) -> Tuple[List, List]:
        token_list = self._tokenizer.cut(word, HMM=True)
        words, tags = [], []
        for token in token_list:
            add = True
            if filter_stopword:
                if token.word in self._jieba_stopwords:
                    add = False
            if punct:
                if token.flag.startswith("x"):
                    add = False
            if add:
                words.append(token.word)
                tags.append(token.flag)
        return words, tags

    def filter_name(self, word_list: List[str]) -> List[str]:
        result = []
        for o in word_list:
            if o not in self._person_name_list:
                result.append(o)
            else:
                result.append("TPERSON")
        return result

    def filter_emoji(self, content: str):
        for o in self._emoji_list:
            content = content.replace(o, "")
        return content

    @staticmethod
    def filter_number(content) -> str:
        content = re.sub(r"[0-9.]+", " TNUMBER", content)
        return content

    @staticmethod
    def filter_url(content):
        content = re.sub(r"http[s]?://[^\s]*", "TURL", content)
        return content

    @staticmethod
    def filter_date(content):
        content = re.sub(r"\d+\s*年\d+\s*月\d+\s*日", "TDATE", content)
        content = re.sub(r"\d+\s*月\d+\s*日", "TDATE", content)
        content = re.sub(r"\d+\s*月\d+\s*日", "TDATE", content)
        content = re.sub(r"\d{2,}\s*年\d+月", "TDATE", content)
        content = re.sub(r"\d{4}-\d{1,2}-\d{1,2}", "TDATE", content)
        return content

    @staticmethod
    def filter_eng_to_tag(content):
        content = re.sub(r"[A-Za-z]+", "TENGLISH", content)
        return content

    def filter_special_punct(self, content):
        for o in self._punct_list:
            content = content.replace(o, " ")
        return content

    @staticmethod
    def _is_equal_query_as_array(source: Iterable, target: Iterable):
        all_in = True
        for element in target:
            if element not in source:
                all_in = False
                break
        return all_in

    def replacement(self, word_list: List[str]):
        if self._is_equal_query_as_array(word_list, ["接单", "继续", "服务", "客户"]):
            return None
        if self._is_equal_query_as_array(word_list, ["用户", "登录"]):
            return None
        if self._is_equal_query_as_array(word_list, ["语音", "通话", "留意", "来电"]):
            return None
        if self._is_equal_query_as_array(word_list, ["客户", "成功", "签署", "协议"]):
            return None
        if self._is_equal_query_as_array(word_list, ["订单", "转让", "留意"]):
            return None
        q = [word for word in word_list if word not in self._sf_stopwords_list]
        p = ["TCOMPANY" if word in self._company_name_list else word for word in q]
        if len(p) < 3:
            q = []
        return q


if __name__ == "__main__":
    tokenizer = Tokenizer()
    w, t = tokenizer.word_segment("添加入了Attention注意力分配机制后，使得Decoder在生成新的Target Sequence时，能得到之前Encoder编码阶段每个字符的隐藏层的>信息向量Hidden State，使得生成新序列的准确度提高。")
    for x, y in zip(w, t):
        print(f"word: {x}, tag: {y}")
