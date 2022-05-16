#-*- coding:utf-8 -*-
import os

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import SimpleFSDirectory

from app.common import utils, project_root_path
from app.common.tokenizer import Tokenizer

tokenizer = Tokenizer()
lucene.initVM(vmargs=['-Djava.awt.headless=true'])

def _get_post_terms(sentence):
    words, tags = tokenizer.word_segment(sentence)
    if len(words) > 0:
        return " ".join(words)

def build_query_condition(words, tags):
    result = ""
    for (x, y) in zip(words, tags):
        if y.startswith("n"):
            result += "+%s " % x
        else:
            result += "%s " % x

    return result

def indexing(file_path, target, remove=True):
    if not os.path.exists(file_path):
        raise BaseException("index error: %s does not exist." % file_path)
    if remove:
        utils.create_dir(target, remove = True)
    directory = SimpleFSDirectory(Paths.get(target))
    analyzer = LimitTokenCountAnalyzer(WhitespaceAnalyzer(), 10000)
    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(directory, config)
    with utils.smart_open(file_path) as fin:
        for x in fin.readlines():
            o = x.split()
            id = o[0].strip()
            post = o[1].strip()
            terms = _get_post_terms(post)
            doc = Document()
            if terms:
                doc.add(Field("id", id, StringField.TYPE_STORED))
                doc.add(Field("post", post, TextField.TYPE_STORED))
                doc.add(Field("terms", terms, TextField.TYPE_STORED))
                writer.addDocument(doc)

    writer.commit()
    writer.close()
    return True

class LuceneSearch():
    '''
    LuceneSearch Object
    '''
    def __init__(self, index_path, analyzer = None):
        self.index_path = index_path
        if analyzer:
            self.analyzer = analyzer
        else:
            self.analyzer = WhitespaceAnalyzer()

    def index(self, data, remove = False):
        '''
        index data: [[id, text], [id1, text1], ...]
        '''
        if remove:
            utils.create_dir(self.index_path, remove = True)
        directory = SimpleFSDirectory(Paths.get(self.index_path))
        config = IndexWriterConfig(self.analyzer)
        writer = IndexWriter(directory, config)

        for o in data:
            id = o[0].strip()
            post = o[1].strip()
            terms = _get_post_terms(post)
            doc = Document()
            if terms:
                doc.add(Field("id", id, StringField.TYPE_STORED))
                doc.add(Field("post", post, TextField.TYPE_STORED))
                doc.add(Field("terms", terms, TextField.TYPE_STORED))
                writer.addDocument(doc)

        writer.commit()
        writer.close()

    def query(self, q, size = 50):
        '''
        Query by lucene sytanx
        '''
        query = QueryParser("terms", self.analyzer).parse(q)
        directory = SimpleFSDirectory(Paths.get(self.index_path))
        searcher = IndexSearcher(DirectoryReader.open(directory))
        scoreDocs = searcher.search(query, size).scoreDocs
        results = []
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            results.append(dict({
                                "post": doc.get("post"),
                                "id": doc.get("id"),
                                "score": scoreDoc.score,
                                "terms": doc.get("terms")
                                }))
        return results


if __name__ == "__main__":
    print("test_index_files")
    to_ = os.path.join(project_root_path, 'tmp', 'test_index')
    from_ = os.path.join(project_root_path, 'corpus', 'gfzq', 'gfzq.2017-08-25.visitor')
    search = LuceneSearch(index_path=to_)
    data = []
    with utils.smart_open(from_) as fin:
        for x in fin.readlines():
            o = x.split()
            id = o[0].strip()
            post = o[1].strip()
            data.append([id, post])

    search.index(data)
    matched = search.query("合规")
    for x in matched:
        print("id: %s, post: %s, score: %s" % (x['id'], x['post'], x['score']))
