# -*- coding:utf-8 -*-
import numbers
import os
import re
import shutil
import unicodedata
from contextlib import contextmanager

import numpy as np
from numpy import unicode

PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)


def make_closing(base, **attrs):
    if not hasattr(base, '__enter__'):
        attrs['__enter__'] = lambda self: self
    if not hasattr(base, '__exit__'):
        attrs['__exit__'] = lambda self, type, value, traceback: self.close()
    return type('Closing' + base.__name__, (base, object), attrs)


def smart_open(fname, mode='rb'):
    _, ext = os.path.splitext(fname)
    if ext == '.bz2':
        from bz2 import BZ2File
        return make_closing(BZ2File)(fname, mode)
    if ext == '.gz':
        from gzip import GzipFile
        return make_closing(GzipFile)(fname, mode)
    return open(fname, mode)


def get_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        '%r cannot be used to seed a np.random.RandomState instance' %
        seed)


@contextmanager
def file_or_filename(input):
    if isinstance(input, str):
        yield smart_open(input)
    else:
        input.seek(0)
        yield input


def deaccent(text):
    norm = unicodedata.normalize("NFD", text)
    result = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def copytree_hardlink(source, dest):
    copy2 = shutil.copy2
    try:
        shutil.copy2 = os.link
        shutil.copytree(source, dest)
    finally:
        shutil.copy2 = copy2


def tokenize(
        text,
        lowercase=False,
        deacc=False,
        encoding='utf8',
        errors="strict",
        to_lower=False,
        lower=False):
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)


def simple_tokenize(text):
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    tokens = [
        token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


'''转换成utf8编码'''


def any2unicode(text, encoding='utf8', errors='strict'):
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


to_unicode = any2unicode


def call_on_class_only(*args, **kwargs):
    raise AttributeError('This method should be called on a class object.')


def is_zhs(chars):
    '''是否全中文'''
    chars = any2unicode(chars)
    for i in chars:
        if not is_zh(i):
            return False
    return True


def is_zh(ch):
    '''是否中文'''
    x = ord(ch)
    if 0x2e80 <= x <= 0x2fef:
        return True
    elif 0x3400 <= x <= 0x4dbf:
        return True
    elif 0x4e00 <= x <= 0x9fbb:
        return True
    elif 0xf900 <= x <= 0xfad9:
        return True
    elif 0x20000 <= x <= 0x2a6df:
        return True
    else:
        return False


def is_punct(ch):
    x = ord(ch)
    if x < 127 and ascii.ispunct(x):
        return True
    elif 0x2000 <= x <= 0x206f:
        return True
    elif 0x3000 <= x <= 0x303f:
        return True
    elif 0xff00 <= x <= 0xffef:
        return True
    elif 0xfe30 <= x <= 0xfe4f:
        return True
    else:
        return False


def create_dir(target, remove=False):
    if os.path.exists(target):
        if remove:
            shutil.rmtree(target)
    os.makedirs(target)
