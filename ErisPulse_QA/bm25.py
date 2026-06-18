"""本地检索算法：Okapi BM25 + 中英文分词器。

纯 Python 实现，不依赖任何外部服务或第三方库，构建与检索都本地完成。
中文采用「字 unigram + bigram」混合，英文/数字按词切分，适配 OneBot 文档场景。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """中英文混合分词：英文/数字按词，中文按单字 + 二元组。"""
    if not text:
        return []
    text = text.lower()
    tokens: List[str] = list(_WORD_RE.findall(text))
    cjk = _CJK_RE.findall(text)
    for ch in cjk:
        tokens.append(ch)  # 单字
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i] + cjk[i + 1])  # 二元组
    return tokens


class BM25Index:
    """Okapi BM25 倒排索引。"""

    def __init__(self, docs: List[Tuple[int, str]], k1: float = 1.5, b: float = 0.75):
        """
        :param docs: [(doc_index, text), ...] 的列表
        """
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.df: dict = defaultdict(int)  # 词项 -> 出现该词的文档数
        self.inverted: dict = defaultdict(list)  # 词项 -> [(doc_index, tf), ...]
        self.N = 0
        self.avgdl = 0.0
        self._build(docs)

    def _build(self, docs: List[Tuple[int, str]]):
        for doc_index, text in docs:
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))
            tf = Counter(tokens)
            for term, cnt in tf.items():
                self.df[term] += 1
                self.inverted[term].append((doc_index, cnt))
        self.N = len(docs)
        total_len = sum(self.doc_len)
        self.avgdl = (total_len / self.N) if self.N else 0.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        # Okapi IDF（带平滑，恒为正）
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """检索：返回 [(doc_index, score), ...]，按分数降序，仅返回 score>0 的结果。"""
        q_tokens = tokenize(query)
        if not q_tokens or self.N == 0:
            return []

        scores: defaultdict = defaultdict(float)
        for term in set(q_tokens):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            postings = self.inverted.get(term)
            if not postings:
                continue
            for doc_index, f in postings:
                dl = self.doc_len[doc_index]
                denom = f + self.k1 * (
                    1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0)
                )
                scores[doc_index] += idf * (f * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_k] if score > 0]
