"""Markdown 文档分块器。

将整篇 Markdown 切成便于向量检索的语义块：按标题层级切分，超长段落再做
定长滑窗切分，每个块带完整的「文档标题 / 标题路径」上下文，便于 LLM 引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    doc_path: str
    doc_title: str
    header_path: str  # 形如 "开发者指南 > 模块开发入门 > 创建模块"
    content: str
    index: int = 0  # 在该文档中的序号

    def as_dict(self) -> dict:
        return {
            "doc_path": self.doc_path,
            "doc_title": self.doc_title,
            "header_path": self.header_path,
            "content": self.content,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            doc_path=d.get("doc_path", ""),
            doc_title=d.get("doc_title", ""),
            header_path=d.get("header_path", ""),
            content=d.get("content", ""),
            index=d.get("index", 0),
        )


@dataclass
class _Section:
    """一个标题下累积的连续文本。"""

    headers: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)

    @property
    def header_path(self) -> str:
        return " > ".join(h for h in self.headers if h)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _is_heading(line: str) -> bool:
    return line.lstrip().startswith("#")


def _heading_text(line: str) -> tuple[int, str]:
    stripped = line.lstrip()
    level = 0
    for ch in stripped:
        if ch == "#":
            level += 1
        else:
            break
    text = stripped[level:].strip().lstrip("#").strip()
    return level, text


def _split_sections(markdown: str) -> List[_Section]:
    """按标题行切分 Markdown，保留标题层级上下文。"""
    sections: List[_Section] = []
    current_headers: List[str] = []
    # 初始 section（无标题的前言）
    sec = _Section(headers=list(current_headers))
    sections.append(sec)

    for line in markdown.splitlines():
        if _is_heading(line):
            level, text = _heading_text(line)
            if not text:
                continue
            # 维护标题层级栈
            current_headers = current_headers[: level - 1]
            while len(current_headers) < level - 1:
                current_headers.append("")
            current_headers = current_headers[: level - 1] + [text]
            # 开启新 section
            sec = _Section(headers=list(current_headers))
            sec.lines.append(line)
            sections.append(sec)
        else:
            sec.lines.append(line)

    # 过滤空 section
    return [s for s in sections if s.text]


def _sliding_window(text: str, size: int, overlap: int) -> List[str]:
    """按字符做定长滑窗切分，尽量在换行处断句。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    pieces: List[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        end = start + size
        if end >= len(text):
            pieces.append(text[start:].strip())
            break
        # 尽量在换行 / 句号处断开，避免切碎代码与句子
        cut = end
        for sep in ("\n\n", "\n", "。", ". ", "；", "; "):
            idx = text.rfind(sep, start, end)
            if idx > start + size // 2:
                cut = idx + len(sep)
                break
        pieces.append(text[start:cut].strip())
        start = max(cut, start + step)
    return [p for p in pieces if p]


def chunk_markdown(
    markdown: str,
    doc_path: str,
    doc_title: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Chunk]:
    """把单篇 Markdown 文档切成若干 Chunk。"""
    sections = _split_sections(markdown)
    chunks: List[Chunk] = []
    idx = 0

    for section in sections:
        body = section.text
        if not body:
            continue
        pieces = _sliding_window(body, chunk_size, chunk_overlap)
        for piece in pieces:
            chunks.append(
                Chunk(
                    doc_path=doc_path,
                    doc_title=doc_title,
                    header_path=section.header_path,
                    content=piece,
                    index=idx,
                )
            )
            idx += 1

    return chunks
