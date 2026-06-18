"""知识库：BM25 本地检索索引 + 文档索引 + 全文缓存，并对外暴露 Agent 工具。

不依赖任何外部嵌入服务：分块文本直接构建本地 BM25 倒排索引，检索零网络开销。

持久化结构（cache_dir）：
- qa-index-{lang}.json ：文档索引 + 分块（纯文本）+ 统计
- docs/{doc_path}      ：每篇文档的完整 Markdown，供 read_document 工具读取
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .bm25 import BM25Index
from .chunker import Chunk

CACHE_VERSION = "2.0"


def _default_cache_dir() -> Path:
    return Path.home() / ".ErisPulse" / "qa-cache"


class KnowledgeBase:
    def __init__(self, sdk, config: dict):
        self.sdk = sdk
        self.logger = sdk.logger.get_child("QA.kb")

        cache_dir = config.get("cache_dir") or ""
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir = self.cache_dir / "docs"

        lang = config.get("language", "zh-CN")
        self.cache_file = self.cache_dir / f"qa-index-{lang}.json"
        self.max_doc_chars = int(config.get("max_doc_chars", 6000) or 6000)

        # 运行时数据
        self.doc_index: List[dict] = []
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Index] = None
        self.stats: dict = {}
        self.built_at: float = 0.0

        self._lock = asyncio.Lock()
        self._ready = False

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    @property
    def is_ready(self) -> bool:
        return self._ready and self.bm25 is not None and len(self.chunks) > 0

    def info(self) -> dict:
        return {
            "ready": self.is_ready,
            "doc_count": len(self.doc_index),
            "chunk_count": len(self.chunks),
            "built_at": self.built_at,
            "stats": self.stats,
            "cache_file": str(self.cache_file),
        }

    def doc_index_text(self) -> str:
        """生成供 system prompt 使用的「文档索引」概览（按分类分组）。"""
        if not self.doc_index:
            return "（暂无文档）"
        lines = []
        current_cat = None
        for entry in self.doc_index:
            cat = entry.get("category", "")
            if cat != current_cat:
                current_cat = cat
                lines.append(f"\n## {cat}")
            subgroup = entry.get("subgroup")
            title = entry.get("title", "")
            path = entry.get("path", "")
            suffix = f"（{subgroup}）" if subgroup else ""
            lines.append(f"- {title}{suffix} → {path}")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------ #
    # BM25 索引构建
    # ------------------------------------------------------------------ #
    def _build_bm25(self):
        docs = [(i, c.content) for i, c in enumerate(self.chunks)]
        self.bm25 = BM25Index(docs)

    # ------------------------------------------------------------------ #
    # 加载 / 保存
    # ------------------------------------------------------------------ #
    def load_from_disk(self) -> bool:
        if not self.cache_file.exists():
            return False
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]
            if not chunks:
                self.logger.warning("缓存中没有分块，丢弃缓存")
                return False
            self.doc_index = data.get("doc_index", [])
            self.chunks = chunks
            self.stats = data.get("stats", {})
            self.built_at = data.get("built_at", 0.0)
            self._build_bm25()
            self._ready = True
            self.logger.info(
                f"从缓存加载知识库: {len(self.doc_index)} 篇文档, {len(chunks)} 个块"
            )
            return True
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {e}")
            return False

    def save_to_disk(self):
        try:
            data = {
                "version": CACHE_VERSION,
                "built_at": self.built_at,
                "stats": self.stats,
                "doc_index": self.doc_index,
                "chunks": [c.as_dict() for c in self.chunks],
            }
            tmp = self.cache_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.cache_file)
        except Exception as e:
            self.logger.error(f"保存索引缓存失败: {e}")

    def _write_full_docs(self, full_docs: Dict[str, str]):
        """把每篇文档的完整 Markdown 写到 cache_dir/docs/{path}。"""
        if self.docs_dir.exists():
            for f in self.docs_dir.rglob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        for path, content in full_docs.items():
            target = self.docs_dir / path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            except Exception as e:
                self.logger.warning(f"写入文档缓存失败 {path}: {e}")

    # ------------------------------------------------------------------ #
    # 构建（管理员触发 /更新文档缓存）
    # ------------------------------------------------------------------ #
    async def build(self, doc_index, full_docs, chunks, stats):
        async with self._lock:
            self._ready = False
            self.doc_index = doc_index or []
            self.chunks = chunks
            self.stats = stats or {}
            self.built_at = time.time()

            self.logger.info(f"构建本地 BM25 索引（{len(chunks)} 个块）…")
            self._build_bm25()
            self._write_full_docs(full_docs or {})
            self.save_to_disk()
            self._ready = True
            self.logger.info(
                f"知识库构建完成: {len(self.doc_index)} 篇文档, {len(chunks)} 个块"
            )

    async def rebuild(self, docs_loader, on_progress=None):
        doc_index, full_docs, chunks, stats = await docs_loader.load_all(
            on_progress=on_progress
        )
        if not chunks:
            raise RuntimeError("没有加载到任何文档块，无法构建知识库")
        await self.build(doc_index, full_docs, chunks, stats)

    # ------------------------------------------------------------------ #
    # Agent 工具
    # ------------------------------------------------------------------ #
    def list_documents(self) -> str:
        """列出所有文档（标题 + 路径 + 分类）。"""
        if not self.doc_index:
            return "当前知识库为空。"
        return self.doc_index_text()

    def search_docs(self, query: str, top_k: int = 5) -> str:
        """本地 BM25 检索：返回与 query 最相关的文档片段。"""
        if not self.is_ready:
            return "知识库尚未就绪。"
        results = self.bm25.search(query, top_k)
        if not results:
            return "未检索到相关内容。"
        parts = []
        for i, (idx, score) in enumerate(results, 1):
            chunk = self.chunks[idx]
            parts.append(
                f"[{i}] (相关度 {score:.2f}) 来源: {chunk.header_path or chunk.doc_title}\n"
                f"文档: {chunk.doc_path}\n"
                f"内容:\n{chunk.content}"
            )
        return "\n\n".join(parts)

    def read_document(self, doc_path: str) -> str:
        """读取指定文档的完整内容（过长会截断）。"""
        if not self.docs_dir.exists():
            return "文档缓存目录不存在。"
        target = self.docs_dir / doc_path
        # 防御目录穿越
        try:
            target.resolve().relative_to(self.docs_dir.resolve())
        except Exception:
            return f"非法的文档路径: {doc_path}"

        if not target.exists() or not target.is_file():
            # 模糊匹配文件名兜底
            name = Path(doc_path).name
            candidates = list(self.docs_dir.rglob(name))
            if candidates:
                target = candidates[0]
            else:
                return f"未找到文档: {doc_path}"

        try:
            content = target.read_text(encoding="utf-8")
        except Exception as e:
            return f"读取文档失败: {e}"

        if len(content) > self.max_doc_chars:
            content = (
                content[: self.max_doc_chars]
                + f"\n\n…（文档过长，已截断至 {self.max_doc_chars} 字符；"
                "如需细节请用 search_docs 工具检索具体段落）"
            )
        return content
