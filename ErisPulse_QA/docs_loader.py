"""官方文档加载器。

参考官网 docs-index / docs-cache 的「CDN(反代) → raw → 失败」降级策略，
从 GitHub 拉取 `docs-mapping.json`（指定语言）以及各 Markdown 正文。

网络策略（自适应）：
- 多个「源」：可配置的反代列表 + GitHub 直连，按顺序尝试；
- 自适应排序：最近成功的源优先，避免每次都先打已失效的反代；
- 熔断器：某源连续失败超过阈值后自动禁用，不再浪费时间；
- 每源重试：单源偶发失败时本地重试，应对不稳定的直连。

返回三类数据：
- doc_index：文档索引（标题/路径/分类），供 LLM 作为「工具感知」上下文
- full_docs：每篇文档的完整 Markdown，供 read_document 工具读取
- chunks：分块后的语义片段，供向量检索
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Tuple

from .chunker import Chunk, chunk_markdown

# 文档源（与官网 docs-cache 一致，一般不会变，直接硬编码）
DOCS_REPO = "ErisPulse/ErisPulse"
DOCS_BRANCH = "Develop/v2"
DOCS_META_PATH = "docs/_meta"


def _raw_base(repo: str, branch: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/"


class DocsLoader:
    def __init__(self, sdk, config: dict):
        self.sdk = sdk
        self.logger = sdk.logger.get_child("QA.docs_loader")
        self.client = sdk.client
        self.repo = DOCS_REPO
        self.branch = DOCS_BRANCH
        self.meta_path = DOCS_META_PATH
        self.language = config.get("language", "zh-CN")
        self.chunk_size = int(config.get("chunk_size", 800) or 800)
        self.chunk_overlap = int(config.get("chunk_overlap", 100) or 100)
        self.version = None  # ErisPulse版本信息

        # 构建源列表：反代（按配置顺序）+ 直连
        self._sources: List[dict] = []
        proxies = config.get("gh_proxy") or []
        if isinstance(proxies, str):
            proxies = [proxies]
        for p in proxies:
            p = (p or "").strip()
            if not p:
                continue
            if not p.endswith("/"):
                p += "/"
            self._sources.append({"name": f"proxy:{p}", "prefix": p})
        # 直连永远作为兜底
        self._sources.append({"name": "direct", "prefix": ""})

        # 源状态：熔断 + 偏好
        self._failures: Dict[str, int] = {s["name"]: 0 for s in self._sources}
        self._disabled: Dict[str, bool] = {s["name"]: False for s in self._sources}
        self._preferred: str = self._sources[0]["name"]  # 最近成功源
        self._failure_threshold = 3  # 连续失败多少次后熔断
        self._per_source_retries = 2  # 每个源本地重试次数
        self._retry_backoff = 0.8  # 重试间隔（秒）
        self._timeout = 20  # 单次请求超时（秒）

        # 并发限制，避免一次性打爆 raw / 反代
        self._sem = asyncio.Semaphore(8)

    # ------------------------------------------------------------------ #
    # 源排序与熔断
    # ------------------------------------------------------------------ #
    def _ordered_sources(self) -> List[dict]:
        """返回按「偏好优先」排序、且未被熔断的源；全部熔断则恢复全部源再试。"""
        active = [s for s in self._sources if not self._disabled[s["name"]]]
        if not active:
            # 全部熔断：重置一次，避免一次全局抖动后永久不可用
            for name in self._disabled:
                self._disabled[name] = False
            self.logger.warning("所有源均已熔断，已重置熔断状态重试")
            active = list(self._sources)
        active.sort(key=lambda s: 0 if s["name"] == self._preferred else 1)
        return active

    def _on_success(self, name: str):
        self._failures[name] = 0
        if self._preferred != name:
            self._preferred = name
            self.logger.info(f"切换到更优文档源: {name}")

    def _on_failure(self, name: str, err: str):
        self._failures[name] += 1
        if self._failures[name] >= self._failure_threshold and not self._disabled[name]:
            self._disabled[name] = True
            self.logger.warning(
                f"文档源 {name} 连续失败 {self._failures[name]} 次，已熔断（{err}）"
            )

    # ------------------------------------------------------------------ #
    # 拉取
    # ------------------------------------------------------------------ #
    async def _fetch_text(self, relative_path: str) -> str:
        raw_url = _raw_base(self.repo, self.branch) + relative_path
        last_err = "未知错误"
        for source in self._ordered_sources():
            url = source["prefix"] + raw_url
            last_err = "未知错误"  # 为每个源重置错误信息
            for attempt in range(self._per_source_retries):
                try:
                    async with self._sem:
                        resp = await self.client.get(url, timeout=self._timeout)
                    if resp.status == 200:
                        text = await resp.text()
                        self._on_success(source["name"])
                        return text
                    last_err = f"{url} -> HTTP {resp.status}"
                except Exception as e:
                    last_err = f"{url} -> {e}"
                if attempt < self._per_source_retries - 1:
                    await asyncio.sleep(self._retry_backoff)
            self._on_failure(source["name"], last_err)
        raise RuntimeError(f"拉取文档失败: {relative_path} ({last_err})")

    async def _fetch_json(self, relative_path: str) -> dict:
        import json

        text = await self._fetch_text(relative_path)
        return json.loads(text)

    def _mapping_relative_path(self) -> str:
        return f"{self.meta_path}/{self.language}/docs-mapping.json"

    def _doc_relative_path(self, doc_path: str) -> str:
        return f"docs/{self.language}/{doc_path}"

    async def load_mapping(self) -> dict:
        return await self._fetch_json(self._mapping_relative_path())

    def iter_doc_entries(self, mapping: dict):
        """遍历 mapping 中的 (category, subgroup_name, doc) 三元组。"""
        categories = mapping.get("categories", {}) or {}
        for cat_name, category in categories.items():
            for doc in category.get("documents", []) or []:
                yield cat_name, None, doc
            for sg_key, sg in (category.get("subgroups", {}) or {}).items():
                for doc in sg.get("documents", []) or []:
                    yield cat_name, sg.get("name", sg_key), doc

    async def load_all(
        self, on_progress=None
    ) -> Tuple[List[dict], Dict[str, str], List[Chunk], dict]:
        """拉取全部文档。

        返回 (doc_index, full_docs, chunks, stats)：
        - doc_index: [{path, title, category, subgroup}]
        - full_docs: {path: markdown_content}
        - chunks: 分块列表
        - stats: 统计信息
        """
        mapping = await self.load_mapping()
        entries = list(self.iter_doc_entries(mapping))

        doc_index: List[dict] = []
        full_docs: Dict[str, str] = {}
        chunks: List[Chunk] = []
        total = len(entries)
        ok = 0
        failed: List[str] = []

        for i, (cat_name, sg_name, doc) in enumerate(entries):
            doc_path = doc.get("path", "")
            doc_title = doc.get("title", doc_path)
            doc_index.append(
                {
                    "path": doc_path,
                    "title": doc_title,
                    "category": cat_name,
                    "subgroup": sg_name or "",
                }
            )
            prefix_parts = [p for p in [cat_name, sg_name] if p]
            try:
                markdown = await self._fetch_text(self._doc_relative_path(doc_path))
            except Exception as e:
                self.logger.warning(f"跳过文档 {doc_path}: {e}")
                failed.append(doc_path)
                if on_progress:
                    await self._call_progress(
                        on_progress, i + 1, total, doc_path, False
                    )
                continue

            full_docs[doc_path] = markdown
            doc_chunks = chunk_markdown(
                markdown,
                doc_path=doc_path,
                doc_title=doc_title,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            prefix = " > ".join(prefix_parts)
            for c in doc_chunks:
                if prefix and c.header_path:
                    c.header_path = f"{prefix} > {doc_title} > {c.header_path}"
                elif prefix:
                    c.header_path = f"{prefix} > {doc_title}"
                chunks.append(c)
            ok += 1
            if on_progress:
                await self._call_progress(on_progress, i + 1, total, doc_path, True)

        stats = {
            "mapping_version": mapping.get("version"),
            "total_categories": mapping.get("total_categories"),
            "doc_total": total,
            "doc_ok": ok,
            "doc_failed": len(failed),
            "failed_paths": failed,
            "chunk_count": len(chunks),
        }
        self.logger.info(
            f"文档加载完成: {ok}/{total} 篇, {len(chunks)} 个块, 失败 {len(failed)}"
        )
        return doc_index, full_docs, chunks, stats

    async def get_version(self) -> str:
        """从远程 pyproject.toml 获取 ErisPulse 版本信息。"""
        try:
            content = await self._fetch_text("pyproject.toml")
            # 简单解析版本信息
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("version ="):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    self.version = version
                    self.logger.info(f"获取到 ErisPulse 版本: {version}")
                    return version
            return "unknown"
        except Exception as e:
            self.logger.warning(f"获取版本信息失败: {e}")
            return "unknown"

    async def download_source_code(
        self, src_dir: Path, on_progress=None
    ) -> Dict[str, list]:
        """下载 ErisPulse 的 src 目录结构。

        返回: {"language": "python", "files": ["path/to/file1.py", "path/to/file2.py", ...]}
        """
        if self.version is None:
            await self.get_version()

        file_paths = []

        # 先尝试获取 src 目录的文件列表（通过 API 递归）
        try:
            # 使用 GitHub API 递归获取目录结构
            api_path = f"repos/{self.repo}/contents/src/ErisPulse"
            last_err = "未知错误"
            for source in self._ordered_sources():
                last_err = "未知错误"  # 重置错误信息
                for attempt in range(self._per_source_retries):
                    try:
                        # 递归获取所有 Python 文件
                        files = await self._collect_files_from_github_api_recursive(
                            api_path, source
                        )
                        if files:
                            file_paths.extend(files)
                            self._on_success(source["name"])
                            self.logger.info(f"通过 API 获取到 {len(files)} 个源码文件")
                            break
                        last_err = "API 返回空文件列表"
                    except Exception as e:
                        last_err = f"GitHub API 请求失败: {e}"
                    if attempt < self._per_source_retries - 1:
                        await asyncio.sleep(self._retry_backoff)
                else:
                    self._on_failure(source["name"], last_err or "未知错误")

        except Exception as e:
            self.logger.warning(f"通过 API 获取文件列表失败: {e}，使用备用方法")

        # 备用方法：手动指定可能存在的文件路径（基于实际ErisPulse结构）
        if not file_paths:
            self.logger.info("使用预定义文件列表作为备用")
            file_paths = [
                # 根目录
                "src/ErisPulse/__init__.py",
                "src/ErisPulse/__main__.py",
                "src/ErisPulse/sdk.py",
                # CLI 模块
                "src/ErisPulse/CLI/__init__.py",
                "src/ErisPulse/CLI/base.py",
                "src/ErisPulse/CLI/cli.py",
                "src/ErisPulse/CLI/console.py",
                "src/ErisPulse/CLI/registry.py",
                "src/ErisPulse/CLI/commands/__init__.py",
                "src/ErisPulse/CLI/commands/create.py",
                "src/ErisPulse/CLI/commands/init.py",
                "src/ErisPulse/CLI/commands/install.py",
                "src/ErisPulse/CLI/commands/language.py",
                "src/ErisPulse/CLI/commands/list.py",
                "src/ErisPulse/CLI/commands/list_remote.py",
                "src/ErisPulse/CLI/commands/run.py",
                "src/ErisPulse/CLI/commands/self_update.py",
                "src/ErisPulse/CLI/commands/uninstall.py",
                "src/ErisPulse/CLI/commands/upgrade.py",
                "src/ErisPulse/CLI/i18n/__init__.py",
                "src/ErisPulse/CLI/i18n/locales/__init__.py",
                "src/ErisPulse/CLI/i18n/locales/en.py",
                "src/ErisPulse/CLI/i18n/locales/ja.py",
                "src/ErisPulse/CLI/i18n/locales/ru.py",
                "src/ErisPulse/CLI/i18n/locales/zh_cn.py",
                "src/ErisPulse/CLI/i18n/locales/zh_tw.py",
                "src/ErisPulse/CLI/utils/__init__.py",
                "src/ErisPulse/CLI/utils/display.py",
                "src/ErisPulse/CLI/utils/file_watcher.py",
                "src/ErisPulse/CLI/utils/package_manager.py",
                # Core 模块
                "src/ErisPulse/Core/__init__.py",
                "src/ErisPulse/Core/adapter.py",
                "src/ErisPulse/Core/client.py",
                "src/ErisPulse/Core/config.py",
                "src/ErisPulse/Core/constants.py",
                "src/ErisPulse/Core/lifecycle.py",
                "src/ErisPulse/Core/logger.py",
                "src/ErisPulse/Core/module.py",
                "src/ErisPulse/Core/router.py",
                "src/ErisPulse/Core/storage.py",
                "src/ErisPulse/Core/Bases/__init__.py",
                "src/ErisPulse/Core/Bases/adapter.py",
                "src/ErisPulse/Core/Bases/client.py",
                "src/ErisPulse/Core/Bases/errors.py",
                "src/ErisPulse/Core/Bases/manager.py",
                "src/ErisPulse/Core/Bases/module.py",
                "src/ErisPulse/Core/Bases/router.py",
                "src/ErisPulse/Core/Bases/storage.py",
                "src/ErisPulse/Core/Bases/websocket.py",
                "src/ErisPulse/Core/Event/__init__.py",
                "src/ErisPulse/Core/Event/base.py",
                "src/ErisPulse/Core/Event/command.py",
                "src/ErisPulse/Core/Event/message.py",
                "src/ErisPulse/Core/Event/message_builder.py",
                "src/ErisPulse/Core/Event/meta.py",
                "src/ErisPulse/Core/Event/notice.py",
                "src/ErisPulse/Core/Event/request.py",
                "src/ErisPulse/Core/Event/session_type.py",
                "src/ErisPulse/Core/Event/wrapper.py",
                "src/ErisPulse/Core/i18n/__init__.py",
                "src/ErisPulse/Core/i18n/constants.py",
                "src/ErisPulse/Core/i18n/locales/__init__.py",
                "src/ErisPulse/Core/i18n/locales/en.py",
                "src/ErisPulse/Core/i18n/locales/ja.py",
                "src/ErisPulse/Core/i18n/locales/ru.py",
                "src/ErisPulse/Core/i18n/locales/zh_cn.py",
                "src/ErisPulse/Core/i18n/locales/zh_tw.py",
                # finders 模块
                "src/ErisPulse/finders/__init__.py",
                "src/ErisPulse/finders/adapter.py",
                "src/ErisPulse/finders/module.py",
                "src/ErisPulse/finders/bases/__init__.py",
                "src/ErisPulse/finders/bases/finder.py",
                # loaders 模块
                "src/ErisPulse/loaders/__init__.py",
                "src/ErisPulse/loaders/adapter.py",
                "src/ErisPulse/loaders/module.py",
                "src/ErisPulse/loaders/strategy.py",
                "src/ErisPulse/loaders/strict.py",
                "src/ErisPulse/loaders/bases/__init__.py",
                "src/ErisPulse/loaders/bases/loader.py",
                # runtime 模块
                "src/ErisPulse/runtime/__init__.py",
                "src/ErisPulse/runtime/config_schema.py",
                "src/ErisPulse/runtime/context.py",
                "src/ErisPulse/runtime/exceptions.py",
                "src/ErisPulse/runtime/frame_config.py",
                # web_status 模块
                "src/ErisPulse/web_status/__init__.py",
            ]

        # 下载文件内容
        total = len(file_paths)
        ok = 0
        failed = []
        src_dir.mkdir(parents=True, exist_ok=True)

        for i, file_path in enumerate(file_paths):
            try:
                content = await self._fetch_text(file_path)
                target_file = src_dir / file_path.replace("src/ErisPulse/", "")
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(content, encoding="utf-8")
                ok += 1
                if on_progress:
                    await self._call_progress(
                        on_progress, i + 1, total, file_path, True
                    )
            except Exception as e:
                self.logger.warning(f"下载源码文件失败 {file_path}: {e}")
                failed.append(file_path)
                if on_progress:
                    await self._call_progress(
                        on_progress, i + 1, total, file_path, False
                    )

        result = {
            "language": "python",
            "files": [f.replace("src/ErisPulse/", "") for f in file_paths],
            "total": total,
            "ok": ok,
            "failed": len(failed),
            "failed_paths": failed,
        }

        self.logger.info(f"源码下载完成: {ok}/{total} 个文件, 失败 {len(failed)}")
        return result

    async def _collect_files_from_github_api_recursive(
        self, api_path: str, source: dict
    ) -> List[str]:
        """递归收集 GitHub API 返回的目录项中的所有 Python 文件。

        Args:
            api_path: API 路径，如 "repos/ErisPulse/ErisPulse/contents/src/ErisPulse/Core"
            source: 当前使用的源信息

        Returns:
            所有 Python 文件的完整路径列表
        """
        files = []

        try:
            url = (
                source["prefix"]
                + f"https://api.github.com/{api_path}?ref={self.branch}"
            )
            async with self._sem:
                headers = {"Accept": "application/vnd.github.v3+json"}
                resp = await self.client.get(
                    url, timeout=self._timeout, headers=headers
                )

            if resp.status != 200:
                self.logger.debug(f"API 请求失败: {url} -> HTTP {resp.status}")
                return files

            data = await resp.json()
            if not isinstance(data, list):
                return files

            for item in data:
                item_type = item.get("type", "")
                item_path = item.get("path", "")
                item_name = item.get("name", "")

                if item_type == "file" and item_name.endswith(".py"):
                    files.append(item_path)
                elif item_type == "dir":
                    # 跳过不需要的目录
                    if item_name in ["__pycache__", ".pytest_cache", "tests", ".git"]:
                        continue

                    # 递归处理子目录
                    sub_path = f"repos/{self.repo}/contents/{item_path}"
                    try:
                        sub_files = await self._collect_files_from_github_api_recursive(
                            sub_path, source
                        )
                        files.extend(sub_files)
                    except Exception as e:
                        self.logger.debug(f"递归获取子目录失败 {item_path}: {e}")
                        continue

        except Exception as e:
            self.logger.debug(f"获取目录内容失败 {api_path}: {e}")

        return files

    @staticmethod
    async def _call_progress(cb, done, total, path, ok):
        try:
            res = cb(done, total, path, ok)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass
