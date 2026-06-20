"""ErisPulse 官方文档问答模块（Agent 模式）。

小eris —— EP 官方看板娘，基于 Agent 工具调用回答 ErisPulse 开发问题。
支持命令 / @机器人 / 私聊 / 回复上下文四种触发方式。
"""

from __future__ import annotations

import re
import time

from ErisPulse import sdk
from ErisPulse.Core.Bases import BaseModule
from ErisPulse.Core.Event import command, message

from .config import load_config
from .docs_loader import DocsLoader
from .knowledge_base import KnowledgeBase
from .llm import LLMClient
from .reply import ProgressiveReply


class Main(BaseModule):
    def __init__(self):
        self.sdk = sdk
        self.logger = sdk.logger.get_child("QA")
        self.config = load_config(sdk)

        self.docs_loader = DocsLoader(sdk, self.config)
        self.kb = KnowledgeBase(sdk, self.config)
        self.llm = LLMClient(sdk, self.config, self._build_tool_executor())

        self._admin_ids = set(self.config.get("admin_ids", []) or [])
        self._enable_at_trigger = bool(self.config.get("enable_at_trigger", True))
        self._enable_private_trigger = bool(
            self.config.get("enable_private_trigger", True)
        )
        self._enable_reply_context = bool(self.config.get("enable_reply_context", True))
        self._context_ttl = int(self.config.get("context_ttl", 1800) or 1800)
        self._building = False
        self._contexts: dict[str, dict] = {}

    def _build_tool_executor(self):
        """把知识库方法包装成 LLM 可调用的工具（参数为 dict）。"""
        kb = self.kb
        top_k_default = int(self.config.get("top_k", 5) or 5)

        def search_docs(args: dict) -> str:
            query = (args.get("query") or "").strip()
            if not query:
                return "query 不能为空。"
            top_k = args.get("top_k") or top_k_default
            try:
                top_k = int(top_k)
            except Exception:
                top_k = top_k_default
            return kb.search_docs(query, top_k)

        def read_document(args: dict) -> str:
            doc_path = (args.get("doc_path") or "").strip()
            if not doc_path:
                return "doc_path 不能为空。"
            return kb.read_document(doc_path)

        def list_documents(args: dict) -> str:
            return kb.list_documents()

        def list_source_files(args: dict) -> str:
            """列出所有可用的 ErisPulse 源码文件。"""
            return kb.list_source_files()

        def read_source_file(args: dict) -> str:
            """读取指定的 ErisPulse 源码文件内容。"""
            file_path = (args.get("file_path") or "").strip()
            if not file_path:
                return "file_path 不能为空。"
            return kb.read_source_file(file_path)

        return {
            "search_docs": search_docs,
            "read_document": read_document,
            "list_documents": list_documents,
            "list_source_files": list_source_files,
            "read_source_file": read_source_file,
        }

    @staticmethod
    def get_load_strategy():
        from ErisPulse.loaders import ModuleLoadStrategy

        return ModuleLoadStrategy(
            lazy_load=False,
            priority=0,
            depends=[],
        )

    async def on_load(self, event):
        loaded = self.kb.load_from_disk()
        if loaded:
            self.logger.info("问答知识库已就绪（来自本地缓存）")
        else:
            self.logger.warning("暂无问答知识库缓存，请管理员执行 /更新文档缓存")

        @command(
            ["问答", "qa"],
            aliases=["ask"],
            help="基于官方文档回答你的 ErisPulse 问题。用法: /问答 <问题>",
        )
        async def ask_handler(event):
            question = self._get_question(event)
            if not question:
                await event.reply(
                    "请输入你的问题，例如：\n/问答 如何创建一个模块？\n/qa 怎么监听群消息？"
                )
                return
            await self._handle_question(event, question)

        @command(
            ["更新文档缓存", "update-docs"],
            aliases=["更新知识库"],
            permission=self._is_admin,
            help="管理员：重新拉取官方文档并重建问答知识库",
        )
        async def update_handler(event):
            await self._handle_update(event)

        @command(
            ["qa状态"],
            aliases=["qa-status"],
            help="查看问答知识库状态",
        )
        async def status_handler(event):
            await self._handle_status(event)

        if self._enable_at_trigger:

            @message.on_at_message()
            async def at_handler(event):
                if not self._is_bot_mentioned(event):
                    return
                question = self._get_at_question(event)
                if not question:
                    return
                await self._handle_question(event, question)

        if self._enable_private_trigger:

            @message.on_private_message()
            async def private_handler(event):
                if self._is_command(event):
                    return
                question = self._get_text(event)
                if not question:
                    return
                history = None
                if self._enable_reply_context:
                    reply_target = self._get_reply_target(event)
                    if reply_target:
                        history = self._get_context(reply_target)
                        self.logger.debug(
                            f"私聊回复检测: target={reply_target}, "
                            f"命中上下文={history is not None}"
                        )
                await self._handle_question(event, question, history)

        if self._enable_reply_context:

            @message.on_message()
            async def reply_context_handler(event):
                if self._is_command(event):
                    return
                try:
                    if event.is_private_message():
                        return
                except Exception:
                    pass
                reply_target = self._get_reply_target(event)
                if not reply_target:
                    return
                ctx = self._get_context(reply_target)
                if ctx is None:
                    return
                self.logger.debug(
                    f"群聊回复检测: target={reply_target}, 命中上下文=True"
                )
                question = self._get_text(event)
                if not question:
                    return
                await self._handle_question(event, question, ctx)

        self.logger.info("QA 模块已加载")

    async def on_unload(self, event):
        self.logger.info("QA 模块已卸载")

    def _is_admin(self, event) -> bool:
        if not self._admin_ids:
            return False
        uid = None
        try:
            uid = event.get_user_id()
        except Exception:
            pass
        return bool(uid and uid in self._admin_ids)

    @staticmethod
    def _is_command(event) -> bool:
        try:
            return bool(event.is_command())
        except Exception:
            return False

    @staticmethod
    def _is_bot_mentioned(event) -> bool:
        """检查机器人是否被 @（mentions 中包含 self_user_id）。"""
        try:
            mentions = event.get_mentions() or []
            self_id = event.get_self_user_id()
            if not self_id:
                return False
            return self_id in mentions
        except Exception:
            return False

    @staticmethod
    def _get_question(event) -> str:
        """从命令事件中取出问题文本（兼容 get_command_args 与 get_text）。"""
        question = ""
        try:
            args = event.get_command_args()
            if args:
                question = " ".join(args).strip()
        except Exception:
            pass
        if not question:
            try:
                question = (event.get_text() or "").strip()
            except Exception:
                question = ""
        return question

    @staticmethod
    def _get_text(event) -> str:
        try:
            return (event.get_text() or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _get_at_question(event) -> str:
        """从 @ 消息中取出问题文本：取纯文本，并去掉被文本化的 @ 昵称前缀。"""
        try:
            text = (event.get_text() or "").strip()
        except Exception:
            return ""
        if text.startswith("@"):
            text = re.sub(r"^@\S{1,32}\s*", "", text).strip()
        return text

    @staticmethod
    def _get_reply_target(event) -> str | None:
        """从 OneBot12 消息段中提取被回复消息的 ID（通用，适配所有平台）。"""
        try:
            segments = event.get_message() or []
            for seg in segments:
                if isinstance(seg, dict) and seg.get("type") == "reply":
                    msg_id = (seg.get("data") or {}).get("message_id")
                    if msg_id:
                        return str(msg_id)
        except Exception:
            pass
        return None

    def _record_context(
        self,
        msg_id: str | None,
        question: str,
        answer: str,
        parent_history: list | None,
    ):
        """记录一条对话上下文，供后续回复引用。"""
        if not msg_id:
            self.logger.debug("无法记录上下文：未获取到消息 ID")
            return
        history = list(parent_history or [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 20:
            history = history[-20:]
        self._contexts[msg_id] = {
            "history": history,
            "ts": time.time(),
        }
        self.logger.debug(f"已记录上下文: msg_id={msg_id}, history_len={len(history)}")
        self._cleanup_contexts()

    def _get_context(self, msg_id: str) -> list | None:
        """根据消息 ID 获取对话上下文历史。"""
        ctx = self._contexts.get(msg_id)
        if ctx is None:
            return None
        if time.time() - ctx["ts"] > self._context_ttl:
            del self._contexts[msg_id]
            return None
        return list(ctx["history"])

    def _cleanup_contexts(self):
        """清理过期的上下文记录。"""
        now = time.time()
        expired = [
            k for k, v in self._contexts.items() if now - v["ts"] > self._context_ttl
        ]
        for k in expired:
            del self._contexts[k]

    async def _handle_question(self, event, question: str, history: list | None = None):
        if not self.kb.is_ready:
            await event.reply(
                "问答知识库尚未就绪，请联系管理员执行 /更新文档缓存 后再试。"
            )
            return

        reply = ProgressiveReply(self.sdk, event)

        if reply._mode == "stream":
            await self._handle_question_stream(event, question, reply, history)
        else:
            await self._answer_progressive(event, question, reply, history)

    async def _answer_progressive(
        self, event, question: str, reply, history: list | None = None
    ):
        """非流式渐进回复：start/update/finish 三阶段。"""
        await reply.start("正在查阅官方文档并思考…")

        async def on_tool_call(desc: str):
            await reply.update(desc)

        try:
            answer = await self.llm.answer(
                question,
                self.kb.doc_index_text(),
                on_tool_call=on_tool_call,
                supports_markdown=reply.supports_markdown,
                history=history,
                version=self.kb.source_version,
                source_files_text=self.kb.source_files_text(),
            )
        except Exception as e:
            self.logger.error(f"LLM 回答失败: {e}")
            err = str(e)
            if "429" in err or "rate" in err.lower() or "速率" in err:
                await reply.finish("抱歉，大模型当前访问繁忙，请稍后再试。")
            elif "403" in err or "401" in err:
                self.logger.error(f"LLM 鉴权/额度问题: {e}")
                await reply.finish("生成回答失败，请稍后再试。")
            else:
                await reply.finish("生成回答失败，请稍后再试。")
            return

        if not answer:
            answer = "未能生成有效回答，请换个说法再试。"
        await reply.finish(answer)
        self._record_context(reply.last_msg_id, question, answer, history)

    async def _handle_question_stream(
        self, event, question: str, reply, history: list | None = None
    ):
        """云湖真流式：思考进度 + 回答逐 token 推送，用分隔线区分。"""
        opened = await reply._stream_open()
        if not opened:
            await self._answer_progressive(event, question, reply, history)
            return

        await reply.stream_write("正在查阅官方文档并思考…\n\n", newline=False)

        doc_index = self.kb.doc_index_text()
        has_thinking = False
        answer_started = False
        error_msg = None
        full_answer_parts = []

        try:
            async for evt_type, payload in self.llm.answer_stream(
                question,
                doc_index,
                supports_markdown=reply.supports_markdown,
                history=history,
                version=self.kb.source_version,
                source_files_text=self.kb.source_files_text(),
            ):
                if evt_type == "thinking":
                    await reply.stream_write(f"- {payload}", newline=True)
                    has_thinking = True
                elif evt_type == "answer_chunk":
                    if not answer_started:
                        if has_thinking:
                            await reply.stream_write("\n---\n\n", newline=False)
                        answer_started = True
                    await reply.stream_write(payload, newline=False)
                    full_answer_parts.append(payload)
                elif evt_type == "done":
                    if not answer_started:
                        if has_thinking:
                            await reply.stream_write("\n---\n\n", newline=False)
                        await reply.stream_write(payload, newline=False)
                        full_answer_parts.append(payload)
                    await reply.stream_end()
                    full_answer = "".join(full_answer_parts) or payload
                    self._record_context(
                        reply.last_msg_id, question, full_answer, history
                    )
                    return
                elif evt_type == "error":
                    error_msg = str(payload)
                    break
        except Exception as e:
            self.logger.error(f"流式回答失败: {e}")
            error_msg = str(e)

        try:
            msg = error_msg or "生成回答时发生未知错误。"
            if "429" in msg or "rate" in msg.lower() or "速率" in msg:
                msg = "抱歉，大模型当前访问繁忙，请稍后再试。"
            elif "403" in msg or "401" in msg:
                self.logger.error(f"LLM 鉴权/额度问题: {error_msg}")
                msg = "生成回答失败，请稍后再试。"
            else:
                msg = "生成回答失败，请稍后再试。"
            await reply.stream_write(f"\n\n{msg}", newline=False)
            await reply.stream_end()
        except Exception as e:
            self.logger.warning(f"关闭流式消息失败: {e}")
            try:
                await reply.stream_end()
            except Exception:
                pass

    async def _handle_update(self, event):
        if self._building:
            await event.reply("知识库正在更新中，请勿重复执行。")
            return
        self._building = True
        start = time.time()
        await event.reply("开始更新文档缓存和源码，过程较久请耐心等待…")
        try:
            last_report = [time.time()]

            async def on_progress(done, total, path, ok):
                now = time.time()
                if done == total or now - last_report[0] > 8:
                    last_report[0] = now
                    pct = (done / total * 100) if total else 0
                    try:
                        await event.reply(f"下载文档进度: {done}/{total} ({pct:.0f}%)")
                    except Exception:
                        pass

            # 1. 获取版本信息
            await event.reply("正在获取 ErisPulse 版本信息…")
            version = await self.docs_loader.get_version()

            # 2. 更新文档缓存
            await event.reply("开始下载文档…")
            await self.kb.rebuild(self.docs_loader, on_progress=on_progress)

            # 3. 下载源码
            await event.reply("开始下载源码…")
            last_report_src = [time.time()]

            async def on_src_progress(done, total, path, ok):
                now = time.time()
                if done == total or now - last_report_src[0] > 5:
                    last_report_src[0] = now
                    pct = (done / total * 100) if total else 0
                    try:
                        await event.reply(f"下载源码进度: {done}/{total} ({pct:.0f}%)")
                    except Exception:
                        pass

            src_result = await self.docs_loader.download_source_code(
                self.kb.src_dir, on_progress=on_src_progress
            )

            # 4. 更新知识库的源码信息
            self.kb.set_source_info(version, src_result.get("files", []))

            info = self.kb.info()
            stats = info.get("stats", {})
            cost = time.time() - start
            await event.reply(
                f"文档缓存和源码更新完成！\n"
                f"- ErisPulse 版本: {version}\n"
                f"- 文档: {stats.get('doc_ok', 0)}/{stats.get('doc_total', 0)} 篇"
                f"（失败 {stats.get('doc_failed', 0)}）\n"
                f"- 知识块: {info.get('chunk_count', 0)} 个\n"
                f"- 源码文件: {src_result.get('ok', 0)}/{src_result.get('total', 0)} 个"
                f"（失败 {src_result.get('failed', 0)}）\n"
                f"- 检索: 本地 BM25\n"
                f"- 耗时: {cost:.1f}s"
            )
        except Exception as e:
            self.logger.error(f"更新文档缓存和源码失败: {e}")
            await event.reply(f"更新文档缓存和源码失败: {e}")
        finally:
            self._building = False

    async def _handle_status(self, event):
        info = self.kb.info()
        stats = info.get("stats", {})
        built = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["built_at"]))
            if info["built_at"]
            else "无"
        )
        status_text = "就绪" if info["ready"] else "未就绪"
        lines = [
            "ErisPulse 问答知识库状态",
            f"- 状态: {status_text}",
            f"- 文档数量: {info.get('doc_count', 0)}",
            f"- 知识块数量: {info.get('chunk_count', 0)}",
            "- 检索方式: 本地 BM25",
            f"- 加载文档: {stats.get('doc_ok', 0)}/{stats.get('doc_total', 0)} 篇",
            f"- 构建时间: {built}",
            f"- 语言: {self.config.get('language')}",
            f"- LLM 模型: {(self.config.get('openai') or {}).get('model', '')}",
            f"- @触发: {'开启' if self._enable_at_trigger else '关闭'}",
            f"- 私聊触发: {'开启' if self._enable_private_trigger else '关闭'}",
            f"- 回复上下文: {'开启' if self._enable_reply_context else '关闭'}",
            f"- 活跃上下文: {len(self._contexts)} 条",
        ]
        await event.reply("\n".join(lines))
