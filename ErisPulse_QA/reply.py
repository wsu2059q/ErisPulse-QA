"""渐进式回复：在支持「编辑/流式」的平台上，用单条消息实时展示进度。

策略选择（按优先级）：
1. 云湖(yunhu) → 流式消息 .Stream("markdown")，整条消息平滑更新
2. 平台支持 .Edit   → 发一条后反复编辑，实时更新进度与最终答案
3. 其余平台        → 回退为多条普通消息

Markdown：平台若支持 Markdown，优先使用并默认折叠最终回答（<details>）；
不支持则回退纯文本。所有平台调用均包裹异常兜底，失败自动降级。
"""

from __future__ import annotations

import asyncio
from typing import Optional


class ProgressiveReply:
    """统一封装「发送 → 更新进度 → 落定最终答案」的回复流程。"""

    def __init__(self, sdk, event):
        self.sdk = sdk
        self.logger = sdk.logger.get_child("QA.reply")
        self.event = event
        self.platform = ""
        try:
            self.platform = event.get_platform() or ""
        except Exception:
            self.platform = ""

        self._adapter = None
        self._send_type = ""
        self._target_id = ""
        self._bot_id = ""
        self._methods: list = []
        self._has_markdown = False
        self._mode = "multi"
        self._msg_id: Optional[str] = None
        self._last_msg_id: Optional[str] = None
        self._queue: Optional[asyncio.Queue] = None
        self._stream_task = None
        self._accumulated = ""

        self._detect()

    @property
    def last_msg_id(self) -> Optional[str]:
        """最近一次成功发送/编辑/流式落定的消息 ID。"""
        return self._last_msg_id

    def _detect(self):
        try:
            adapter_inst, send_type, target_id, bot_id = (
                self.event._get_adapter_and_target()
            )
            self._adapter = adapter_inst
            self._send_type = send_type
            self._target_id = target_id
            self._bot_id = bot_id
        except Exception as e:
            self.logger.debug(f"无法解析发送目标，使用普通回复: {e}")
            return

        try:
            self._methods = self.sdk.adapter.list_sends(self.platform) or []
        except Exception:
            self._methods = []
        self._has_markdown = any(m.lower() == "markdown" for m in self._methods)
        self.logger.debug(
            f"平台 {self.platform} 支持方法: {self._methods[:12]}{'...' if len(self._methods) > 12 else ''}"
            f", Markdown={self._has_markdown}, mode={self._mode}"
        )

        if self.platform == "yunhu" and "Stream" in self._methods:
            self._mode = "stream"
        elif "Edit" in self._methods:
            self._mode = "edit"

    @property
    def _content_type(self) -> str:
        return "markdown" if self._has_markdown else "text"

    @property
    def supports_markdown(self) -> bool:
        """当前平台是否支持 Markdown。"""
        return self._has_markdown

    def _send_chain(self):
        """构造 Send.To(...) 链（多 Bot 时带上 Using）。"""
        if self._adapter is None:
            return None
        chain = self._adapter.Send.To(self._send_type, self._target_id)
        if self._bot_id and hasattr(chain, "Using"):
            chain = chain.Using(self._bot_id)
        return chain

    async def start(self, text: str):
        """发送初始消息。"""
        if self._mode == "stream":
            await self._start_stream(text)
        elif self._mode == "edit":
            await self._send_initial_edit(text)
        else:
            await self._plain_send(text)

    async def update(self, text: str):
        """更新进度（追加展示新的一行）。text 为本次新增的内容。"""
        if self._mode == "stream":
            if self._queue is not None:
                await self._queue.put(text)
        elif self._mode == "edit":
            self._accumulated += ("\n" + text) if self._accumulated else text
            await self._edit(self._accumulated)
        else:
            await self._plain_send(text)

    async def finish(self, final_text: str):
        """落定最终答案。final_text 为完整最终回答。"""
        if self._mode == "stream":
            if self._queue is not None:
                await self._queue.put(final_text)
                await self._queue.put(None)
            if self._stream_task:
                try:
                    await self._stream_task
                except Exception as e:
                    self.logger.warning(f"流式消息异常，降级发送: {e}")
                    await self._plain_send(final_text)
        elif self._mode == "edit":
            content = (
                (self._accumulated + "\n\n" + final_text).strip()
                if self._accumulated
                else final_text
            )
            head = content[:2000]
            await self._edit(head)
            rest = content[2000:]
            for i in range(0, len(rest), 2000):
                await self._plain_send(rest[i : i + 2000])
        else:
            for i in range(0, len(final_text), 2000):
                await self._plain_send(final_text[i : i + 2000])

    async def _plain_send(self, text: str):
        result = None
        try:
            method = "Markdown" if self._has_markdown else "Text"
            result = await self.event.reply(text, method=method)
        except Exception:
            try:
                result = await self.event.reply(text)
            except Exception as e:
                self.logger.warning(f"普通回复失败: {e}")
                result = None
        if result is not None:
            extracted = self._extract_msg_id(result)
            if extracted:
                self._last_msg_id = extracted

    async def _send_initial_edit(self, text: str):
        try:
            chain = self._send_chain()
            if chain is None:
                self._mode = "multi"
                await self._plain_send(text)
                return
            if self._has_markdown and hasattr(chain, "Markdown"):
                result = await chain.Markdown(text)
            else:
                result = await chain.Text(text)
            self._msg_id = self._extract_msg_id(result)
            self._accumulated = text
            if self._msg_id:
                self._last_msg_id = self._msg_id
            else:
                self._mode = "multi"
        except Exception as e:
            self.logger.warning(f"编辑模式初始发送失败，降级为普通回复: {e}")
            self._mode = "multi"
            await self._plain_send(text)

    async def _edit(self, text: str):
        if not self._msg_id:
            await self._plain_send(text)
            return
        try:
            chain = self._send_chain()
            if chain is None:
                await self._plain_send(text)
                return
            if self._has_markdown:
                result = await chain.Edit(self._msg_id, text, content_type="markdown")
            else:
                result = await chain.Edit(self._msg_id, text)
            extracted = self._extract_msg_id(result)
            if extracted:
                self._last_msg_id = extracted
        except Exception as e:
            self.logger.warning(f"编辑消息失败，降级为普通回复: {e}")
            self._mode = "multi"
            self._msg_id = None
            await self._plain_send(text)

    async def _start_stream(self, initial: str):
        """兼容旧接口：以一段初始文本启动流。"""
        await self._stream_open()
        if initial:
            await self.stream_write(initial, newline=True)

    async def _stream_open(self) -> bool:
        """打开一条空的流式消息。成功返回 True；不支持流式则降级并返回 False。"""
        if self._mode != "stream":
            return False
        self._queue = asyncio.Queue()
        queue = self._queue

        async def content_gen():
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, (bytes, bytearray)):
                    yield bytes(item)
                else:
                    yield str(item).encode("utf-8")

        try:
            chain = self._send_chain()
            if chain is None:
                self._mode = "multi"
                return False
            stream_obj = chain.Stream(self._content_type, content_gen())
            if isinstance(stream_obj, asyncio.Task):
                self._stream_task = stream_obj
            elif asyncio.iscoroutine(stream_obj):
                self._stream_task = asyncio.create_task(stream_obj)
            else:
                self._mode = "multi"
                self._queue = None
                return False
            return True
        except Exception as e:
            self.logger.warning(f"流式消息启动失败，降级为普通回复: {e}")
            self._mode = "multi"
            self._stream_task = None
            self._queue = None
            return False

    async def stream_write(self, text: str, newline: bool = False):
        """向流式消息追加一段文本。newline=True 时在末尾补换行。"""
        if self._queue is None or not text:
            return
        data = (text + ("\n" if newline else "")).encode("utf-8")
        await self._queue.put(data)

    async def stream_end(self):
        """结束流式消息。"""
        if self._queue is not None:
            await self._queue.put(None)
        if self._stream_task:
            try:
                result = await self._stream_task
                extracted = self._extract_msg_id(result)
                if extracted:
                    self._last_msg_id = extracted
            except Exception as e:
                self.logger.warning(f"流式消息结束异常: {e}")
        self._queue = None

    @staticmethod
    def _extract_msg_id(result) -> Optional[str]:
        if not result:
            return None
        if isinstance(result, dict):
            return result.get("message_id") or (result.get("data") or {}).get(
                "message_id"
            )
        return None
