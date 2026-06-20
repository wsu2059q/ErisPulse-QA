# ErisPulse-QA

> 内部项目，仅开源，不发布到模块商店。供 ErisPulse 官方文档问答使用。

ErisPulse 官方文档问答模块，支持文档检索和源码查看。

## 特点

- **Agent 工具循环**：LLM 先看到完整文档索引和源码文件列表，再自主调用工具收集信息
  - `search_docs(query, top_k)` —— 本地 BM25 检索官方文档片段
  - `read_document(doc_path)` —— 读取某篇文档完整内容
  - `list_documents()` —— 列出所有文档
  - `list_source_files()` —— 列出所有可用源码文件
  - `read_source_file(file_path)` —— 读取指定源码文件内容
- **版本追踪**：自动获取并显示 ErisPulse 版本信息，确保回答基于正确版本
- **源码集成**：更新知识库时自动下载源码，AI 可查看实际实现细节
- **流式回答**：云湖平台逐 token 推送，思考过程实时可见；其余平台自动降级
- **平台自适应**：自动检测平台能力（Markdown / 编辑 / 流式），选择最优回复方式
- **多轮上下文**：回复机器人的消息即可延续对话，自动构建上下文链
- **多种触发方式**：命令 / @机器人 / 私聊直接提问 / 回复机器人消息
- **零外部依赖**：检索用本地 BM25（无 numpy、无嵌入 API）

## 安装

```bash
epsdk install git+https://github.com/wsu2059q/ErisPulse-QA.git
# 或本地开发
epsdk install ./ErisPulse-QA
```

## 配置

在项目的 `config/config.toml` 中添加：

```toml
[QA]
admin_ids = [ "u_xxxxx", ]
language = "zh-CN"
gh_proxy = []

[QA.openai]
api_url = "https://api.siliconflow.cn/v1/chat/completions"
api_key = "sk-xxxxxxxx"
model = "Qwen/Qwen2.5-72B-Instruct"
```

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_at_trigger` | `true` | 群聊 @机器人 是否触发问答 |
| `enable_private_trigger` | `true` | 私聊是否直接触发问答（无需命令） |
| `enable_reply_context` | `true` | 回复机器人消息时是否延续对话上下文 |
| `context_ttl` | `1800` | 对话上下文保留时长（秒） |
| `top_k` | `5` | search_docs 返回的片段数 |
| `max_doc_chars` | `6000` | read_document 单篇文档最大字符数 |
| `chunk_size` | `800` | 文档分块大小（字符） |
| `chunk_overlap` | `100` | 分块重叠（字符） |
| `cache_dir` | `""` | 缓存目录（空则用 `~/.ErisPulse/qa-cache`） |

> **关于 `model`**：本模块依赖模型的 **function-calling** 能力。请选用支持工具调用的模型

## 使用

### 提问方式

```
/问答 如何创建一个模块？       # 命令触发
/qa 怎么监听群消息？           # 别名
@机器人 什么是懒加载？         # 群聊 @ 触发
（私聊直接发消息即可）          # 私聊自动触发
（回复机器人的回答继续追问）    # 回复上下文触发
```

### 管理员命令

```
/更新文档缓存    # 重新拉取文档和源码并重建知识库
/qa状态          # 查看知识库状态
```

> 首次使用必须先执行一次 `/更新文档缓存` 构建知识库。
> 更新过程会：1. 获取 ErisPulse 版本信息；2. 下载官方文档；3. 下载源码文件；4. 构建索引。

### 多轮对话

回复机器人的任意一条回答即可继续追问，系统会自动带上之前的对话上下文：

```
你: /问答 怎么创建模块？
小eris: 创建模块需要...
你: （回复上条消息）那怎么注册命令呢？
小eris: 注册命令可以使用...（基于上一轮上下文回答）
```

## 回复策略（自动选择）

| 平台能力 | 策略 |
|----------|------|
| 云湖（支持 Stream） | 流式发送 |
| 支持 Edit | 发送后反复编辑，实时更新进度 |
| 支持 Markdown | 输出 Markdown 格式 |
| 仅纯文本 | 自动降级为纯文本输出 |
| 其余 | 多条普通消息 |

## 缓存

构建后会在 `cache_dir`（默认 `~/.ErisPulse/qa-cache/`）生成：

```
qa-cache/
├── qa-index-zh-CN.json   # 文档索引 + 分块 + BM25 索引
├── docs/                 # 每篇文档的完整 Markdown
└── src/                  # ErisPulse 源码文件
    ├── __init__.py
    ├── Core/
    ├── finders/
    └── ...
```

## 模块结构

```
ErisPulse_QA/
├── __init__.py
├── Core.py            # 模块入口：生命周期、命令、触发、上下文管理
├── config.py          # 配置加载与默认值
├── docs_loader.py     # 从 GitHub 拉取文档和源码（反代→直连 + 熔断）
├── chunker.py         # Markdown 语义分块
├── bm25.py            # 本地 BM25 检索
├── knowledge_base.py  # 知识库：索引 + 全文 + 检索 + 源码查看
├── llm.py             # LLM Agent：function-calling + 版本信息
└── reply.py           # 渐进式回复：流式 / 编辑 / 降级
```

## License

MIT
