"""QA 模块配置管理。读取 [QA] 配置段并与内置默认值合并。"""

DEFAULTS = {
    "admin_ids": [],
    "gh_proxy": [],
    "openai": {
        "api_url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": "",
        "model": "Qwen/Qwen2.5-72B-Instruct",
    },
}

CONFIG_SECTION = "QA"


def load_config(sdk):
    """读取 [QA] 配置，与默认值合并后返回。"""
    raw = sdk.config.getConfig(CONFIG_SECTION) or {}
    merged = dict(DEFAULTS)
    merged.update(raw)

    if isinstance(merged.get("admin_ids"), str):
        merged["admin_ids"] = [
            a.strip() for a in merged["admin_ids"].split(",") if a.strip()
        ]

    proxy_raw = merged.get("gh_proxy")
    if isinstance(proxy_raw, list):
        proxies = [str(p).strip() for p in proxy_raw if str(p).strip()]
    else:
        proxy_str = (str(proxy_raw) if proxy_raw else "").strip()
        proxies = [p.strip() for p in proxy_str.split(",") if p.strip()]
    for i, p in enumerate(proxies):
        if not p.endswith("/"):
            proxies[i] = p + "/"
    merged["gh_proxy"] = proxies

    return merged
