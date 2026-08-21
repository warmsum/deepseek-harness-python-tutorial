# 15｜网络搜索与网页抓取

> 预计时间：55 分钟 ｜ 前置：完成第 14 章 ｜ 本章调用真实 DeepSeek Web Search 与真实网络

第 14 章让智能体可以把工作拆给多个子智能体，但主智能体和子智能体都仍受模型已有知识的限制。模型的内置知识来自训练数据，无法保证软件版本、价格和新闻等信息仍然有效。遇到这类问题时，智能体需要搜索网络，找到可能相关的来源；如果摘要不足，再读取某个网页的正文。本章分别实现 `web_search` 和 `web_fetch`，并说明两者为什么应当保持独立。

DeepSeek Web Search 没有单独的 `POST /search` 接口。客户端需要向 Anthropic 兼容的 `/messages` 端点发起一次模型请求，并在请求中声明 `web_search_20250305` 服务器工具。服务器执行搜索后，以结构化内容块返回来源。因此，一次搜索也会产生模型调用的延迟和 token 用量，比普通搜索接口更重。

## 学习目标

完成本章后，你将能够：

- 区分专用搜索端点、模型内服务器工具和直接抓取网页；
- 调用 DeepSeek 的 Anthropic 兼容 `/messages` 端点完成搜索；
- 接收必填 `queries` 数组，并发搜索、去重并按排名轮询合并来源；
- 从结构化内容块中提取来源、摘录和发布日期，不把模型自由生成的文字当作搜索结果；
- 使用 `web_fetch` 获取网页，并对正文进行清理和截断。

## 15.1 原理：搜索的三种形态

联网搜索常见的实现方式有三种，各有取舍：

| 形态 | 代表 | 取舍 |
|------|------|------|
| 专用搜索端点 | Exa、Perplexity | 快、便宜，但要额外服务商与密钥 |
| 模型内服务器工具 | DeepSeek Web Search | 需要一次模型调用，但可以复用模型密钥 |
| 自己抓网页 | 通用爬虫 | 最灵活，但要处理反爬、HTML 解析 |

本章使用第二种方式，复用 `DEEPSEEK_API_KEY`，不需要新的服务密钥。搜索请求使用 Anthropic 兼容地址 `https://api.deepseek.com/anthropic/v1`，与第 01 章聊天接口的 `https://api.deepseek.com` 不同，因此不能直接复用 `DEEPSEEK_BASE_URL`。模型名为 `deepseek-v4-flash`，请求头还要包含 `anthropic-version: 2023-06-01`。

## 15.2 一条查询怎样完成一次搜索

搜索代码分成两层。`_search_one()` 每次只处理一条查询，负责调用 DeepSeek 并返回统一的 `WebSearchResult`；`search()` 接收多条查询，并负责并发调用、去重和合并结果。先看单条查询：

```python
class WebSearchClient:
    def _search_one(self, query, max_results, max_uses) -> WebSearchResult:
        response = httpx.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "authorization": f"Bearer {self.api_key}",
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Perform a web search for the query: {query}",
                    }],
                }],
                "tools": [
                    {"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": max_uses}
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        # ...解析内容块
```

它与第 01 章的聊天接口有四处协议差异：

1. 端点：/anthropic/v1/messages，Anthropic 兼容，不是 /chat/completions；
2. 认证头：按 DeepSeek 当前兼容要求同时发送 `x-api-key` 与 Bearer；
3. 工具声明：Anthropic 服务器工具格式，type 为 `web_search_20250305`，`max_uses` 是服务器最多搜几次的上限，默认 5；
4. 结果数量：DeepSeek 没有 `max_results` 请求参数，客户端需要在 URL 去重后自行截断，并用 `truncated` 告诉调用方还有来源未返回。

## 15.3 工具层：`queries` 数组与轮询合并

`web_search` 接收必填的 `queries` 数组，其中可以包含 1 到 4 条查询；只搜索一次时也要使用单元素数组。程序会先检查数量，再删除完全重复的查询。因此，即使传入 5 条相同内容，也仍然超过数量上限，不能借去重绕过预算。

```python
def search(self, queries, max_results=5, max_uses=5, max_queries=4):
    if not queries:
        raise ValueError("queries 至少需要一条查询")
    if len(queries) > max_queries:
        raise ValueError(f"queries 最多只能有 {max_queries} 条查询")
    if any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("queries 中的每一项都必须是非空字符串")

    unique_queries = list(dict.fromkeys(queries))
    # 单条直接调用；多条通过 ThreadPoolExecutor 并发调用 _search_one
```

完全相同的查询只执行第一次。多条查询完成后，程序不会简单地把结果首尾相接，而是先取每条查询排名第一的结果，再取各自排名第二的结果，同时按 URL 去重，最后应用整批的 `max_results` 限制。这样，结果较多的一条查询不会挤掉其他查询。

任一查询失败时，整批搜索都会返回错误，不使用已经成功的部分结果。教学版可以取消尚未开始的线程任务，但无法中断已经发出的同步 HTTP 请求；官方实现还会通过共享取消信号通知其他请求停止。

## 15.4 只接受能够确认来源的结果

响应的 content 是块列表，两种块与本章有关：

- `web_search_tool_result` 是结构化搜索结果，其中的 `web_search_result` 条目组成来源清单。程序只从这些字段读取来源，不从模型生成的普通文本中提取 URL。
- `text` 块可能包含模型生成的回答和 `citations` 引用。普通回答不作为搜索结果返回；程序只读取与来源 URL 对应的 `cited_text`，作为该来源的摘录。

解析时采用一条严格规则：

```python
        if not found_result_block:
            raise RuntimeError(
                "[WEB_PROVIDER_ERROR] 响应中没有 web_search_tool_result 块"
            )
```

如果响应中没有搜索结果块，例如模型没有触发搜索而是直接回答，客户端就报告错误，不能把普通模型文本当作搜索结果。否则，调用方无法判断内容是否真的来自网络。一次请求中的多次搜索还可能返回同一页面，因此来源会按 URL 去重。

搜索结果还可能包含不可直接阅读的 `encrypted_content`，客户端不能把它当作摘要。可读摘录来自 `citations` 中的 `cited_text`，并按 URL 关联到相应来源；`page_age` 则转换成 `published_at`。最终的 `WebSearchResult` 只包含来源列表和是否截断，不包含模型自由生成的回答。

## 15.5 抓取网页：web_fetch

第二个工具直接对指定 URL 发起 HTTP GET 请求，再提取网页标题与正文片段：

```python
def web_fetch(url: str, timeout_seconds: float = 20.0) -> str:
    response = httpx.get(url, timeout=timeout_seconds, follow_redirects=True)
    response.raise_for_status()
    html = response.text
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "(无标题)"
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                     flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return f"标题: {title}\n\n正文片段: {text[:800]}"
```

教学版使用正则表达式提取标题、移除 HTML 标签，并截取正文前 800 个字符。它只用于演示“获取、清理、截断”这条流程，不适合直接作为生产环境中的网页抓取器。

教学版没有检查目标是否为内网地址，也允许请求跳转到其他域名。因此，不要用它抓取不可信 URL，也不要把它部署到能够访问敏感内网的环境。官方实现还会检查 URL 协议和凭据、限制响应大小、拒绝二进制内容，并处理取消信号，但同样没有完整的 SSRF 私网防护。

## 15.6 运行完整示例

```bash
uv run python chapters/15-external-capabilities/src/demo.py
```

真实输出，搜索结果随时间变化，结构稳定：

```
=== ① Web Search：真实搜索 DeepSeek Harness ===
  queries: ["DeepSeek Harness 是什么？", "DeepSeek Harness 官方仓库地址"]
  来源（5 条）：
  - GitHub - deepseek-ai/DeepSeek-Harness
    https://github.com/deepseek-ai/DeepSeek-Harness
    DeepSeek Harness is an open-source agent harness…
  - DeepSeek Harness documentation
    https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/README.zh.md
  ...
  是否因 max_results 截断: True 或 False

=== ② web_fetch：真实抓取网页 ===
  标题: GitHub - deepseek-ai/deepseek-harness: DeepSeek Harness: Everything is a Plugin. · GitHub

  正文片段: GitHub - deepseek-ai/deepseek-harness: DeepSeek Harness: ...
  …（正文片段截断于 800 字符）
```

三个观察点：① 两条查询并发执行，来源按排名轮流合并并按 URL 去重；② 来源来自结构化结果块，标题、URL、可选摘录与发布时间一起返回，模型自由生成的文本没有混入结果；③ `web_fetch` 读取真实 HTML，再提取标题与正文片段。

## 本章小结

- `WebSearchClient._search_one`：调用 Anthropic 兼容的 `/messages` 端点，并解析结构化搜索结果
- `WebSearchClient.search`：校验查询数组，并发搜索，按 URL 去重，再按排名轮流合并
- `web_fetch`：真实 GET、标题与正文提取、返回长度限制
- 三种联网方式：专用搜索接口、模型内服务器工具和直接抓取网页

## 对照官方

| 官方实现 | 我们对应实现 | 说明 |
|----------|--------------|------|
| [`packages/web/tool-web/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/web/tool-web/README.zh.md) | `WebSearchClient.search` | 对齐必填 `queries`、最多 4 条、先校验后去重、并发调用、URL 去重、轮询合并与整批失败；教学版不能中断已运行的同步 HTTP 线程 |
| [`packages/web/web-search-deepseek/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/web/web-search-deepseek/README.zh.md) | `_search_one` | 与官方一样使用 Anthropic Messages 接口和服务器搜索工具，只接受结构化来源与引用，并限制结果数量和拒绝重定向 |
| [`packages/web/web-fetch-http/README.zh.md`](https://github.com/deepseek-ai/DeepSeek-Harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/web/web-fetch-http/README.zh.md) | `web_fetch` | 教学版只保留 GET、文本清理与截断；官方还有完整的传输卫生和资源上限 |
| 官方凭据扩展位置 | `load_api_key` | 官方每次搜索都会重新读取凭据和配置；教学版在 `WebSearchClient` 初始化时只读取一次，更换密钥后需要新建客户端 |

## 练习

1. 对于数学常识、软件最新版本、新闻事件和一篇已知 URL 的长文，智能体应直接回答、搜索网络还是抓取网页？请从时效性、成本、证据需求和延迟解释选择。
2. 为一个有歧义的研究问题设计 2–4 条互补查询，并说明如何合并、去重和排序来源。什么情况下应该继续扩展查询，什么情况下已有证据已经足够？
3. 搜索结果带有标题、摘要和 URL，并不意味着内容可信。设计一套来源筛选与引用规则，考虑重复转载、SEO 垃圾、发布日期缺失、网页提示注入和相互矛盾的来源。
4. 将 `web_search` 与 `web_fetch` 作为两个独立工具接入智能体，完成一个需要最新信息和原文证据的任务。最终回答应区分搜索摘要与抓取正文，保留来源，并在搜索或抓取失败时明确说明未验证的部分。
