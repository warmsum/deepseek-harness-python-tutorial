"""第 15 章 demo：真实的外部能力。

运行（在项目根目录，需要 .env）：
    uv run python chapters/15-external-capabilities/src/demo.py

两节（都是真实网络调用）：
① Web Search：真实调用 DeepSeek Anthropic 兼容端点，服务器侧搜索，
   只打印结构化来源，不信任提供方生成的自由文本；
② web_fetch：校验并抓取 DeepSeek Harness 的 GitHub 页面。
"""

from __future__ import annotations

from web_tools import WebSearchClient, web_fetch


def main() -> None:
    print("=== ① Web Search：真实搜索 DeepSeek Harness ===")
    client = WebSearchClient()
    queries = ["DeepSeek Harness 是什么？", "DeepSeek Harness 官方仓库地址"]
    print(f"  queries: {queries}")
    result = client.search(queries)
    print(f"  来源（{len(result.sources)} 条）：")
    for source in result.sources:
        print(f"  - {source.title}")
        print(f"    {source.url}")
        if source.snippet:
            print(f"    {source.snippet[:120]}…")
    print(f"  是否因 max_results 截断: {result.truncated}")

    print()
    print("=== ② web_fetch：真实抓取网页 ===")
    content = web_fetch("https://github.com/deepseek-ai/deepseek-harness")
    for line in content.splitlines():
        print(f"  {line[:100]}")


if __name__ == "__main__":
    main()
