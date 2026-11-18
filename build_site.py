"""把 README.md + results/ 构建成静态网站（site/），用于部署到 Cloudflare Pages。

用法:
    python build_site.py
    npx wrangler pages deploy site --project-name us-momentum-factor
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
REPO_URL = "https://github.com/EASONHUANG822/us-momentum-factor"

TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="在 S&P 500 上复现 Jegadeesh-Titman 12-1 动量因子，并与 Ken French UMD 对照验证。">
<style>
:root {{
  --bg: #fbfbf9; --fg: #1d1f21; --muted: #5f6368; --border: #e3e3df;
  --accent: #2f5d9e; --code-bg: #f1f1ee; --th-bg: #f4f4f1;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #15171a; --fg: #e6e6e3; --muted: #9aa0a6; --border: #2c2f34;
    --accent: #8ab4f8; --code-bg: #1f2226; --th-bg: #1c1f23;
  }}
  img {{ background: #fff; border-radius: 6px; }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.75 -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB",
        "Microsoft YaHei", "Segoe UI", sans-serif;
}}
main {{ max-width: 860px; margin: 0 auto; padding: 48px 16px 80px; }}
.topbar {{ display: flex; justify-content: space-between; align-items: center;
  font-size: 14px; color: var(--muted); margin-bottom: 24px; gap: 12px; flex-wrap: wrap; }}
.topbar a {{ color: var(--accent); text-decoration: none; }}
h1 {{ font-size: 1.9em; line-height: 1.3; margin: 0 0 .6em; }}
h2 {{ font-size: 1.35em; margin: 2.2em 0 .7em; padding-bottom: .3em; border-bottom: 1px solid var(--border); }}
a {{ color: var(--accent); }}
img {{ max-width: 100%; height: auto; display: block; margin: 1.2em auto; }}
.table-wrap {{ overflow-x: auto; margin: 1em 0; }}
table {{ border-collapse: collapse; font-size: 14px; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid var(--border); padding: 6px 10px; text-align: left; white-space: nowrap; }}
th {{ background: var(--th-bg); }}
td[style*="right"], th[style*="right"] {{ text-align: right; }}
code {{ background: var(--code-bg); padding: .1em .35em; border-radius: 4px; font-size: .9em; }}
pre {{ background: var(--code-bg); padding: 14px 16px; border-radius: 6px; overflow-x: auto; line-height: 1.5; }}
pre code {{ background: none; padding: 0; }}
footer {{ margin-top: 64px; font-size: 13px; color: var(--muted); }}
</style>
</head>
<body>
<main>
<div class="topbar"><span>Factor Replication · US Equities</span><a href="{repo}">GitHub 源码 →</a></div>
{body}
<footer>由 <code>build_site.py</code> 从仓库 README 与 results/ 自动生成 · <a href="{repo}">{repo}</a></footer>
</main>
</body>
</html>
"""


def main() -> None:
    md_text = (ROOT / "README.md").read_text(encoding="utf-8")
    title = md_text.splitlines()[0].lstrip("# ").strip()
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    # 宽表格横向滚动（手机端不撑破页面）
    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body)
    body = body.replace("</table>", "</table></div>")
    # 仓库内源码链接 -> GitHub
    body = re.sub(r'href="(?!https?://|results/|#)([^"]+)"', rf'href="{REPO_URL}/blob/main/\1"', body)

    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    shutil.copytree(ROOT / "results", SITE / "results")
    (SITE / "index.html").write_text(TEMPLATE.format(title=title, body=body, repo=REPO_URL),
                                     encoding="utf-8")
    print(f"Built {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
