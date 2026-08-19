"""Phase 1261: validate the static industry progress page and theory chapter."""

from __future__ import annotations

import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "frontend" / "website"
PAGE = SITE / "industry_progress.html"
THEORY = ROOT / "research" / "IntelligentTheory.md"
RESULT = ROOT / "tests" / "glm5" / "result" / "phase1261_industry_progress_page.json"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[str] = []
        self.stylesheets: list[str] = []
        self.lang = ""
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "html":
            self.lang = values.get("lang", "") or ""
        if values.get("id"):
            self.ids.append(values["id"] or "")
        if tag == "a" and values.get("href"):
            self.links.append(values["href"] or "")
        if tag == "link" and values.get("rel") == "stylesheet" and values.get("href"):
            self.stylesheets.append(values["href"] or "")
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)


def parse(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def main() -> int:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    add("page_exists", PAGE.is_file(), str(PAGE.relative_to(ROOT)))
    if not PAGE.is_file():
        raise SystemExit(1)

    page_text = PAGE.read_text(encoding="utf-8")
    parser = parse(PAGE)
    duplicate_ids = sorted(name for name, count in Counter(parser.ids).items() if count > 1)
    add("document_language", parser.lang == "zh-CN", parser.lang)
    add("document_title", "行业进展" in "".join(parser.title_parts), "".join(parser.title_parts))
    add("unique_ids", not duplicate_ids, duplicate_ids)
    add("stylesheet_wired", parser.stylesheets == ["styles.css"], parser.stylesheets)

    required_content = [
        "OpenAI",
        "Google DeepMind",
        "Anthropic",
        "Multimodal Neurons",
        "Gemma Scope 2",
        "Global Workspace",
        "行业共同积累的 14 块拼图",
        "研究路线的共同演化",
        "2026 年 8 月 13 日",
    ]
    missing_content = [item for item in required_content if item not in page_text]
    add("required_evidence_content", not missing_content, missing_content)
    forbidden_analysis = [
        "项目启示",
        "对当前研究",
        "多事件因果运输路径",
        "停止条件",
        "WP0",
        "未来响应等价类",
    ]
    retained_analysis = [item for item in forbidden_analysis if item in page_text]
    add("no_current_project_analysis", not retained_analysis, retained_analysis)
    mojibake_tokens = ["�", "缁撹", "鎬荤粨", "鈫"]
    detected_mojibake = [token for token in mojibake_tokens if token in page_text]
    add("utf8_content_clean", not detected_mojibake, detected_mojibake)

    missing_local_links: list[str] = []
    missing_fragments: list[str] = []
    for href in parser.links:
        parsed = urlparse(href)
        if parsed.scheme or href.startswith("mailto:") or href == "#":
            continue
        path_text, _, fragment = href.partition("#")
        target = PAGE if not path_text else SITE / path_text
        if not target.is_file():
            missing_local_links.append(href)
            continue
        if fragment:
            target_parser = parse(target)
            if fragment not in target_parser.ids:
                missing_fragments.append(href)
    add("local_links_exist", not missing_local_links, sorted(set(missing_local_links)))
    add("local_fragments_exist", not missing_fragments, sorted(set(missing_fragments)))

    html_pages = sorted(SITE.glob("*.html"))
    agi_page_text = (SITE / "agi_project.html").read_text(encoding="utf-8")
    secondary_links_ok = (
        'class="research-subnav"' in page_text
        and 'class="research-subnav"' in agi_page_text
        and 'href="industry_progress.html"' in agi_page_text
    )
    add("agi_page_secondary_navigation", secondary_links_ok, "AGI secondary navigation is visible inside both research pages")

    non_plain_primary_nav = [
        path.name
        for path in html_pages
        if 'class="nav-dropdown"' in path.read_text(encoding="utf-8")
        or 'class="nav-submenu"' in path.read_text(encoding="utf-8")
    ]
    add("plain_primary_navigation", not non_plain_primary_nav, non_plain_primary_nav)

    page_subnav = [
        path.name
        for path in html_pages
        if 'class="research-subnav"' in path.read_text(encoding="utf-8")
    ]
    add("secondary_nav_only_on_agi_pages", page_subnav == ["agi_project.html", "industry_progress.html"], page_subnav)
    add(
        "secondary_nav_has_no_parent_label",
        "<span>AGI研究</span>" not in page_text and "<span>AGI研究</span>" not in agi_page_text,
        "the centered page submenu contains only the two page links",
    )

    primary_nav = page_text.split('<nav class="nav-links"', 1)[1].split("</nav>", 1)[0]
    parent_active = 'class="active" href="agi_project.html"' in primary_nav
    add("agi_parent_active", parent_active, "AGI research remains the active primary navigation item")

    add(
        "active_page_subnav_item",
        'class="active" href="industry_progress.html" aria-current="page"' in page_text,
        "industry progress marks its in-page secondary navigation item as current",
    )

    css_text = (SITE / "styles.css").read_text(encoding="utf-8")
    hero_rule = css_text.split(".industry-hero {", 1)[1].split("}", 1)[0]
    lead_rule = css_text.split(".industry-page .industry-lead {", 1)[1].split("}", 1)[0]
    subnav_rule = css_text.split(".research-subnav {", 1)[1].split("}", 1)[0]
    add(
        "secondary_nav_is_centered",
        "justify-content: center;" in subnav_rule and "margin: 1.15rem auto 0;" in subnav_rule,
        "the page submenu is centered at all viewport widths",
    )
    add(
        "industry_hero_is_white",
        "background: #ffffff;" in hero_rule and "color: #10284d;" in hero_rule,
        "the complete industry progress hero uses a white background with dark text",
    )
    add(
        "lead_is_not_isolated_white_card",
        "background: transparent;" in lead_rule and "box-shadow: none;" in lead_rule,
        "the lead is integrated into the white hero instead of being the only white block",
    )
    required_selectors = [
        ".industry-hero",
        ".company-jump-nav",
        ".lab-grid",
        ".lab-profile",
        ".research-focus",
        ".company-progress .lab-timeline",
        ".research-subnav",
        ".industry-page .industry-lead",
        ".industry-page .industry-lead-label",
        ".puzzle-grid",
        ".analysis-stage-grid",
        ".claim-audit",
        "@media (max-width: 980px)",
        "@media (max-width: 640px)",
    ]
    missing_selectors = [selector for selector in required_selectors if selector not in css_text]
    add("responsive_styles_present", not missing_selectors, missing_selectors)

    theory_text = THEORY.read_text(encoding="utf-8")
    theory_requirements = [
        "## 七，行业进展：机械可解释性的外部证据与项目约束",
        "### 7.3 行业已经积累的十四块机制拼图",
        "### 7.6 基本研究对象：多事件未来响应等价类",
        "### 7.7 下一阶段的大任务",
        "H_E\\sim_{\\mathcal T,c,\\varepsilon}H'_E",
    ]
    missing_theory = [item for item in theory_requirements if item not in theory_text]
    add("theory_chapter_complete", not missing_theory, missing_theory)

    failures = [check["name"] for check in checks if not check["passed"]]
    report = {
        "phase": 1261,
        "page": str(PAGE.relative_to(ROOT)),
        "site_page_count": len(html_pages),
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
