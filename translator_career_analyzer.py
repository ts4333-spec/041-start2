"""
translator_career_analyzer.py
────────────────────────────────────────────────────────────────
역자 커리어 기반 원서 언어 추론 모듈

trans.py의 커리어 분석 구조 +
lang_field.py의 정밀 원제 언어 판별 로직을 결합.

핵심 파이프라인:
  역자 이름
    → 알라딘 ItemSearch API (역자가 번역한 책 N권)
    → 책마다 원제 언어 판별
        비라틴 원제 → 유니코드 즉시 판별 (무료, 빠름)
        라틴 원제   → gpt_guess_from_original_title_only (정밀)
        원제 없음   → 카테고리 힌트
    → 언어별 카운트 집계
    → 주 번역 언어 확정
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import json
import time
import urllib.request
import urllib.error
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# lang_field.py를 같은 디렉터리에 두고 import
from lang_field import LangFieldBuilder, ISDS_LANGUAGE_CODES, _RE_LATIN, _RE_NON_LATIN

# ── 알라딘 API 설정 ─────────────────────────────────────────────
_ITEM_SEARCH_URL = "http://www.aladin.co.kr/ttb/api/ItemSearch.aspx"
_API_VERSION     = "20131101"


def _get_json(url: str, params: dict, timeout: int = 15) -> dict:
    """알라딘 API JSON GET 헬퍼."""
    query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    full_url = f"{url}?{query}"
    try:
        req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


import urllib.parse


# ── 역자 커리어 도서 조회 ────────────────────────────────────────
def fetch_translator_catalog(
    translator_name: str,
    ttbkey: str,
    max_results: int = 50,
) -> List[dict]:
    """
    역자 이름으로 알라딘 ItemSearch API를 조회하여
    해당 역자가 번역한 책 목록을 반환.
    """
    if not ttbkey or not translator_name.strip():
        return []
    data = _get_json(_ITEM_SEARCH_URL, {
        "ttbkey":       ttbkey.strip(),
        "QueryType":    "Author",
        "Query":        translator_name.strip(),
        "MaxResults":   str(max_results),
        "start":        "1",
        "SearchTarget": "Book",
        "output":       "js",
        "Version":      _API_VERSION,
        "OptResult":    "authors",
    })
    return data.get("item", []) or []


# ── 책 1권의 원제 언어 판별 ──────────────────────────────────────
def detect_book_original_language(
    book: dict,
    builder: LangFieldBuilder,
    gpt_enabled: bool = True,
) -> Tuple[str, str]:
    """
    책 1권의 원서 언어를 판별.
    반환: (isds_code, method)
      · isds_code: 'eng'/'fre'/'jpn' 등, 판별 불가면 'und'
      · method:    판별 방법 설명 (로그/디버그용)
    """
    sub = book.get("subInfo") or {}
    original_title = (
        sub.get("originalTitle") or sub.get("subTitle") or ""
    ).strip()
    category_text = book.get("categoryName") or ""

    # ── 1순위: 원제 유니코드 즉시 판별 (비라틴) ──────────────────
    if original_title:
        uni_lang = builder.detect_language_by_unicode(original_title)
        if uni_lang not in ("und", "kor"):
            return uni_lang, f"유니코드 감지 (원제: {original_title[:30]})"

        # ── 2순위: 라틴 원제 → GPT 정밀 판별 ────────────────────
        is_latin_only = (
            bool(_RE_LATIN.search(original_title))
            and not _RE_NON_LATIN.search(original_title)
        )
        if is_latin_only and gpt_enabled:
            gpt_lang = builder.gpt_guess_from_original_title_only(original_title)
            if gpt_lang and gpt_lang != "und":
                return gpt_lang, f"GPT 원제 판별 (원제: {original_title[:30]})"
            return "und", f"GPT 판별 불가 (원제: {original_title[:30]})"

    # ── 3순위: 카테고리 힌트 ─────────────────────────────────────
    if category_text:
        cat_lang = builder.detect_language_from_category(category_text)
        if cat_lang:
            return cat_lang, f"카테고리 힌트 ({category_text[:40]})"

    return "und", "단서 없음"


# ── 역자 커리어 전체 분석 ────────────────────────────────────────
def analyze_translator_career(
    translator_name: str,
    ttbkey: str,
    openai_client=None,
    model: str = "gpt-4o-mini",
    max_results: int = 50,
    gpt_enabled: bool = True,
    dbg_fn=None,
) -> Dict[str, Any]:
    """
    역자 이름으로 커리어 전체를 분석하여 주 번역 언어를 확정.

    반환값:
    {
        "translator_name": str,
        "total_books": int,               # 조회된 총 책 수
        "analyzed_books": int,            # 판별 성공한 책 수
        "language_counts": {              # 언어별 빈도
            "jpn": 12, "fre": 8, ...
        },
        "primary_language": str,          # 최다 빈도 언어 코드
        "primary_language_name": str,     # 한국어 언어명
        "confidence": str,                # "high"/"medium"/"low"
        "books_detail": [                 # 책별 판별 결과
            {"title": ..., "original_title": ..., "lang": ..., "method": ...}
        ]
    }
    """
    _log = dbg_fn or (lambda *a: None)

    builder = LangFieldBuilder(
        openai_client=openai_client,
        model=model,
        dbg_fn=_log,
    )

    _log(f"📚 [{translator_name}] 커리어 조회 시작 (최대 {max_results}권)")
    books = fetch_translator_catalog(translator_name, ttbkey, max_results)
    _log(f"📚 [{translator_name}] {len(books)}권 조회됨")

    language_counts: Counter = Counter()
    books_detail: List[dict] = []
    gpt_call_count = 0

    for book in books:
        title = book.get("title") or "(제목 없음)"
        sub = book.get("subInfo") or {}
        original_title = (sub.get("originalTitle") or sub.get("subTitle") or "").strip()

        # 라틴 원제는 GPT 호출 — 호출 간격 조절
        is_latin = (
            bool(original_title)
            and bool(_RE_LATIN.search(original_title))
            and not _RE_NON_LATIN.search(original_title)
        )
        if is_latin and gpt_enabled:
            gpt_call_count += 1
            if gpt_call_count > 1:
                time.sleep(0.3)   # 레이트리밋 방지

        lang, method = detect_book_original_language(book, builder, gpt_enabled)

        if lang != "und":
            language_counts[lang] += 1

        books_detail.append({
            "title":          title[:50],
            "original_title": original_title[:40] if original_title else None,
            "lang":           lang,
            "lang_name":      ISDS_LANGUAGE_CODES.get(lang, "?"),
            "method":         method,
        })
        _log(f"  · {title[:30]} → {lang} ({method})")

    # ── 결과 집계 ────────────────────────────────────────────────
    total = len(books)
    analyzed = sum(1 for b in books_detail if b["lang"] != "und")

    if not language_counts:
        primary_lang = "und"
        confidence   = "low"
    else:
        primary_lang, top_count = language_counts.most_common(1)[0]
        ratio = top_count / analyzed if analyzed else 0
        confidence = "high" if ratio >= 0.6 else "medium" if ratio >= 0.35 else "low"

    _log(f"\n📊 [{translator_name}] 분석 완료")
    _log(f"   총 {total}권 조회 / {analyzed}권 판별 성공")
    _log(f"   언어 분포: {dict(language_counts.most_common(5))}")
    _log(f"   주 번역 언어: {primary_lang} (신뢰도: {confidence})")

    return {
        "translator_name":      translator_name,
        "total_books":          total,
        "analyzed_books":       analyzed,
        "language_counts":      dict(language_counts.most_common()),
        "primary_language":     primary_lang,
        "primary_language_name": ISDS_LANGUAGE_CODES.get(primary_lang, "판별 불가"),
        "confidence":           confidence,
        "books_detail":         books_detail,
    }
