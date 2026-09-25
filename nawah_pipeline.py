"""
Nawah — Multi-Agent Business Advisory Pipeline
================================================
This module is the server-ready extraction of `nawah_professional_v3.ipynb`.
It contains the exact same five-stage pipeline:

    Researcher v2 -> Verifier 1 -> Planner v2 -> Verifier 2 -> Orchestrator

plus the conversation layer (`nawah_turn`) used for follow-up questions with
session memory. Nothing about the agent LOGIC was changed from the notebook —
only the Colab-specific bits were removed:

  - `google.colab.userdata` -> plain `os.environ` (use a `.env` file or real
    environment variables / secrets manager in production)
  - relative `open("knowledge_2.json")` -> configurable `NAWAH_DATA_DIR`
  - Gradio / notebook printing -> this module never prints; call sites decide

Required environment variables:
  OPENAI_API_KEY       (required)
  LANGCHAIN_API_KEY    (optional -- enables LangSmith tracing)
  NAWAH_DATA_DIR        (optional -- defaults to ./data ; must contain
                          knowledge_2.json, quality_issues.json, manifest.json)

NOTE (carried over from the notebook, unchanged):
  Model names (`gpt-5.6-sol` / `-terra` / `-luna`) and the `web_search` tool
  are the best guess available at notebook-authoring time — verify them
  against the current OpenAI docs before relying on this in production.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import json
import time
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from openai import OpenAI

# --------------------------------------------------------------------------
# Tracing (LangSmith) -- optional, exactly as in the notebook
# --------------------------------------------------------------------------
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "Nawah")
else:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --------------------------------------------------------------------------
# Model names -- verify against OpenAI docs before production use
# --------------------------------------------------------------------------
GPT_SOL = "gpt-5.6-sol"
GPT_TERRA = "gpt-5.6-terra"
GPT_LUNA = "gpt-5.6-luna"

PRICING = {
    "gpt_sol": {"input": 5.0, "output": 30.0},
    "gpt_terra": {"input": 2.5, "output": 15.0},
    "gpt_luna": {"input": 0.5, "output": 3.0},
}
USD_TO_SAR = 3.75

# --------------------------------------------------------------------------
# Language / Nationality
# --------------------------------------------------------------------------
Language = Literal["ar", "en"]
Nationality = Literal["saudi", "non_saudi", "unspecified"]
LANGUAGE_NAMES = {"ar": "Arabic", "en": "English"}


def detect_language(text: str) -> Language:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    arabic_letters = sum(1 for ch in letters if "\u0600" <= ch <= "\u06FF")
    return "ar" if (arabic_letters / len(letters)) > 0.3 else "en"


# --------------------------------------------------------------------------
# Cost tracking -- per-request (not a global list in the server, see below)
# --------------------------------------------------------------------------
def _new_cost_log() -> List[Dict]:
    return []


def log_cost(cost_log: List[Dict], agent_name: str, model_key: str,
             input_tokens: int, output_tokens: int) -> float:
    price = PRICING[model_key]
    cost_usd = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    cost_sar = cost_usd * USD_TO_SAR
    cost_log.append({
        "agent": agent_name, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "cost_sar": cost_sar,
    })
    return cost_sar


# --------------------------------------------------------------------------
# Shared utilities
# --------------------------------------------------------------------------
def extract_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])


_NON_RETRYABLE_CODES = {
    "insufficient_quota", "invalid_api_key", "model_not_found", "permission_denied",
}


def _is_retryable(error: Exception) -> bool:
    code = getattr(getattr(error, "body", None), "get", lambda *_: None)("code") \
        if hasattr(error, "body") else None
    if code is None:
        code = getattr(error, "code", None)
    if code in _NON_RETRYABLE_CODES:
        return False
    status = getattr(error, "status_code", None)
    if status == 429 and code == "project_spend_limit_exceeded":
        return False
    return True


def call_with_retry(fn, retries: int = 3, delay: float = 2.0):
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if not _is_retryable(e):
                raise
            time.sleep(delay)
    raise last_error


# --------------------------------------------------------------------------
# Data layer v5 -- read the complete Nawah data folder, not one bundled JSON
# --------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("NAWAH_DATA_DIR", "./data")).resolve()


def _first_existing(*candidates: Path) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


_CONSOLIDATED = _first_existing(
    DATA_DIR / "nawah_consolidated_dataset" / "consolidated",
    DATA_DIR / "consolidated",
)

# Prefer the canonical split dataset because it preserves dependencies, sources,
# service rules and Balady rules as separate first-class records.
if _CONSOLIDATED:
    def _load_json(name, default):
        fp = _CONSOLIDATED / name
        if not fp.exists():
            return default
        with open(fp, encoding="utf-8") as f:
            return json.load(f)

    SOURCE_REGISTRY: List[Dict] = _load_json("source_registry.json", [])
    BALADY_ACTIVITIES: List[Dict] = _load_json("balady_activities_with_rules.json", [])
    SBC_ACTIVITIES: List[Dict] = _load_json("sbc_activities.json", [])
    SERVICE_KNOWLEDGE: Dict = _load_json("service_knowledge.json", {})
    CROSSWALK_CANDIDATES: List[Dict] = _load_json("activity_crosswalk_candidates.json", [])
    QUALITY_ISSUES: List[Dict] = _load_json("quality_issues.json", [])
    DATA_MANIFEST: Dict = _load_json("manifest.json", {})
else:
    # Backward compatibility with the older single-file export.
    kb_path = DATA_DIR / "knowledge_2.json"
    if not kb_path.exists():
        raise FileNotFoundError(
            f"Nawah data not found under {DATA_DIR}. Extract data.zip so this folder "
            "contains nawah_consolidated_dataset/consolidated (recommended), or the "
            "legacy knowledge_2.json."
        )
    with open(kb_path, encoding="utf-8") as f:
        _KB = json.load(f)
    SOURCE_REGISTRY = _KB.get("source_registry", [])
    BALADY_ACTIVITIES = _KB.get("balady_activities_with_rules", [])
    SBC_ACTIVITIES = _KB.get("sbc_activities", [])
    SERVICE_KNOWLEDGE = _KB.get("service_knowledge", {})
    CROSSWALK_CANDIDATES = _KB.get("activity_crosswalk_candidates", [])
    with open(DATA_DIR / "quality_issues.json", encoding="utf-8") as f:
        QUALITY_ISSUES = json.load(f)
    manifest_path = DATA_DIR / "manifest.json"
    DATA_MANIFEST = json.load(open(manifest_path, encoding="utf-8")) if manifest_path.exists() else {}

# Also register EVERY readable data file in the supplied data directory.  This
# includes originals, processed audit/review files, corrections, JSONL, CSV,
# TXT and Markdown.  They are searched on demand by search_all_data(), rather
# than loading 100+ MB into every prompt.
_READABLE_EXTS = {".json", ".jsonl", ".txt", ".md", ".csv"}
ALL_DATA_FILES: List[Path] = sorted(
    p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in _READABLE_EXTS
)


# First-class official source/service registry. sources.json is intentionally
# loaded separately from source_registry.json because it contains the broad
# authority + service category + official URL catalogue used by Nawah.
def _load_source_catalog() -> List[Dict]:
    rows: List[Dict] = []
    paths = sorted(p for p in DATA_DIR.rglob("sources*.json") if p.is_file())
    # Put an exact sources.json before backup/original copies.
    paths.sort(key=lambda p: (p.name != "sources.json", len(p.parts), str(p)))
    for fp in paths:
        try:
            raw = json.load(open(fp, encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("sources", []) if isinstance(raw, dict) else []
            for item in items:
                if isinstance(item, dict) and item.get("url"):
                    row = dict(item); row["_registry_file"] = str(fp.relative_to(DATA_DIR)); rows.append(row)
        except (OSError, json.JSONDecodeError):
            pass
    # Keep consolidated source_registry too, after explicit sources.json.
    for item in SOURCE_REGISTRY:
        if isinstance(item, dict) and item.get("url"):
            row = dict(item); row["_registry_file"] = "consolidated/source_registry.json"; rows.append(row)
    out=[]; seen=set()
    for row in rows:
        url=str(row.get("url", "")).strip()
        if url and url not in seen:
            seen.add(url); out.append(row)
    return out

SOURCE_CATALOG: List[Dict] = _load_source_catalog()


def search_sources(query: str, top_k: int = 8) -> List[Dict]:
    import re
    q=query.lower().strip()
    terms=[t for t in re.findall(r"[\w\u0600-\u06ff]+", q) if len(t)>1]
    aliases={
        "بلدي":["balady","commercial_licenses"], "بلدية":["balady","commercial_licenses"],
        "رخصة":["license","licensing","commercial_licenses"], "ترخيص":["license","licensing"],
        "سجل":["commerce","business_startup","formation"], "تجاري":["commerce","commercial","business"],
        "استثمار":["investment","misa"], "ضريبة":["zatca","vat"], "زكاة":["zatca"],
        "فاتورة":["e_invoicing","zatca"], "غذاء":["food","sfda"], "مقهى":["food","commercial_licenses"],
        "مطعم":["food","commercial_licenses"], "عمال":["employment","qiwa","gosi"],
        "توظيف":["employment","qiwa","gosi"], "صحي":["healthcare","moh"],
        "سياحة":["tourism","hospitality"], "عقار":["real_estate","rega"],
    }
    expanded=list(terms)
    for t in terms: expanded.extend(aliases.get(t, []))
    hits=[]
    for src in SOURCE_CATALOG:
        if src.get("enabled") is False: continue
        authority=str(src.get("authority", "")).lower(); category=str(src.get("category", "")).lower()
        activity=str(src.get("business_activity", "")).lower(); blob=" ".join((authority,category,activity))
        score=0
        for term in expanded:
            t=term.lower()
            if t in category: score+=5
            elif t in authority: score+=4
            elif t in activity: score+=3
            elif t in blob: score+=1
        if q and q in blob: score+=10
        if score: hits.append((score,src))
    hits.sort(key=lambda x:(-x[0], str(x[1].get("authority","")), str(x[1].get("category",""))))
    result=[]
    for score,src in hits[:max(1,min(top_k,20))]:
        item={k:v for k,v in src.items() if not k.startswith("_")}
        item["match_score"]=score; item["registry_file"]=src.get("_registry_file"); result.append(item)
    return result


def best_source_url(query: str) -> Optional[str]:
    hits=search_sources(query, top_k=1)
    return hits[0].get("url") if hits else None


def data_inventory() -> Dict:
    return {
        "data_dir": str(DATA_DIR),
        "files_readable": len(ALL_DATA_FILES),
        "files": [str(p.relative_to(DATA_DIR)) for p in ALL_DATA_FILES],
        "canonical_counts": {
            "sources_consolidated": len(SOURCE_REGISTRY), "sources_catalog": len(SOURCE_CATALOG), "balady_activities": len(BALADY_ACTIVITIES),
            "sbc_activities": len(SBC_ACTIVITIES), "crosswalks": len(CROSSWALK_CANDIDATES),
            "quality_issues": len(QUALITY_ISSUES),
        },
    }


def search_all_data(query: str, top_k: int = 12) -> List[Dict]:
    """Literal/token search across every readable file in data/.  Useful for
    records that were not promoted into the consolidated canonical objects."""
    terms = [t for t in query.lower().split() if len(t) > 1]
    if not terms:
        return []
    hits = []
    for fp in ALL_DATA_FILES:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    low = line.lower()
                    score = sum(low.count(t) for t in terms)
                    if score:
                        hits.append({"score": score, "file": str(fp.relative_to(DATA_DIR)),
                                     "line": line_no, "text": line.strip()[:1800]})
        except OSError:
            continue
    hits.sort(key=lambda x: (-x["score"], x["file"], x["line"]))
    return hits[:max(1, min(top_k, 30))]


_BALADY_BY_ID: Dict[str, Dict] = {str(b.get("activity", {}).get("activity_id")): b for b in BALADY_ACTIVITIES}


def get_quality_flags(topic_or_location_substring: str) -> List[Dict]:
    hits = []
    for issue in QUALITY_ISSUES:
        blob = json.dumps(issue, ensure_ascii=False)
        if topic_or_location_substring.lower() in blob.lower():
            hits.append(issue)
    return hits


def is_blocked(topic_or_location_substring: str) -> bool:
    return any(i.get("severity") == "BLOCKING_FOR_AUTOMATED_ANSWERS"
               for i in get_quality_flags(topic_or_location_substring))


def source_for(authority_substring: str) -> List[Dict]:
    return [s for s in SOURCE_REGISTRY
            if authority_substring.lower() in s.get("authority", "").lower()]


def _official_url(*needles: str) -> Optional[str]:
    needles = tuple(n.lower() for n in needles if n)
    for src in SOURCE_CATALOG:
        if not src.get("enabled", True) or not src.get("url"):
            continue
        blob = " ".join(str(src.get(k, "")) for k in ("authority", "category", "business_activity")).lower()
        if any(n in blob for n in needles):
            return src["url"]
    return None


def get_crosswalk_candidates(sbc_activity_id: str) -> List[Dict]:
    return [c for c in CROSSWALK_CANDIDATES if str(c.get("sbc_activity_id")) == str(sbc_activity_id)]


def get_balady_rules_for_activity(balady_activity_id: str) -> Optional[Dict]:
    return _BALADY_BY_ID.get(str(balady_activity_id))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_sbc_texts = [" ".join(filter(None, [a.get("nameAr"), a.get("descriptionAr"),
              (a.get("levelOne") or {}).get("nameAr"), (a.get("levelTwo") or {}).get("nameAr")]))
              for a in SBC_ACTIVITIES]
_sbc_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50000)
_sbc_matrix = _sbc_vectorizer.fit_transform(_sbc_texts) if _sbc_texts else None

_balady_texts = [" ".join(filter(None, [b.get("activity", {}).get("activity_name"),
                 b.get("activity", {}).get("description"), b.get("activity", {}).get("section_name"),
                 b.get("activity", {}).get("main_activity_name")])) for b in BALADY_ACTIVITIES]
_balady_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=70000)
_balady_matrix = _balady_vectorizer.fit_transform(_balady_texts) if _balady_texts else None


def _sbc_search(query: str, top_k: int = 5) -> List[Dict]:
    if _sbc_matrix is None: return []
    sims = cosine_similarity(_sbc_vectorizer.transform([query]), _sbc_matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{**SBC_ACTIVITIES[i], "similarity": round(float(sims[i]), 4)} for i in top_idx]


def search_balady_activities(query: str, top_k: int = 5) -> List[Dict]:
    """Search Balady directly. This avoids treating the SBC->Balady candidate
    crosswalk as the source of truth for activity identification."""
    if _balady_matrix is None: return []
    sims = cosine_similarity(_balady_vectorizer.transform([query]), _balady_matrix)[0]
    out=[]
    for i in np.argsort(sims)[::-1][:top_k]:
        rec=BALADY_ACTIVITIES[i]; a=rec.get("activity", {})
        rule_urls=[]
        for r in rec.get("rules", []): rule_urls.extend(r.get("urls") or [])
        out.append({"activity_id": str(a.get("activity_id")), "activity_name": a.get("activity_name"),
                    "description": a.get("description"), "supervising_authority": a.get("supervising_authority_name"),
                    "similarity": round(float(sims[i]),4), "rules_count": len(rec.get("rules", [])),
                    "source_url": (rule_urls[0] if rule_urls else _official_url("balady", "municipal"))})
    return out


def search_sbc_activities(query: str, top_k: int = 5) -> List[Dict]:
    results = _sbc_search(query, top_k)
    for r in results:
        candidates = get_crosswalk_candidates(r.get("activityId"))
        r["balady_candidates"] = candidates
        r["balady_candidate_rules_preview"] = [{
            "balady_activity_id": c.get("balady_activity_id"), "balady_activity_name": c.get("balady_activity_name"),
            "match_status": c.get("status"), "rules_count": len((_BALADY_BY_ID.get(str(c.get("balady_activity_id"))) or {}).get("rules", [])),
            "supervising_authority": (_BALADY_BY_ID.get(str(c.get("balady_activity_id"))) or {}).get("activity", {}).get("supervising_authority_name")
        } for c in candidates[:3]]
    return results


def _service_source_url(domain: str, service: Optional[Dict]=None) -> Optional[str]:
    if service:
        for key in ("source_url", "url", "serviceUrl"):
            if service.get(key): return service[key]
    block=SERVICE_KNOWLEDGE.get(domain, {})
    for src in block.get("sources", []) if isinstance(block, dict) else []:
        if isinstance(src, dict) and src.get("url"): return src["url"]
    needles={"sbc":("commerce","business center","commercial_registration"),
             "misa":("investment","misa"), "ejar":("ejar",)}.get(domain,(domain,))
    return _official_url(*needles)


def get_dependency_rules() -> Dict:
    return {k: SERVICE_KNOWLEDGE.get(k, {}).get("dependencies", [])
            for k in ("sbc", "misa", "ejar")}

# --------------------------------------------------------------------------
# Researcher v2
# --------------------------------------------------------------------------
_RESEARCHER_V2_TOOLS = [
    {"type":"function","name":"search_official_sources","description":"يبحث في sources.json كمرجع أساسي للجهات والخدمات وروابطها الرسمية. استخدمه لكل خدمة أو ترخيص لإرجاع الرابط المخزن في البيانات.","parameters":{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer"}},"required":["query"]}},
    {"type":"function","name":"search_balady_direct","description":"يبحث مباشرة في كامل أنشطة بلدي وقواعدها لتحديد النشاط دون الاعتماد على crosswalk مرشح.","parameters":{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer"}},"required":["query"]}},
    {"type":"function","name":"search_all_data","description":"يبحث في كل ملفات data المتاحة (canonical/originals/processed/corrections وJSON/JSONL/TXT/CSV/MD) عند الحاجة لمعلومة أو رابط أو اعتماد غير ظاهر في الطبقة المجمعة.","parameters":{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer"}},"required":["query"]}},
    {"type":"function","name":"get_dependency_rules","description":"يعيد الاعتماديات الصريحة من بيانات SBC وMISA وإيجار كما هي، لا اعتماديات مخترعة.","parameters":{"type":"object","properties":{}}},
    {
        "type": "function", "name": "search_activities",
        "description": (
            "يبحث في فهرس الأنشطة التجارية المدقَّق (٢٦٩٦ نشاط من المركز السعودي "
            "للأعمال بعد إزالة التكرار) عن أقرب نشاط للوصف. كل نتيجة تتضمن "
            "balady_candidate_rules_preview -- معاينة قواعد بلدي حقيقية مرتبطة "
            "بروابط *مرشّحة غير مؤكدة* (status: CANDIDATE_VERIFY)، لا تُعامل "
            "كمطابقة أكيدة إلا بعد التحقق."
        ),
        "parameters": {"type": "object",
                        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                        "required": ["query"]},
    },
    {
        "type": "function", "name": "get_balady_rules",
        "description": (
            "تجيب القواعد البلدية الحقيقية الكاملة (تصاريح، اشتراطات، الجهة "
            "المشرفة) لنشاط بلدي محدد بمعرّفه (balady_activity_id) -- من "
            "الملف المعتمد الرسمي مباشرة، لا من رابط مرشّح."
        ),
        "parameters": {"type": "object",
                        "properties": {"balady_activity_id": {"type": "string"}},
                        "required": ["balady_activity_id"]},
    },
    {
        "type": "function", "name": "get_sbc_registration_rules",
        "description": "شروط ومستندات تسجيل تجاري حقيقية (حجز اسم / قيد سجل فردي) من service_knowledge.sbc.",
        "parameters": {"type": "object",
                        "properties": {"service_id": {"type": "string",
                                        "enum": ["sole_proprietorship_cr", "trade_name_reservation"]}},
                        "required": ["service_id"]},
    },
    {
        "type": "function", "name": "get_misa_knowledge",
        "description": (
            "قواعد وزارة الاستثمار (MISA) الكاملة من service_knowledge.misa -- "
            "الأهلية، الأنشطة المقيدة، القواعد الخليجية، المستندات، والبنود "
            "غير المحلولة (unresolved_for_nawah) التي يجب عدم تأكيدها كحقيقة."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function", "name": "get_ejar_knowledge",
        "description": (
            "قواعد منصة إيجار الكاملة من service_knowledge.ejar، متضمنة حالة "
            "تعارض الرسوم (SOURCE_CONFLICT) والقواعد غير المؤكدة -- يجب عدم "
            "ذكر رقم رسوم واحد قاطع."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function", "name": "check_quality_flags",
        "description": "يتحقق من quality_issues.json لموضوع معيّن قبل تأكيد أي معلومة عنه.",
        "parameters": {"type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"]},
    },
]


def _execute_researcher_v2_tool(name: str, arguments: Dict) -> Dict:
    if name == "search_balady_direct":
        return {"results": search_balady_activities(arguments["query"], arguments.get("top_k", 5))}
    if name == "search_official_sources":
        return {"results": search_sources(arguments["query"], arguments.get("top_k", 8))}
    if name == "search_all_data":
        return {"results": search_all_data(arguments["query"], arguments.get("top_k", 12))}
    if name == "get_dependency_rules":
        return get_dependency_rules()
    if name == "search_activities":
        results = search_sbc_activities(arguments["query"], top_k=arguments.get("top_k", 5))
        formatted = []
        for r in results:
            preview = r["balady_candidate_rules_preview"]
            # BUGFIX: this used to hardcode the word "CANDIDATE_VERIFY" for
            # *any* non-empty preview, even when a candidate's own
            # match_status was already "CONFIRMED". That made every single
            # requirement in the final plan read as unverified, regardless
            # of what the crosswalk data actually said. Now the note only
            # warns about the candidates that are genuinely unresolved, and
            # says so plainly when every candidate returned is confirmed.
            unresolved = [c for c in preview if str(c.get("match_status", "")).upper() != "CONFIRMED"]
            if not preview:
                note = None
            elif unresolved:
                note = ("بعض روابط بلدي في هذه المعاينة لا تزال مرشّحة (CANDIDATE_VERIFY) "
                         "وبعضها قد يكون مؤكدًا (CONFIRMED) -- تحققي من match_status لكل "
                         "مرشّح بنفسه، ثم استخدمي get_balady_rules للتفاصيل الكاملة، واذكري "
                         "الحالة الحقيقية لكل عنصر كما وردت لا كقاعدة عامة على كل شيء")
            else:
                note = "كل روابط بلدي في هذه المعاينة مؤكدة (CONFIRMED) -- استخدمي get_balady_rules للتفاصيل الكاملة"
            formatted.append({
                "activityId": r["activityId"], "nameAr": r["nameAr"],
                "similarity": r["similarity"],
                "licenseRequiredBeforeIssue": r.get("licenseRequiredBeforeIssue"),
                "preAuthority": r.get("preAuthority"),
                "licenseRequiredAfterIssue": r.get("licenseRequiredAfterIssue"),
                "postAuthority": r.get("postAuthority"),
                "balady_candidate_rules_preview": preview,
                "note": note,
            })
        return {"results": formatted}
    if name == "get_balady_rules":
        rec = get_balady_rules_for_activity(arguments["balady_activity_id"])
        if not rec:
            return {"error": "activity_id not found in canonical balady dataset"}
        
        rule_urls = [u for r in rec.get("rules", []) for u in (r.get("urls") or [])]
        return {"activity": rec["activity"], "rules": rec["rules"], "rules_count": len(rec["rules"]),
                "source_url": (rule_urls[0] if rule_urls else _official_url("balady", "municipal"))}
    if name == "get_sbc_registration_rules":
        for s in SERVICE_KNOWLEDGE["sbc"]["services"]:
            if s.get("serviceId") == arguments["service_id"]:
                return {**s, "source_url": _service_source_url("sbc", s)}
        return {"error": "not found"}
    if name == "get_misa_knowledge":
        return {**SERVICE_KNOWLEDGE["misa"], "source_url": _service_source_url("misa")}
    if name == "get_ejar_knowledge":
        return {**SERVICE_KNOWLEDGE["ejar"], "source_url": _service_source_url("ejar")}
    if name == "check_quality_flags":
        return {"flags": get_quality_flags(arguments["topic"])}
    return {"error": f"unknown tool: {name}"}


_RESEARCHER_V2_SYSTEM: Dict[Language, str] = {
    "ar": ("أنتِ باحثة في مكتب استشارات أعمال، تجمعين المتطلبات النظامية لفكرة "
           "عمل من قاعدة بيانات مدقَّقة فقط -- ممنوع منعًا باتًا البحث المفتوح "
           "أو الاعتماد على معرفتك العامة، لأن كل معلومة يجب أن تُنسب لمصدر في "
           "هذي القاعدة تحديدًا.\n\n"
           "خطوات إلزامية:\n"
           "0. السكوب مفتوح لأي نشاط في السعودية. استنتجي النشاط من نص المستخدم نفسه ولا تحوّليه إلى مطعم أو مقهى أو نشاط افتراضي. إذا قال مستشفى/مشفى فابحثي كمنشأة صحية، وإذا قال مصنع فابحثي كنشاط صناعي، وهكذا. لا توجد قائمة أنشطة مغلقة.\n"
           "1. استخدمي search_activities لتحديد تصنيف SBC، واستخدمي search_balady_direct أيضًا لتحديد نشاط بلدي مباشرة من اسمه ووصفه. لا تعرضي CANDIDATE_VERIFY للعميل؛ هو إشارة داخلية فقط.\n"
           "2. إذا كان أي نشاط يطلبه المستخدم يحتاج موقعًا فعليًا، أيًا كان قطاعه (صحي، صناعي، تعليمي، مهني، تجاري، تقني، سياحي أو غيره)، استخدمي "
           "get_balady_rules بمعرّف النشاط البلدي الأقرب من المعاينة للحصول على "
           "القواعد الكاملة والحقيقية. انظري إلى match_status الحقيقي لهذا المرشّح "
           "بالضبط كما ورد: إذا كان 'CONFIRMED' فالربط مؤكد وتقدرين تصنّفي الاعتماد "
           "عليه 'verified' (بشرط عدم وجود علم جودة آخر عليه)؛ وإذا كان لا يزال "
           "'CANDIDATE_VERIFY' أو غير ذلك، فاذكري صراحة أنه مرشّح وغير مؤكد رسميًا. "
           "ممنوع تعميم كلمة 'مرشّح' أو 'CANDIDATE_VERIFY' على كل الروابط بلا "
           "استثناء -- كل رابط يوصف بحالته الحقيقية فقط.\n"
           "3. استخدمي get_sbc_registration_rules لخطوتي التسجيل التجاري.\n"
           "4. إذا كانت الجنسية غير سعودي أو غير محددة، استخدمي get_misa_knowledge.\n"
           "5. إذا كان النشاط يحتاج عقد إيجار، استخدمي get_ejar_knowledge.\n"
           "6. استدعي get_dependency_rules دائمًا وابني الاعتماديات من البيانات الصريحة.\n"
           "7. لكل متطلب أو خدمة، استخدمي search_official_sources للبحث في sources.json وإرفاق رابط الخدمة الرسمي المخزن فيه؛ sources.json هو المرجع الأول للجهة والخدمة والرابط.\n"
           "8. إذا لم تجدي معلومة/رابطًا في الأدوات المنظمة أو sources.json، استخدمي search_all_data للبحث في كل ملفات data قبل اعتبارها مفقودة.\n"           "9. لا تتوقفي عند الرخصة البلدية. ابنِي ملف تأسيس متكامل وابحثي، حسب انطباقه على المشروع، عن: "
           "الاستثمار لغير السعودي، الاسم التجاري، السجل/تأسيس الكيان، التسجيلات الحكومية المرتبطة بالسجل، "
           "الزكاة/ضريبة الدخل، اختيار/توثيق الموقع أو عقد الإيجار، السلامة، الرخصة التجارية/البلدية، "
           "ملف المنشأة والموارد البشرية، قوى والتأمينات عند التوظيف، الشهادات الصحية للأنشطة الغذائية، "
           "ضريبة القيمة المضافة عند تحقق حدها، والفوترة الإلكترونية عند انطباقها. لكل بند استخدمي sources.json "
           "ثم search_all_data، ولا تضيفي بندًا لا تجدين له سندًا في البيانات.\n"
           "10. اجعلي title وdescription وdocuments_needed عربية واضحة للعميل؛ لا تضعي أكواد الأنشطة أو "
           "أسماء حالات التدقيق التقنية أو ملاحظات المطابقة الداخلية في النص الموجّه للعميل.\n"
           "8. لأي معلومة حساسة استخدمي check_quality_flags. لا تضعي تحذير غير مؤكد لمجرد أن crosswalk مرشح إذا كانت الخطوة نفسها مثبتة مباشرة في بيانات الخدمة/بلدي.\n\n"
           "لكل متطلب تكتشفينه، حددي status بدقة:\n"
           "- 'verified': مذكور صراحة بمصدر رسمي بدون أي علم جودة عليه\n"
           "- 'unverified': مذكور لكن عليه علم REVIEW أو UNVERIFIED\n"
           "- 'blocked': BLOCKING_FOR_AUTOMATED_ANSWERS -- لا تذكري رقمًا محددًا، "
           "بس اذكري وجود الالتزام مع وصف التعارض\n"
           "- 'missing': أشارت الأدوات إنها MISSING_DATA\n\n"
           "أعيدي JSON فقط:\n"
           "{{\"requirements\": [{{\"id\": \"...\", \"title\": \"...\", "
           "\"authority\": \"...\", \"description\": \"...\", \"status\": \"...\", "
           "\"quality_note\": \"... أو null\", \"source_url\": \"... أو null\"}}]}}"),
    "en": ("You are a researcher at a business advisory office, gathering the "
           "regulatory requirements for a business idea from an audited "
           "database ONLY -- no open web search, no reliance on general "
           "knowledge, every fact must trace to this specific database.\n\n"
           "Mandatory steps:\n"
           "1. Use search_activities to identify the activity and preview its "
           "candidate balady rules.\n"
           "2. If the activity needs physical premises (shop, restaurant, "
           "factory), use get_balady_rules with the closest balady activity id "
           "from the preview to get the full, real rules. Check that candidate's "
           "actual match_status exactly as returned: if it is 'CONFIRMED', the "
           "link is officially confirmed and you may classify it 'verified' "
           "(unless another quality flag applies); if it is still "
           "'CANDIDATE_VERIFY' or anything else, state explicitly that it is a "
           "candidate match, not officially confirmed. Never blanket-label every "
           "link as 'candidate' or 'CANDIDATE_VERIFY' -- describe each link by "
           "its own real status only.\n"
           "3. Use get_sbc_registration_rules for both registration steps.\n"
           "4. If nationality is non-Saudi or unspecified, use get_misa_knowledge.\n"
           "5. If the activity needs a lease, use get_ejar_knowledge.\n"
           "6. For anything sensitive (fees, durations, percentages), use "
           "check_quality_flags before including it.\n\n"
           "For every requirement found, set status precisely:\n"
           "- 'verified': stated explicitly by an official source, no quality flag\n"
           "- 'unverified': stated but flagged REVIEW or UNVERIFIED\n"
           "- 'blocked': BLOCKING_FOR_AUTOMATED_ANSWERS -- never state a specific "
           "figure, just note the requirement exists and describe the conflict\n"
           "- 'missing': tools indicate MISSING_DATA\n\n"
           "Return JSON only:\n"
           "{{\"requirements\": [{{\"id\": \"...\", \"title\": \"...\", "
           "\"authority\": \"...\", \"description\": \"...\", \"status\": \"...\", "
           "\"quality_note\": \"... or null\", \"source_url\": \"... or null\"}}]}}"),
}


def _format_onboarding_answers(answers: Optional[Dict], language: Language) -> str:
    """Turn the pre-plan onboarding answers into a short context block the
    researcher can use to decide which tools to call (e.g. skip get_ejar_knowledge
    if the user already has a lease; call get_misa_knowledge only if relevant).
    Never invents values -- omits any key the user didn't actually answer."""
    if not answers:
        return ""
    labels_ar = {
        "has_lease": "لديه عقد إيجار (إيجار)؟",
        "name_type": "اسم تجاري أم اسم شخصي؟",
        "registration_started": "بدأ أي إجراء تسجيل مسبقًا؟",
        "location_ready": "لديه موقع/محل محدد؟",
    }
    labels_en = {
        "has_lease": "Has an Ejar lease contract?",
        "name_type": "Trade name or personal name?",
        "registration_started": "Already started any registration step?",
        "location_ready": "Has a specific location/premises?",
    }
    labels = labels_ar if language == "ar" else labels_en
    lines = [f"- {labels.get(k, k)}: {v}" for k, v in answers.items() if v not in (None, "")]
    if not lines:
        return ""
    header = "\n\nإجابات العميل على أسئلة ما قبل الخطة:\n" if language == "ar" else "\n\nClient's answers to the pre-plan questions:\n"
    return header + "\n".join(lines)


@traceable(name="Researcher v2", run_type="chain")
def research_agent_v2(business_idea: str, nationality: Nationality = "unspecified",
                       language: Optional[Language] = None, max_tool_rounds: int = 4,
                       cost_log: Optional[List[Dict]] = None,
                       answers: Optional[Dict] = None) -> Dict:
    cost_log = cost_log if cost_log is not None else _new_cost_log()
    language = language or detect_language(business_idea)
    label = "فكرة العمل" if language == "ar" else "Business idea"
    nat_label = "جنسية صاحب العمل" if language == "ar" else "Owner nationality"
    prompt = (_RESEARCHER_V2_SYSTEM[language] + f"\n\n{label}: {business_idea}\n{nat_label}: {nationality}"
              + _format_onboarding_answers(answers, language))

    def _first_call():
        return client.responses.create(model=GPT_TERRA, input=prompt, tools=_RESEARCHER_V2_TOOLS)

    response = call_with_retry(_first_call)
    tool_calls = 0
    for _ in range(max_tool_rounds):
        calls = [it for it in response.output if getattr(it, "type", None) == "function_call"]
        if not calls:
            break
        outputs = []
        for call in calls:
            args = json.loads(call.arguments) if call.arguments else {}
            result = _execute_researcher_v2_tool(call.name, args)
            outputs.append({"type": "function_call_output", "call_id": call.call_id,
                             "output": json.dumps(result, ensure_ascii=False)})
            tool_calls += 1

        def _next_call():
            return client.responses.create(model=GPT_TERRA, previous_response_id=response.id,
                                            input=outputs, tools=_RESEARCHER_V2_TOOLS)
        response = call_with_retry(_next_call)

    data = extract_json(response.output_text)
    data["tool_calls_made"] = tool_calls
    data.setdefault("requirements", [])
    # Fill official service links from the dataset when the model omitted them.
    for req in data["requirements"]:
        blob=(str(req.get("title", ""))+" "+str(req.get("authority", ""))+" "+str(req.get("description", ""))).lower()
        if not req.get("source_url"):
            # First try the explicit sources.json service registry. The returned URL
            # is always copied from the dataset; it is never generated by the model.
            req["source_url"] = best_source_url(blob)
            if req.get("source_url"):
                req["source_origin"] = "sources.json"
            elif any(x in blob for x in ("اسم تجاري","سجل تجاري","commercial","وزارة التجارة","المركز السعودي")):
                req["source_url"]=_service_source_url("sbc")
            elif any(x in blob for x in ("بلدي","بلدية","municip")):
                req["source_url"]=_official_url("balady","municipal")
            elif any(x in blob for x in ("استثمار","misa","investment")):
                req["source_url"]=_service_source_url("misa")
            elif any(x in blob for x in ("إيجار","ejar","lease")):
                req["source_url"]=_service_source_url("ejar")
        # Source-backed facts are verified unless a real quality flag says otherwise.
        if req.get("source_url") and req.get("status") == "unverified" and not req.get("quality_note"):
            req["status"]="verified"
    usage = response.usage
    log_cost(cost_log, "Researcher v2", "gpt_terra", usage.input_tokens, usage.output_tokens)
    return data


# --------------------------------------------------------------------------
# Verifier 1
# --------------------------------------------------------------------------
_EXPECTED_STEP_KEYWORDS = {
    "all": ["اسم تجاري", "trade name", "سجل تجاري", "commercial regist"],
    "non_saudi_only": ["استثمار", "MISA", "investment"],
}


def _deterministic_completeness_check(requirements: List[Dict], nationality: Nationality) -> List[str]:
    blob = json.dumps(requirements, ensure_ascii=False).lower()
    gaps = []
    for kw_group, label in [(_EXPECTED_STEP_KEYWORDS["all"], "تسجيل تجاري/اسم تجاري")]:
        if not any(kw.lower() in blob for kw in kw_group):
            gaps.append(f"لا يوجد متطلب يغطي: {label}")
    if nationality in ("non_saudi", "unspecified"):
        if not any(kw.lower() in blob for kw in _EXPECTED_STEP_KEYWORDS["non_saudi_only"]):
            gaps.append("لا يوجد متطلب استثماري (MISA) رغم أن الجنسية غير سعودي أو غير محددة")
    return gaps


def _deterministic_status_check(requirements: List[Dict]) -> List[Dict]:
    mismatches = []
    for req in requirements:
        title = req.get("title", "") + " " + req.get("description", "")
        claimed = req.get("status", "unverified")
        _TERM_TO_QUALITY_KEYWORD = {
            "ejar": "ejar", "إيجار": "ejar", "misa": "misa", "استثمار": "misa",
            "balady": "balady", "بلدي": "balady", "sbc": "sbc",
        }
        true_status = "verified"
        matching_flags = []
        for term, quality_keyword in _TERM_TO_QUALITY_KEYWORD.items():
            if term in title.lower():
                matching_flags.extend(get_quality_flags(quality_keyword))
        if any(f["severity"] == "BLOCKING_FOR_AUTOMATED_ANSWERS" for f in matching_flags):
            true_status = "blocked"
        elif any(f["severity"] in ("REVIEW", "UNVERIFIED") for f in matching_flags):
            true_status = "unverified"
        if true_status != "verified" and claimed == "verified":
            mismatches.append({"requirement_id": req.get("id"), "claimed_status": claimed,
                                "true_status": true_status,
                                "matching_flags": [f["reason"] for f in matching_flags]})
    return mismatches


def _deterministic_ordering_check(requirements: List[Dict]) -> List[str]:
    problems = []

    def _index_of(keyword: str) -> Optional[int]:
        for i, r in enumerate(requirements):
            if keyword.lower() in (r.get("title", "") + r.get("id", "")).lower():
                return i
        return None

    name_idx = _index_of("trade_name") or _index_of("اسم تجاري")
    cr_idx = _index_of("sole_proprietorship") or _index_of("قيد سجل")
    if name_idx is not None and cr_idx is not None and name_idx > cr_idx:
        problems.append("حجز الاسم التجاري يظهر بعد القيد بدل قبله -- يخالف الاعتماد البلوكي (BLOCKING) في sbc.dependencies")
    misa_idx = _index_of("investment") or _index_of("استثمار") or _index_of("misa")
    if misa_idx is not None and cr_idx is not None and misa_idx > cr_idx:
        problems.append("خطوة الاستثمار (MISA) تظهر بعد السجل التجاري بدل قبله -- يخالف misa.dependencies")
    return problems


_VERIFIER1_SYSTEM: Dict[Language, str] = {
    "ar": ("أنتِ مدققة أولى في مكتب استشارات أعمال. راجعنا آلياً قائمة المتطلبات "
           "ووجدنا الفحوصات أدناه بالفعل -- مهمتك فقط تحويلها لملاحظات تدقيق "
           "واضحة بالعربية للمستشار البشري، وإضافة أي ملاحظة إضافية تلاحظينها "
           "بقراءة القائمة (تكرار، تناقض داخلي، معلومة غامضة). لا تُسقطي أي "
           "فحص آلي ولا تخففي حدّته."),
    "en": ("You are the first-pass verifier at a business advisory office. "
           "Automated checks already ran and are below -- your job is only to "
           "turn them into clear Arabic/English verification notes for the "
           "human advisor, plus any extra issue you notice reading the list "
           "(duplication, internal contradiction, vague wording). Never drop "
           "or soften an automated check."),
}


@traceable(name="Verifier 1 (Research Check)", run_type="chain")
def verifier_1_agent(research_data: Dict, nationality: Nationality = "unspecified",
                      language: Optional[Language] = None,
                      cost_log: Optional[List[Dict]] = None) -> Dict:
    cost_log = cost_log if cost_log is not None else _new_cost_log()
    requirements = research_data.get("requirements", [])
    language = language or "ar"

    completeness_gaps = _deterministic_completeness_check(requirements, nationality)
    status_mismatches = _deterministic_status_check(requirements)
    ordering_problems = _deterministic_ordering_check(requirements)

    checks_blob = json.dumps({
        "completeness_gaps": completeness_gaps, "status_mismatches": status_mismatches,
        "ordering_problems": ordering_problems, "requirements_reviewed": len(requirements),
    }, ensure_ascii=False, indent=2)

    prompt = (
        _VERIFIER1_SYSTEM[language] + "\n\n"
        + ("الفحوصات الآلية:\n" if language == "ar" else "Automated checks:\n")
        + checks_blob
        + ("\n\nأعيدي JSON فقط: {{\"notes\": [\"...\"], \"overall_flag\": "
           "\"clean أو needs_review أو blocking_issues_found\"}}" if language == "ar" else
           "\n\nReturn JSON only: {{\"notes\": [\"...\"], \"overall_flag\": "
           "\"clean or needs_review or blocking_issues_found\"}}")
    )

    def _call():
        return client.responses.create(model=GPT_SOL, input=prompt)

    response = call_with_retry(_call)
    verification = extract_json(response.output_text)
    verification["completeness_gaps"] = completeness_gaps
    verification["status_mismatches"] = status_mismatches
    verification["ordering_problems"] = ordering_problems
    if not verification.get("overall_flag"):
        verification["overall_flag"] = (
            "blocking_issues_found" if (completeness_gaps or status_mismatches or ordering_problems)
            else "clean")

    usage = response.usage
    log_cost(cost_log, "Verifier 1", "gpt_sol", usage.input_tokens, usage.output_tokens)
    return {"research_data": research_data, "verification": verification}


# --------------------------------------------------------------------------
# Planner v2
# --------------------------------------------------------------------------
_PLANNER_V2_SYSTEM: Dict[Language, str] = {
    "ar": ("أنتِ مخططة أعمال في مكتب استشارات. حوّلي قائمة المتطلبات المدقَّقة "
           "أدناه لخطة عمل مرتبة زمنيًا. يجب مراعاة:\n"
           "- حقول status وquality_note وCANDIDATE_VERIFY وSOURCE_CONFLICT أدوات تدقيق داخلية فقط؛ "
           "ممنوع نسخ أي منها أو شرحها داخل title أو description أو needed أو caveat الموجّه للعميل.\n"
           "- لا تنشئي خطوة عميلة من معلومة blocked/missing غير مسنودة. إذا كان جزء من الخطوة غير مسنود "
           "احذفي ذلك الجزء فقط واحتفظي بالمعلومة المثبتة من المصدر الرسمي.\n"
           "- العميل يرى الإجراء العملي فقط: ماذا يفعل، لدى أي جهة، ماذا يحتاج، وما الخدمة الرسمية.\n"
           "- رتّبي الخطوات حسب الاعتماد المنطقي (تسجيل استثماري قبل تجاري، "
           "اسم تجاري قبل قيد السجل، سجل تجاري قبل أي ترخيص لاحق).\n"
           "- استخدمي فقط المعلومات الموجودة بالقائمة، لا تضيفي شيئًا من عندك.\n"
           "- \"needed\": انسخي أي مستندات/شروط مذكورة صراحة بوصف المتطلب أو "
           "quality_note لتلك الخطوة فقط (قائمة نصية قصيرة) -- إن لم يُذكر شيء، "
           "أعيدي قائمة فارغة []، لا تخترعي مستندًا.\n"
           "- \"dependencies\": أرقام الخطوات (order) التي يجب إنجازها قبل هذي "
           "الخطوة، مبنية على نفس منطق الترتيب أعلاه -- [] إن لم توجد.\n"
           "- \"source_url\": انسخي حقل source_url من المتطلب الأصلي بالضبط كما "
           "ورد -- لا تكتبي رابطًا من عندك أبدًا، وإن كان source_url في "
           "المتطلب null فأعيديه null.\n\n"
           "أعيدي JSON فقط:\n"
           "{{\"plan\": [{{\"order\": 1, \"title\": \"...\", \"authority\": \"...\", "
           "\"description\": \"...\", \"status\": \"verified أو unverified أو blocked\", "
           "\"caveat\": \"... أو null\", \"needed\": [\"...\"], "
           "\"dependencies\": [1, 2], \"source_url\": \"... أو null\"}}]}}"),
    "en": ("You are a business planner at an advisory office. Turn the "
           "verified requirements list below into a time-ordered action plan. "
           "You must respect:\n"
           "- Any requirement with status='blocked' becomes a step saying "
           "'requires direct confirmation from the official authority', citing "
           "the conflict from quality_note -- never state a definitive figure.\n"
           "- Any requirement with status='unverified' or 'missing' is clearly "
           "marked as unconfirmed in the step's description.\n"
           "- Order steps by real dependency (investment registration before "
           "commercial registration for foreign investors; trade name before "
           "CR; CR before any downstream license).\n"
           "- Use only the information in the list -- add nothing of your own.\n"
           "- \"needed\": copy any documents/prerequisites explicitly stated in "
           "that requirement's description or quality_note only (short string "
           "list) -- if nothing is stated, return an empty list [], never "
           "invent a document.\n"
           "- \"dependencies\": the order numbers of steps that must be done "
           "before this one, following the same ordering logic above -- [] if "
           "none.\n"
           "- \"source_url\": copy the source_url field from the original "
           "requirement exactly as given -- never write a link of your own, "
           "and if the requirement's source_url is null, return null.\n\n"
           "Return JSON only:\n"
           "{{\"plan\": [{{\"order\": 1, \"title\": \"...\", \"authority\": \"...\", "
           "\"description\": \"...\", \"status\": \"verified, unverified, or blocked\", "
           "\"caveat\": \"... or null\", \"needed\": [\"...\"], "
           "\"dependencies\": [1, 2], \"source_url\": \"... or null\"}}]}}"),
}


@traceable(name="Planner v2", run_type="chain")
def planner_agent_v2(verifier_1_output: Dict, business_idea: str,
                      language: Optional[Language] = None,
                      cost_log: Optional[List[Dict]] = None) -> Dict:
    cost_log = cost_log if cost_log is not None else _new_cost_log()
    language = language or detect_language(business_idea)
    research_data = verifier_1_output["research_data"]
    verification = verifier_1_output["verification"]

    if language == "ar":
        prompt = (_PLANNER_V2_SYSTEM[language]
                  + f"\n\nفكرة العمل: {business_idea}\n\nالمتطلبات:\n"
                  + json.dumps(research_data.get("requirements", []), ensure_ascii=False, indent=2)
                  + "\n\nتقرير التدقيق الأول:\n"
                  + json.dumps(verification, ensure_ascii=False, indent=2))
    else:
        prompt = (_PLANNER_V2_SYSTEM[language]
                  + f"\n\nBusiness idea: {business_idea}\n\nRequirements:\n"
                  + json.dumps(research_data.get("requirements", []), ensure_ascii=False, indent=2)
                  + "\n\nFirst verification report:\n"
                  + json.dumps(verification, ensure_ascii=False, indent=2))

    def _call():
        return client.responses.create(model=GPT_SOL, input=prompt)

    response = call_with_retry(_call)
    data = extract_json(response.output_text)
    data.setdefault("plan", [])

    # Deterministic guard, same pattern as _deterministic_status_check above:
    # a real source_url may only be one that actually exists in the requirements
    # list. If the model wrote a URL that isn't in that set, it's a fabrication --
    # drop it to null rather than ship an invented government link.
    real_urls = ({r.get("source_url") for r in research_data.get("requirements", []) if r.get("source_url")} | {x.get("url") for x in SOURCE_CATALOG if x.get("url")})
    for step in data["plan"]:
        step.setdefault("needed", [])
        step.setdefault("dependencies", [])
        if step.get("source_url") and step["source_url"] not in real_urls:
            step["source_url"] = None
        else:
            step.setdefault("source_url", None)

    # Deterministic dependency pass based on the explicit dataset dependency graph.
    title_to_order = [(st.get("order"), (st.get("title","")+" "+st.get("description","")).lower()) for st in data["plan"]]
    def find_order(words):
        for order, blob in title_to_order:
            if any(w.lower() in blob for w in words): return order
        return None
    trade=find_order(["اسم تجاري","trade name"]); cr=find_order(["سجل تجاري","commercial registration","قيد سجل"]); inv=find_order(["وزارة الاستثمار","misa","تسجيل استثماري"]); bal=find_order(["رخصة بلدية","ترخيص بلدي","بلدي"]); lease=find_order(["إيجار","عقد إيجار","lease"])
    for st in data["plan"]:
        deps=set(st.get("dependencies") or [])
        o=st.get("order")
        if o==cr and trade: deps.add(trade)
        if o==cr and inv: deps.add(inv)
        if o==bal and cr: deps.add(cr)
        if o==bal and lease: deps.add(lease)
        st["dependencies"]=sorted(d for d in deps if isinstance(d,int) and d!=o)

    usage = response.usage
    log_cost(cost_log, "Planner v2", "gpt_sol", usage.input_tokens, usage.output_tokens)
    return data



# --------------------------------------------------------------------------
# Client-safe plan projection
# --------------------------------------------------------------------------
_INTERNAL_TOKENS = (
    "CANDIDATE_VERIFY", "SOURCE_CONFLICT", "NEAR_VARIATION", "NEAR VARIATION",
    "BLOCKING_FOR_AUTOMATED_ANSWERS", "UNVERIFIED", "REVIEW", "quality_note",
)

def _clean_client_text(value: Any) -> str:
    """Remove internal pipeline/debug vocabulary from client-facing strings."""
    if value is None:
        return ""
    text = str(value)
    # Remove whole sentences/lines that leak internal verification machinery.
    chunks = re.split(r"(?<=[.!؟\n])\s+", text)
    kept = []
    for chunk in chunks:
        upper = chunk.upper()
        if any(tok.upper() in upper for tok in _INTERNAL_TOKENS):
            continue
        # Also suppress code-like activity matching commentary.
        if re.search(r"\bSBC\s*\d{4,}\b", chunk, flags=re.I) and any(
            w in chunk for w in ("ربط", "تصنيف", "مرشح", "مطابق")
        ):
            continue
        kept.append(chunk.strip())
    out = " ".join(x for x in kept if x).strip()
    out = re.sub(r"\s{2,}", " ", out)
    return out

def _client_safe_plan(plan_data: Dict) -> Dict:
    """Return only user-useful fields; keep verification machinery internal."""
    public = []
    for raw in plan_data.get("plan", []):
        title = _clean_client_text(raw.get("title"))
        desc = _clean_client_text(raw.get("description"))
        needed = [_clean_client_text(x) for x in (raw.get("needed") or [])]
        needed = [x for x in needed if x]
        url = raw.get("source_url")
        # Do not expose unsupported/debug-only steps. A client-facing regulatory
        # step must be tied to a URL copied from the project's official source data.
        if not title or not url:
            continue
        public.append({
            "order": len(public) + 1,
            "title": title,
            "authority": _clean_client_text(raw.get("authority")),
            "description": desc,
            "needed": needed,
            "dependencies": list(raw.get("dependencies") or []),
            "source_url": url,
        })
    # Re-map dependencies after filtered steps. Dependencies are presentation
    # helpers only; if a referenced internal step disappeared, drop that edge.
    old_to_new = {}
    for new_i, raw in enumerate([x for x in plan_data.get("plan", []) if x.get("source_url")], 1):
        old_to_new[raw.get("order")] = new_i
    for step in public:
        step["dependencies"] = sorted({
            old_to_new[d] for d in step["dependencies"] if d in old_to_new
        })
    return {"plan": public}

# --------------------------------------------------------------------------
# Verifier 2
# --------------------------------------------------------------------------
def _deterministic_plan_grounding_check(plan: List[Dict], requirements: List[Dict]) -> Dict:
    req_titles = {r.get("id"): r.get("title", "") for r in requirements}
    req_status = {r.get("id"): r.get("status", "unverified") for r in requirements}
    plan_blob = " ".join(s.get("title", "") + s.get("description", "") for s in plan).lower()

    dropped_requirements = []
    for req_id, title in req_titles.items():
        key_fragment = title[:15].lower() if title else ""
        if key_fragment and key_fragment not in plan_blob:
            dropped_requirements.append({"id": req_id, "title": title})

    status_downgrades = []
    for step in plan:
        step_status = step.get("status")
        for req_id, title in req_titles.items():
            if title[:15] and title[:15].lower() in (step.get("title", "") + step.get("description", "")).lower():
                true_status = req_status[req_id]
                if true_status in ("blocked", "missing") and step_status == "verified":
                    status_downgrades.append({
                        "plan_step_title": step.get("title"), "requirement_id": req_id,
                        "requirement_status": true_status, "plan_claimed_status": step_status,
                    })
    return {"dropped_requirements": dropped_requirements, "status_downgrades": status_downgrades}


_VERIFIER2_SYSTEM: Dict[Language, str] = {
    "ar": ("أنتِ مدققة ثانية، تراجعين الخطة النهائية المبنية من قِبل المخططة قبل "
           "ما توصل للعميل. فحوصات آلية جرت بالفعل (أدناه) تقارن الخطة "
           "بالمتطلبات المصدرية -- حوّليها لملاحظات واضحة، وأضيفي أي ملاحظة عن "
           "تناسق الخطة نفسها (ترتيب منطقي، وضوح الصياغة، عدم وجود لغة تسويقية "
           "مبالغ فيها أو وعود غير مؤكدة). لا تُسقطي أي فحص آلي."),
    "en": ("You are the second verifier, reviewing the final plan the Planner "
           "built before it reaches the client. Automated checks already ran "
           "(below) comparing the plan to its source requirements -- turn them "
           "into clear notes, plus anything you notice about the plan's own "
           "coherence (logical order, clarity, no overpromising or unconfirmed "
           "guarantees). Never drop an automated check."),
}


@traceable(name="Verifier 2 (Plan Check)", run_type="chain")
def verifier_2_agent(plan_data: Dict, verifier_1_output: Dict,
                      language: Optional[Language] = None,
                      cost_log: Optional[List[Dict]] = None) -> Dict:
    cost_log = cost_log if cost_log is not None else _new_cost_log()
    language = language or "ar"
    plan = plan_data.get("plan", [])
    requirements = verifier_1_output["research_data"].get("requirements", [])

    grounding = _deterministic_plan_grounding_check(plan, requirements)

    prompt = (
        _VERIFIER2_SYSTEM[language] + "\n\n"
        + ("الخطة:\n" if language == "ar" else "The plan:\n")
        + json.dumps(plan, ensure_ascii=False, indent=2)
        + ("\n\nفحوصات آلية (مطابقة الخطة بالمتطلبات المصدرية):\n" if language == "ar"
           else "\n\nAutomated checks (plan vs. source requirements):\n")
        + json.dumps(grounding, ensure_ascii=False, indent=2)
        + ("\n\nأعيدي JSON فقط: {{\"notes\": [\"...\"], \"overall_flag\": "
           "\"clean أو needs_review أو blocking_issues_found\"}}" if language == "ar" else
           "\n\nReturn JSON only: {{\"notes\": [\"...\"], \"overall_flag\": "
           "\"clean, needs_review, or blocking_issues_found\"}}")
    )

    def _call():
        return client.responses.create(model=GPT_SOL, input=prompt)

    response = call_with_retry(_call)
    verification = extract_json(response.output_text)
    verification["dropped_requirements"] = grounding["dropped_requirements"]
    verification["status_downgrades"] = grounding["status_downgrades"]
    if not verification.get("overall_flag"):
        verification["overall_flag"] = (
            "blocking_issues_found"
            if (grounding["dropped_requirements"] or grounding["status_downgrades"]) else "clean")

    usage = response.usage
    log_cost(cost_log, "Verifier 2", "gpt_sol", usage.input_tokens, usage.output_tokens)
    return {"plan_data": plan_data, "verification": verification}


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
_ORCH_V2_PRINT_SYSTEM: Dict[Language, str] = {
    "ar": ("أنتِ مستشارة أعمال أولى في نواة للاستشارات. اكتبي مذكرة استشارية "
           "احترافية ونهائية للعميل من الخطة المدقَّقة مرتين أدناه (تدقيق "
           "البيانات وتدقيق الخطة). قواعد صارمة:\n"
           "- استخدمي فقط ما ورد بالخطة -- لا تخترعي رقمًا أو تفصيلًا.\n"
           "- الخطة التي تصلك هنا هي النسخة العامة المنقحة؛ لا تعرضي أي حالات تدقيق داخلية أو "
           "أكواد تقنية أو ملاحظات تحقق.\n"
           "- اكتبي فقط الإجراءات العملية وروابط الخدمات الرسمية الموجودة في الخطة.\n"
           "- أسلوب مكتب استشارات بشري محترف تمامًا -- ممنوع أي عبارة تكشف "
           "كونك نموذج ذكاء اصطناعي.\n"
           "- رتّبي المذكرة بعناوين واضحة: نظرة عامة، خطوات التأسيس مرتبة، "
           "ملاحظات تحتاج تأكيدًا، التوقيع الختامي."),
    "en": ("You are a senior advisor at Nawah Advisory. Write a final, "
           "professional advisory memo for the client from the twice-verified "
           "plan below (data verification and plan verification). Strict rules:\n"
           "- Use only what's in the plan -- never invent a figure or detail.\n"
           "- Any step with status='blocked' is written clearly as an item "
           "requiring direct confirmation from the authority, citing the "
           "conflict calmly and professionally, without alarm.\n"
           "- Any step with status='unverified' is marked with wording like "
           "'per the latest available information; direct confirmation is "
           "advised'.\n"
           "- Write entirely like a professional human advisory office -- "
           "never reveal you are an AI model.\n"
           "- Structure the memo with clear headers: overview, ordered setup "
           "steps, items requiring confirmation, closing signature."),
}


@traceable(name="Orchestrator v2 - Print Plan", run_type="chain")
def orchestrator_v2_print_plan(business_idea: str, verifier_2_output: Dict,
                                language: Optional[Language] = None,
                                cost_log: Optional[List[Dict]] = None) -> str:
    cost_log = cost_log if cost_log is not None else _new_cost_log()
    language = language or detect_language(business_idea)
    plan_data = verifier_2_output["plan_data"]
    verification_2 = verifier_2_output["verification"]

    prompt = (
        _ORCH_V2_PRINT_SYSTEM[language]
        + f"\n\n{'فكرة العمل' if language == 'ar' else 'Business idea'}: {business_idea}"
        + f"\n\n{'الخطة' if language == 'ar' else 'Plan'}:\n"
        + json.dumps(plan_data.get("plan", []), ensure_ascii=False, indent=2)
        + f"\n\n{'تقرير التدقيق الثاني' if language == 'ar' else 'Second verification report'}:\n"
        + json.dumps(verification_2, ensure_ascii=False, indent=2)
    )

    def _call():
        return client.responses.create(model=GPT_LUNA, input=prompt)

    response = call_with_retry(_call)
    usage = response.usage
    log_cost(cost_log, "Orchestrator v2 (print)", "gpt_luna", usage.input_tokens, usage.output_tokens)
    return response.output_text


def run_nawah_pipeline_v3(business_idea: str, nationality: Nationality = "unspecified",
                           language: Optional[Language] = None,
                           answers: Optional[Dict] = None) -> Dict:
    """Fast production path: one grounded research agent + one planning agent.
    Verification metadata stays deterministic/internal to avoid extra model round trips.
    The business scope is intentionally open and is inferred from business_idea.
    """
    cost_log = _new_cost_log()
    language = language or detect_language(business_idea)

    research_data = research_agent_v2(
        business_idea, nationality, language, max_tool_rounds=4,
        cost_log=cost_log, answers=answers
    )

    # Fast deterministic verification: no extra LLM call.
    reqs = research_data.get("requirements", [])
    verification = {
        "notes": [],
        "overall_flag": "internal_checks_complete",
        "completeness_gaps": _deterministic_completeness_check(reqs, nationality),
        "status_mismatches": _deterministic_status_check(reqs),
        "ordering_problems": _deterministic_ordering_check(reqs),
    }
    v1_output = {"research_data": research_data, "verification": verification}
    plan_data = planner_agent_v2(v1_output, business_idea, language, cost_log=cost_log)
    public_plan = _client_safe_plan(plan_data)

    # The UI only needs the actionable plan. Avoid a third/fourth model call to
    # write a memo the user does not want to see.
    return {
        "final_text": "",
        "language": language,
        "research": {"requirements_count": len(reqs), "tool_calls_made": research_data.get("tool_calls_made", 0)},
        "verifier_1": {"overall_flag": "internal_checks_complete"},
        "plan": public_plan,
        "internal_plan": plan_data,
        "verifier_2": {"overall_flag": "skipped_fast_path"},
        "cost_log": cost_log,
        "cost_total_sar": round(sum(e["cost_sar"] for e in cost_log), 4),
        "data_inventory": data_inventory(),
    }


# --------------------------------------------------------------------------
# Conversation layer -- session-aware follow-up handling
# --------------------------------------------------------------------------
Intent = Literal["new_business_request", "specific_question", "greeting_or_unclear"]

_QA_TOOLS_V3 = _RESEARCHER_V2_TOOLS + [{"type": "web_search"}]

_INTENT_SYSTEM: Dict[Language, str] = {
    "ar": ("أنتِ مصنّفة نوايا في مكتب استشارات أعمال. صنّفي رسالة العميل إلى "
           "واحدة فقط: new_business_request (فكرة مشروع جديدة أو مختلفة)، "
           "specific_question (سؤال محدد، بملف قائم أو بدونه)، أو "
           "greeting_or_unclear (تحية أو رسالة غامضة). أعيدي JSON فقط: "
           "{{\"intent\": \"...\"}}"),
    "en": ("You are the intent classifier at a business advisory office. "
           "Classify the client's message as exactly one of: "
           "new_business_request, specific_question, or greeting_or_unclear. "
           "Return JSON only: {{\"intent\": \"...\"}}"),
}


def _case_summary(session: Dict) -> str:
    if not session.get("has_case"):
        return "-"
    return f"{session.get('business_idea')} | {session.get('nationality')}"


def classify_intent(message: str, session: Dict, language: Language,
                     cost_log: List[Dict]) -> Intent:
    prompt = (_INTENT_SYSTEM[language]
              + f"\n\nhas_case: {bool(session.get('has_case'))}\ncase: {_case_summary(session)}"
              + f"\n\nmessage: «{message}»")

    def _call():
        return client.responses.create(model=GPT_LUNA, input=prompt)

    response = call_with_retry(_call)
    usage = response.usage
    log_cost(cost_log, "Intent classifier", "gpt_luna", usage.input_tokens, usage.output_tokens)
    intent = extract_json(response.output_text).get("intent")
    return intent if intent in ("new_business_request", "specific_question",
                                 "greeting_or_unclear") else "specific_question"


_QA_SYSTEM: Dict[Language, str] = {
    "ar": ("أنت مساعد نواة للمعاملات والتراخيص داخل السعودية. مهمتك إجابة السؤال الحالي مباشرة "
           "باستخدام سياق المحادثة وبيانات نواة الرسمية. لا تتصرف كموظف استقبال ولا تكتب مقدمات تسويقية.\n\n"
           "قواعد واجهة المستخدم:\n"
           "- تذكّر النشاط والمدينة والموضوع من سياق الجلسة. إذا قال المستخدم لاحقًا (بيطري) فهذا يحدد النشاط للسؤال السابق، ولا تسأله عنه مرة أخرى.\n"
           "- ابدأ بالجواب، بحد أقصى 4-7 أسطر قصيرة ما لم يطلب المستخدم التفصيل.\n"
           "- إذا وُجد رابط خدمة رسمي في نتائج الأدوات، اختم بسطر: رابط الخدمة: URL\n"
           "- لا تعرض إطلاقًا أسماء أعلام الجودة أو أكواد النظام أو التصنيفات الداخلية مثل BLOCKING وREVIEW وCANDIDATE_VERIFY وSOURCE_CONFLICT وNEAR_VARIATION أو أكواد SBC.\n"
           "- لا تطبع رموز استشهاد داخلية مثل turn? أو cite أو أسماء الأدوات.\n"
           "- لا تقل: شكرًا لتواصلكم، فريق نواة، مستشار حقيقي، ملف تأسيس، أو نحن نساعدك؛ هذه محادثة عملية مباشرة.\n"
           "- لا تسأل سؤال متابعة إذا كان الجواب ممكنًا من السياق الحالي.\n"
           "- لا تخترع شرطًا أو رسمًا أو رابطًا. إذا لم تدعم البيانات تفصيلًا دقيقًا، اذكر فقط الجزء المدعوم دون كشف ملاحظات التدقيق الداخلية."),
    "en": ("You are Nawah's Saudi transactions and licensing assistant. Answer the current question directly using conversation context and Nawah's official-source data. Keep the answer concise, preserve activity/city/topic context, never expose internal quality flags, tool names, codes, or citation tokens, and never invent requirements, fees, or URLs."),
}


def _clean_public_chat(text: str) -> str:
    """Final UI firewall: internal reasoning/QA labels must never reach users."""
    if not text:
        return text
    # Remove bogus/internal citation tokens produced as plain model text.
    text = re.sub(r"(?:cite|filecite)[^]*", "", text)
    text = re.sub(r"\\?\[?\s*turn\??[^\]\s]*\s*\]?", "", text, flags=re.I)
    # Remove lines that expose internal verification vocabulary.
    banned = re.compile(r"\b(?:BLOCKING|REVIEW|CANDIDATE_VERIFY|SOURCE_CONFLICT|NEAR_VARIATION|UNVERIFIED)\b", re.I)
    lines = [ln for ln in text.splitlines() if not banned.search(ln)]
    text = "\n".join(lines).strip()
    # Strip stale advisory-office signatures/greetings.
    text = re.sub(r"(?im)^\s*(?:شكرًا لتواصلكم.*|فريق نواة.*)\s*$", "", text).strip()
    return text


@traceable(name="Nawah Licensing Chat Fast", run_type="chain")
def answer_follow_up(message: str, session: Dict, language: Language,
                      max_tool_rounds: int = 1, cost_log: Optional[List[Dict]] = None) -> str:
    cost_log = cost_log if cost_log is not None else _new_cost_log()
    context = {
        "topic": session.get("chat_topic"),
        "activity": session.get("chat_activity"),
        "city": session.get("chat_city"),
        "business_idea": session.get("business_idea"),
    }
    prompt = _QA_SYSTEM[language] + "\n\nسياق محفوظ: " + json.dumps(context, ensure_ascii=False) + f"\nالسؤال الحالي: «{message}»"

    def _first_call():
        return client.responses.create(model=GPT_LUNA, input=prompt, tools=_QA_TOOLS_V3)

    response = call_with_retry(_first_call)
    # One tool round maximum: enough for local official-source lookup, avoids long agent loops.
    calls = [it for it in response.output if getattr(it, "type", None) == "function_call"]
    if calls:
        outputs = []
        for call in calls[:4]:
            args = json.loads(call.arguments) if call.arguments else {}
            result = _execute_researcher_v2_tool(call.name, args)
            outputs.append({"type": "function_call_output", "call_id": call.call_id,
                            "output": json.dumps(result, ensure_ascii=False)})
        def _next_call():
            return client.responses.create(model=GPT_LUNA, previous_response_id=response.id,
                                            input=outputs, tools=_QA_TOOLS_V3)
        response = call_with_retry(_next_call)

    usage = response.usage
    log_cost(cost_log, "Licensing chat", "gpt_luna", usage.input_tokens, usage.output_tokens)
    return _clean_public_chat(response.output_text)


_GREETING_SYSTEM: Dict[Language, str] = {
    "ar": ("أنتِ منسّقة استقبال في نواة للاستشارات. اكتبي ردًا قصيرًا (سطرين) "
           "ومهنيًا وغير مكرر الصياغة، توضّح إنك تقدرين تساعدين إما بإعداد ملف "
           "تأسيس كامل أو بالإجابة على سؤال محدد. ممنوع الإشارة لكونك ذكاء اصطناعي."),
    "en": ("You are the intake coordinator at Nawah Advisory. Write a short "
           "(two-line), professional, non-repetitive reply explaining you can "
           "either prepare a full setup file or answer a specific question. "
           "Never reveal you are an AI model."),
}


def greet_or_ask(message: str, language: Language, cost_log: List[Dict]) -> str:
    prompt = _GREETING_SYSTEM[language] + f"\n\nmessage: {message}"

    def _call():
        return client.responses.create(model=GPT_LUNA, input=prompt)

    response = call_with_retry(_call)
    usage = response.usage
    log_cost(cost_log, "Greeting", "gpt_luna", usage.input_tokens, usage.output_tokens)
    return response.output_text


_EXTRACT_SYSTEM: Dict[Language, str] = {
    "ar": ("استخرجي من رسالة العميل: business_idea (أو null)، nationality "
           "(saudi/non_saudi/unspecified)، missing_business_idea (true/false). "
           "أعيدي JSON فقط."),
    "en": ("Extract from the client's message: business_idea (or null), "
           "nationality (saudi/non_saudi/unspecified), missing_business_idea "
           "(true/false). Return JSON only."),
}

_CLARIFY: Dict[Language, str] = {
    "ar": ("شكرًا لتواصلكم مع نواة للاستشارات.\n\nلنعدّ ملف تأسيس دقيق لمشروعكم، "
           "نرجو تزويدنا بوصف مبسّط لنشاطكم التجاري المقترح.\n\nفريق نواة للاستشارات"),
    "en": ("Thank you for reaching out to Nawah Advisory.\n\nTo prepare an "
           "accurate setup file, kindly share a brief description of your "
           "intended business activity.\n\nThe Nawah Advisory Team"),
}


def new_session() -> Dict:
    return {"has_case": False}


@traceable(name="Nawah Conversation Turn v3", run_type="chain")
def nawah_turn(message: str, session: Optional[Dict] = None) -> Tuple[str, Dict, List[Dict]]:
    """Same behaviour as the notebook's `nawah_turn`, plus it returns the
    per-turn `cost_log` as a third element (handy for an API response)."""
    cost_log = _new_cost_log()
    session = dict(session) if session else new_session()
    language = detect_language(message)
    intent = classify_intent(message, session, language, cost_log)

    if intent == "greeting_or_unclear":
        return greet_or_ask(message, language, cost_log), session, cost_log

    if intent == "specific_question":
        return answer_follow_up(message, session, language, cost_log=cost_log), session, cost_log

    # new_business_request
    extract_prompt = _EXTRACT_SYSTEM[language] + f"\n\nmessage: «{message}»"

    def _extract_call():
        return client.responses.create(model=GPT_LUNA, input=extract_prompt)

    extract_response = call_with_retry(_extract_call)
    usage = extract_response.usage
    log_cost(cost_log, "Intake extraction", "gpt_luna", usage.input_tokens, usage.output_tokens)
    intake = extract_json(extract_response.output_text)

    if intake.get("missing_business_idea") or not intake.get("business_idea"):
        return _CLARIFY[language], session, cost_log

    nationality: Nationality = intake.get("nationality") or "unspecified"
    if nationality not in ("saudi", "non_saudi", "unspecified"):
        nationality = "unspecified"

    result = run_nawah_pipeline_v3(intake["business_idea"], nationality, language)
    cost_log.extend(result["cost_log"])

    session.update({
        "has_case": True,
        "business_idea": intake["business_idea"],
        "nationality": nationality,
        "memo_text": result["final_text"],
        "plan": result["plan"].get("plan", []),
    })
    return result["final_text"], session, cost_log


def _update_chat_context(message: str, session: Dict) -> None:
    """Cheap deterministic memory for licensing chat; no extra model call."""
    m = message.strip()
    # Topics are deliberately broad; preserve the last explicit one.
    topic_terms = ("رخص", "ترخيص", "سجل", "اسم تجاري", "زكاة", "ضريبة", "فاتور", "بلدي", "شهادة صحية", "تأمينات", "قوى", "استثمار")
    if any(t in m for t in topic_terms):
        session["chat_topic"] = m
    # City memory when explicitly mentioned.
    for city in ("الرياض", "جدة", "مكة", "مكة المكرمة", "المدينة", "المدينة المنورة", "الدمام", "الخبر", "الظهران", "الطائف", "أبها", "تبوك", "بريدة", "حائل", "جازان", "نجران"):
        if city in m:
            session["chat_city"] = city
            break
    # A short non-question answer after a saved topic is normally the activity,
    # e.g. topic='إصدار الرخصة البلدية' then message='بيطري'.
    questionish = any(x in m for x in ("؟", "?", "كيف", "وش", "ما ", "هل", "كم", "وين", "متى"))
    if session.get("chat_topic") and not questionish and len(m) <= 80 and not any(t in m for t in topic_terms):
        session["chat_activity"] = m
    # Explicit activity phrases also update memory.
    mm = re.search(r"(?:لنشاط|نشاط|لمشروع|مشروع)\s+([^،,.؟?]{2,60})", m)
    if mm:
        session["chat_activity"] = mm.group(1).strip()


def nawah_chat_api(message: str, session: Optional[Dict] = None) -> Dict:
    """Fast context-aware licensing chat. It never launches the full business-plan pipeline."""
    cost_log = _new_cost_log()
    session = dict(session) if session else {"has_case": False}
    _update_chat_context(message, session)
    language = detect_language(message)

    # If user supplied only the missing activity, combine it with the saved topic.
    effective = message
    if session.get("chat_activity") == message.strip() and session.get("chat_topic"):
        effective = f"{session['chat_topic']} — النشاط: {session['chat_activity']}"

    reply = answer_follow_up(effective, session, language, max_tool_rounds=1, cost_log=cost_log)
    return {"reply": reply, "session": session,
            "cost_total_sar": round(sum(e["cost_sar"] for e in cost_log), 4)}


# --------------------------------------------------------------------------
# Location advisor -- for users who don't yet have a specific premises
# --------------------------------------------------------------------------
# Unlike every other agent above, this one is NOT grounded in the audited
# knowledge_2.json/service_knowledge dataset -- there is no verified rent,
# foot-traffic, or competition data in this project. It uses live web_search
# instead, and its whole job is to stay honest about that difference: general,
# directional, source-attributed guidance -- never a confident "go here"
# verdict dressed up as data, because that's exactly the kind of fabricated
# certainty this project is built to avoid.
_LOCATION_SYSTEM: Dict[Language, str] = {
    "ar": ("أنتِ محلّلة سوق تساعدين رائدة أعمال ما عندها موقع محدد بعد لمشروعها. "
           "استخدمي أداة web_search لجمع معلومات منشورة وحديثة عن أحياء/مناطق "
           "مناسبة لنشاطها بالمدينة المذكورة (كثافة سكانية، حركة مشاة، تركّز "
           "أنشطة مشابهة أو منافسة، تقارير إيجارات تجارية إن وُجدت). ابحثي أيضًا "
           "بشكل صريح في موقع عقار sa.aqar.fm عن محلات/مكاتب تجارية معروضة في "
           "الأحياء المرشحة، وفضّلي رابط عقار حديثًا كـ source_url عندما توجد نتيجة مناسبة.\n\n"
           "قواعد صارمة:\n"
           "- كل منطقة تقترحينها لازم مبنية على نتيجة بحث فعلية.\n"
           "- ممنوع تختلقي رقم إيجار أو نسبة منافسة أو كثافة إن ما لقيتيه فعليًا "
           "بالبحث.\n"
           "- هذا استرشاد عام لتضييق الخيارات، مو قرارًا نهائيًا.\n"
           "- إن لم تجدي معلومات كافية عن المدينة/النشاط، قولي ذلك بصراحة بدل "
           "التخمين، وأعيدي areas كقائمة فاضية مع تعليل بالحقل note.\n\n"
           "أعيدي JSON فقط بهذا الشكل بالضبط (بدون Markdown، بدون روابط مطوّلة "
           "داخل النص، اسم المصدر فقط بدون رابط utm أو تتبّع):\n"
           "{{\"areas\": [{{\"name\": \"اسم الحي/المنطقة\", \"why\": \"سطر أو "
           "سطرين يشرحان السبب فقط\", \"source_name\": \"اسم الموقع أو التقرير\", "
           "\"source_url\": \"الرابط الأساسي بدون معاملات تتبع، أو null\"}}], "
           "\"note\": \"ملاحظة عامة أو سبب عدم وجود نتائج كافية، أو null\"}}"),
    "en": ("You are a market analyst helping an entrepreneur who doesn't have "
           "a specific location yet. Use the web_search tool to gather recent, "
           "published information about neighborhoods/areas in the stated city "
           "that suit their activity (population density, foot traffic, "
           "clustering of similar or competing businesses, commercial rent "
           "reports if available). Also explicitly search sa.aqar.fm for current "
           "commercial shops/offices in candidate areas, and prefer a recent Aqar "
           "listing/search page as source_url when a relevant result exists.\n\n"
           "Strict rules:\n"
           "- Every area you suggest must be based on an actual search result.\n"
           "- Never invent a rent figure, competition percentage, or density "
           "number you didn't actually find.\n"
           "- This is general guidance to narrow options, not a final verdict.\n"
           "- If you can't find enough information for that city/activity, say "
           "so plainly instead of guessing, and return an empty areas list with "
           "a note explaining why.\n\n"
           "Return JSON only, in exactly this shape (no Markdown, no tracking "
           "parameters in URLs, just the plain source name and a clean link):\n"
           "{{\"areas\": [{{\"name\": \"neighborhood/area name\", \"why\": "
           "\"one or two plain sentences explaining why\", \"source_name\": "
           "\"site or report name\", \"source_url\": \"the base link with no "
           "tracking params, or null\"}}], \"note\": \"general caveat, or why "
           "results were thin, or null\"}}"),
}

_LOCATION_DISCLAIMER: Dict[Language, str] = {
    "ar": "اقتراحات استرشادية عامة مبنية على بحث عام، لا على بيانات إيجار أو منافسة مدقّقة. تحققي ميدانيًا من توفر عقار مناسب واشتراطات بلدي النشاط في المنطقة قبل أي التزام.",
    "en": "General guidance based on public search results, not audited rent or competition data. Verify on the ground that suitable premises exist and confirm the activity's municipal requirements in the area before committing to anything.",
}


@traceable(name="Location Advisor", run_type="chain")
def location_advisor_agent(business_idea: str, city_hint: Optional[str] = None,
                            language: Optional[Language] = None,
                            max_tool_rounds: int = 2,
                            cost_log: Optional[List[Dict]] = None) -> Dict:
    """Returns *structured* area suggestions (name/why/source) instead of a
    freeform paragraph, so the frontend can render clean cards -- and, where
    the area name is a known district, plot it on a map -- rather than
    dumping raw model text with inline citation links in it."""
    cost_log = cost_log if cost_log is not None else _new_cost_log()
    language = language or detect_language(business_idea)
    city_line = f"\nCity: {city_hint}" if city_hint else ""
    prompt = _LOCATION_SYSTEM[language] + f"\n\nBusiness idea: {business_idea}{city_line}"

    def _first_call():
        return client.responses.create(model=GPT_TERRA, input=prompt, tools=[{"type": "web_search"}])

    response = call_with_retry(_first_call)
    for _ in range(max_tool_rounds):
        calls = [it for it in response.output if getattr(it, "type", None) == "function_call"]
        if not calls:
            break

        def _next_call():
            return client.responses.create(model=GPT_TERRA, previous_response_id=response.id,
                                            input=[], tools=[{"type": "web_search"}])
        response = call_with_retry(_next_call)

    usage = response.usage
    log_cost(cost_log, "Location Advisor", "gpt_terra", usage.input_tokens, usage.output_tokens)

    try:
        parsed = extract_json(response.output_text)
        areas = parsed.get("areas") or []
        note = parsed.get("note")
    except (json.JSONDecodeError, ValueError):
        # Model didn't return clean JSON -- degrade to a single freeform
        # note instead of shipping broken/partial JSON to the frontend.
        areas = []
        note = response.output_text

    return {
        "areas": areas,
        "note": note,
        "disclaimer": _LOCATION_DISCLAIMER[language],
        # Kept for older frontend builds that still read plain text.
        "suggestions": response.output_text,
        "cost_total_sar": round(sum(e["cost_sar"] for e in cost_log), 4),
    }