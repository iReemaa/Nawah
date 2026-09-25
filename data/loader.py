"""
Loads the consolidated Nawah dataset and enforces the review gate described
in quality_issues.json / nawah_consolidated_dataset/consolidated/README.md.

This is deliberately NOT a semantic-search / embeddings layer: the source
data is already structured (activity IDs, ISIC codes, rule records), so a
deterministic keyed lookup is both cheaper and more auditable than RAG over
raw text. A hook for the pre-built FAISS index is included for later.

Hard rules encoded here (do not relax without re-reading quality_issues.json):
  1. service_knowledge["ejar"]["fees"] is BLOCKING_FOR_AUTOMATED_ANSWERS —
     never surfaced as evidence, ever.
  2. Anything under a `unverified_rules` key is excluded from authoritative
     evidence.
  3. service_knowledge["misa"]["unresolved_for_nawah"] entries are surfaced
     only as needs_verification=True, never as confirmed requirements.
  4. A Balady rule with an empty `urls` list is kept but marked
     needs_verification=True (73,454 / 81,380 original records have no
     record-level URL per manifest.json).
  5. activity_crosswalk_candidates.json links are NEVER auto-merged between
     Balady and SBC activities — surfaced only as a labelled suggestion.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import SETTINGS

DATA_DIR = Path(SETTINGS.data_dir)
CORRECTIONS_DIR = Path(SETTINGS.corrections_dir)

# ---------------------------------------------------------------------------
# Files from data.zip that exist but are DELIBERATELY not loaded, and why.
# Documented here instead of silently skipped, so the reason survives
# code review / handoff:
#
#   data/knowledge.json
#       36MB "master" merge file. It is truncated / invalid JSON (parsing
#       fails partway through an activity_crosswalk_candidates array — the
#       file simply stops mid-object). Until it's regenerated from a
#       completed run, nothing in the pipeline reads it. Everything it was
#       meant to consolidate (source_registry, sbc_activities,
#       balady_activities_with_rules, service_knowledge, crosswalk
#       candidates) already exists individually under consolidated/, which
#       IS loaded below.
#
#   data/processed/zatca_extraction_review.json,
#   data/processed/zatca_cleaned_review.json
#       Per their own `approved: 0` / `review_status: PENDING_HUMAN_REVIEW`
#       fields, none of the 11 scraped ZATCA pages were approved — the
#       extracted text is raw site chrome/navigation, not service content.
#       Not used as evidence text. The ZATCA *links themselves* (from
#       source_registry.json, which is a curated list, not scraped text)
#       are still safe to surface — see zatca_evidence() below.
#
#   data/processed/unified_quarantine_faiss, unified_structured_faiss
#       Pre-built FAISS indexes over the full balady_rag.json corpus. See
#       load_faiss_index() below for why they're not wired up yet.
#
#   data/backups/*, data/processed/review_backups/*,
#   data/processed/document_review_queue.jsonl,
#   data/processed/content_audit_report.txt, source_failures.txt,
#   ingestion_preview.txt
#       Working files from the human-review pipeline that produced
#       consolidated/. Useful for audit, not for runtime evidence.
# ---------------------------------------------------------------------------


def _load_json(filename: str) -> Any:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Unzip data.zip and point NAWAH_DATA_DIR at "
            f"nawah_consolidated_dataset/consolidated/ (see README.md)."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_source_registry() -> list[dict]:
    return _load_json("source_registry.json")


@lru_cache(maxsize=1)
def load_service_knowledge() -> dict:
    return _load_json("service_knowledge.json")


@lru_cache(maxsize=1)
def load_sbc_activities() -> dict[str, dict]:
    """Keyed by activityId, e.g. '011101'."""
    records = _load_json("sbc_activities.json")
    return {r["activityId"]: r for r in records}


@lru_cache(maxsize=1)
def load_balady_activities() -> dict[str, dict]:
    """Keyed by activity_id, e.g. '1237'. This file is large (~38MB) —
    only loaded once, lazily, on first access."""
    records = _load_json("balady_activities_with_rules.json")
    return {r["activity"]["activity_id"]: r for r in records}


@lru_cache(maxsize=1)
def load_quality_issues() -> list[dict]:
    return _load_json("quality_issues.json")


@lru_cache(maxsize=1)
def load_crosswalk_candidates() -> list[dict]:
    return _load_json("activity_crosswalk_candidates.json")


@lru_cache(maxsize=1)
def load_investor_guide_correction() -> str | None:
    """
    data/corrections/investor_guide_page_6.txt — the human-reviewed,
    APPROVED replacement text for MISA Investor Guide page 6 (see
    review_decisions.jsonl: page 6 went needs_repair -> needs_repair ->
    approved on 2026-09-20). This is the one piece of real MISA
    registration-requirement text in the whole dataset that's cleared for
    use, so it replaces the placeholder unresolved_for_nawah entries.
    """
    path = CORRECTIONS_DIR / "investor_guide_page_6.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def source_registry_url(category: str) -> str | None:
    """First enabled source_registry.json URL for a given category, e.g.
    'commercial_registration', 'foreign_investment_company_formation',
    'commercial_licenses', 'vat_rules_and_registration'."""
    for entry in load_source_registry():
        if entry.get("category") == category and entry.get("enabled", True):
            return entry.get("url")
    return None


def search_source_registry(query: str, limit: int = 5) -> list[dict]:
    """Keyword match over category/authority for the Q&A agent — lets a
    free-text question surface the right official link even outside the
    activity/roadmap flow (e.g. 'GOSI', 'visa', 'VAT')."""
    q = query.strip().lower()
    if not q:
        return []
    hits = []
    for entry in load_source_registry():
        haystack = f"{entry.get('authority', '')} {entry.get('category', '')}".lower()
        if q in haystack:
            hits.append(entry)
        if len(hits) >= limit:
            break
    return hits


@lru_cache(maxsize=1)
def load_gov_sources() -> list[dict]:
    """
    data/saudi_gov_sources.json — the curated, manually-reviewed allow-list
    of Saudi government domains the Web Fallback Agent is permitted to
    search/fetch from. This is a SEPARATE file from source_registry.json:
    source_registry.json is category -> single curated URL for the
    structured pipeline; this file is a broader entity/domain allow-list
    used only to restrict live web search when the structured KB has
    nothing for a given activity.

    Every entry has a `status` of "confirmed" or "needs_verification" —
    see the file's own "_ملاحظة" note. Only "confirmed" entries are used
    as search-domain restrictions by default (see trusted_domains below);
    "needs_verification" entries are surfaced to a human reviewer, never
    silently trusted by the agent.
    """
    # Deliberately package-relative, NOT DATA_DIR: unlike the consolidated
    # dataset (which gets rebuilt/rescraped and lives under data_source/),
    # this file is a small, hand-curated allow-list that ships with the
    # code itself and shouldn't move if NAWAH_DATA_DIR ever changes.
    path = Path(__file__).parent / "saudi_gov_sources.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("entities", [])


def trusted_domains(include_needs_verification: bool = False) -> set[str]:
    """
    Domain allow-list for the Web Fallback Agent, derived from
    saudi_gov_sources.json. By default only 'confirmed' entries are
    included — the whole point of this list is that the agent must never
    search or accept results from a domain nobody has actually verified
    as the real government site, so a 'needs_verification' name (or a
    typo'd lookalike domain) can't quietly become a trusted source just
    because it's present in the file.
    """
    from urllib.parse import urlparse

    domains: set[str] = set()
    for entry in load_gov_sources():
        if not entry.get("url"):
            continue
        if entry.get("status") != "confirmed" and not include_needs_verification:
            continue
        host = urlparse(entry["url"]).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.add(host)
    return domains


def load_faiss_index():
    """
    Hook for later: data/processed/unified_structured_faiss holds a
    pre-built FAISS index over the full balady_rag.json corpus for semantic
    search. Not wired up yet (no network in this build environment to
    validate embedding-model compatibility) — load with
    `langchain_community.vectorstores.FAISS.load_local(...)` once you've
    confirmed which embedding model produced it.
    """
    raise NotImplementedError(
        "FAISS semantic search not wired up yet — see docstring."
    )


# ---------------------------------------------------------------------------
# Activity search (deterministic substring match over Arabic/English names)
# ---------------------------------------------------------------------------


def search_sbc_activities(query: str, limit: int = 5) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return []
    hits = []
    for record in load_sbc_activities().values():
        haystack = " ".join(
            str(record.get(k, "")) for k in ("nameAr", "nameEn", "descriptionAr", "descriptionEn")
        ).lower()
        if q in haystack:
            hits.append(record)
        if len(hits) >= limit:
            break
    return hits


def search_balady_activities(query: str, limit: int = 5) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return []
    hits = []
    for record in load_balady_activities().values():
        activity = record["activity"]
        haystack = " ".join(
            str(activity.get(k, "")) for k in ("activity_name", "isic_name", "description")
        ).lower()
        if q in haystack:
            hits.append(record)
        if len(hits) >= limit:
            break
    return hits


# ---------------------------------------------------------------------------
# Quality-gated evidence extraction
# ---------------------------------------------------------------------------


def sbc_activity_to_evidence(record: dict) -> list[dict]:
    """Turn one SBC activity record into raw evidence dicts (pre-Pydantic).
    Purely structural fields, always considered authoritative (no free-text
    provenance issue here — this is Nawah's own curated catalog field)."""
    activity_id = record["activityId"]
    items = []

    items.append(
        {
            "source_id": f"sbc:{activity_id}:license_before",
            "authority": "Saudi Business Center",
            "text_ar": (
                f"النشاط '{record.get('nameAr')}' يتطلب ترخيصًا قبل إصدار السجل التجاري."
                if record.get("licenseRequiredBeforeIssue")
                else f"النشاط '{record.get('nameAr')}' لا يتطلب ترخيصًا مسبقًا قبل إصدار السجل التجاري."
            ),
            "text_en": (
                f"Activity '{record.get('nameEn')}' requires a license BEFORE the "
                "commercial registration is issued."
                if record.get("licenseRequiredBeforeIssue")
                else f"Activity '{record.get('nameEn')}' does not require a pre-issue license."
            ),
            "url": None,
            "needs_verification": False,
        }
    )

    if record.get("licenseRequiredAfterIssue") and record.get("postAuthority"):
        items.append(
            {
                "source_id": f"sbc:{activity_id}:license_after",
                "authority": record["postAuthority"].get("nameEn", "Sector authority"),
                "text_ar": (
                    f"بعد إصدار السجل التجاري، يتطلب النشاط موافقة/ترخيص من "
                    f"{record['postAuthority'].get('nameAr')} خلال "
                    f"{record.get('maxDurationForPostAuthority', 'غير محدد')} يوم."
                ),
                "text_en": (
                    f"After registration, this activity requires approval/license from "
                    f"{record['postAuthority'].get('nameEn')} within "
                    f"{record.get('maxDurationForPostAuthority', 'unspecified')} days."
                ),
                "url": None,
                "needs_verification": False,
            }
        )

    if record.get("activityMinimumCapital"):
        items.append(
            {
                "source_id": f"sbc:{activity_id}:min_capital",
                "authority": "Saudi Business Center",
                "text_ar": f"الحد الأدنى لرأس المال لهذا النشاط: {record['activityMinimumCapital']}.",
                "text_en": f"Minimum capital for this activity: {record['activityMinimumCapital']}.",
                "url": None,
                "needs_verification": False,
            }
        )

    return items


def balady_activity_to_evidence(record: dict) -> list[dict]:
    """Turn Balady rules into evidence, gated per rule #4 above: no
    record-level URL -> needs_verification=True, never presented as
    fully confirmed."""
    activity = record["activity"]
    activity_id = activity["activity_id"]
    items = []
    for rule in record.get("rules", []):
        urls = rule.get("urls") or []
        items.append(
            {
                "source_id": f"balady_rule:{rule['rule_id']}",
                "authority": activity.get("supervising_authority_name", "Balady"),
                "text_ar": rule.get("text"),
                "text_en": None,  # Balady corpus is Arabic-only in this dataset
                "url": urls[0] if urls else None,
                "updated_date": rule.get("updated_date"),
                "needs_verification": len(urls) == 0,
                "verification_reason": (
                    None if urls else "No record-level source URL in Balady dataset."
                ),
            }
        )
    return items


def ejar_evidence(applies_to: str | None = None) -> list[dict]:
    """Requirements/eligibility from Ejar — fees are intentionally never
    exposed here (BLOCKING per quality_issues.json)."""
    ejar = load_service_knowledge().get("ejar", {})
    items = []
    for doc in ejar.get("required_documents", []):
        if applies_to and applies_to not in doc.get("applies_to", []):
            continue
        items.append(
            {
                "source_id": f"ejar:required_documents:{doc['document']}",
                "authority": "Ejar",
                "text_ar": doc["document"]
                + (f" ({doc.get('condition')})" if doc.get("condition") else ""),
                "text_en": None,
                "url": (ejar.get("sources") or [{}])[0].get("url"),
                "needs_verification": False,
            }
        )
    return items


def misa_evidence() -> list[dict]:
    """MISA requirements, honoring the unresolved_for_nawah review gate,
    plus the approved investor_guide_page_6.txt correction (real, verified
    requirements + processing time — not an estimate, it's the guide's own
    stated 10 working days)."""
    misa = load_service_knowledge().get("misa", {})
    items = []

    correction_text = load_investor_guide_correction()
    if correction_text:
        items.append(
            {
                "source_id": "misa:investor_guide_page_6:approved",
                "authority": "MISA",
                "text_ar": None,
                "text_en": (
                    "Investment registration requires: (1) a certified copy of the "
                    "commercial register of the participating foreign establishment; "
                    "(2) an identity document for any GCC-national partner not in "
                    "Absher; (3) certified financial statements for the last fiscal "
                    "year; (4) activity-specific requirements per the guide's Section "
                    "05.00; Special Residency Permit holders are exempt from (1)-(3). "
                    "Estimated processing time: 10 working days. The registration fee "
                    "itself is billed after approval, payable within 15 business days."
                ),
                "url": source_registry_url("investment_registration_and_activity_requirements"),
                "updated_date": "2026-09-20",
                "needs_verification": False,
            }
        )

    for req in misa.get("requirements", []) if isinstance(misa.get("requirements"), list) else []:
        items.append(
            {
                "source_id": f"misa:requirements:{req if isinstance(req, str) else req.get('id')}",
                "authority": "MISA",
                "text_ar": req if isinstance(req, str) else req.get("text_ar") or req.get("description"),
                "text_en": None,
                "url": None,
                "needs_verification": False,
            }
        )

    for unresolved in misa.get("unresolved_for_nawah", []):
        items.append(
            {
                "source_id": f"misa:unresolved:{unresolved.get('id')}",
                "authority": "MISA",
                "text_ar": unresolved.get("description"),
                "text_en": None,
                "url": None,
                "needs_verification": True,
                "verification_reason": "Listed in misa.unresolved_for_nawah — extraction incomplete.",
            }
        )

    return items


def zatca_evidence() -> list[dict]:
    """
    ZATCA (VAT/tax) registration is a near-universal post-commercial-
    registration step, so it's worth surfacing — but the only clean thing
    in this dataset for ZATCA is the curated source_registry.json link
    itself. The scraped page text (zatca_extraction_review.json /
    zatca_cleaned_review.json) is 0% approved and mostly site navigation
    chrome, so it is intentionally NOT used as evidence text here. Always
    flagged needs_verification so the Planning/Q&A agents phrase this as
    "go verify on the portal," never as a fully confirmed requirement.
    """
    url = source_registry_url("vat_rules_and_registration")
    return [
        {
            "source_id": "zatca:vat_registration:link_only",
            "authority": "ZATCA",
            "text_ar": "قد يلزمك التسجيل في ضريبة القيمة المضافة (VAT) لدى هيئة الزكاة والضريبة والجمارك بعد إصدار سجلك التجاري، حسب حجم أعمالك ونشاطك.",
            "text_en": "You may need to register for VAT with ZATCA after your commercial registration is issued, depending on your revenue and activity.",
            "url": url,
            "needs_verification": True,
            "verification_reason": (
                "Dataset's ZATCA page extractions are unapproved/pending review — "
                "link is from the curated source registry, but confirm current "
                "thresholds and process directly on the portal."
            ),
        }
    ]
