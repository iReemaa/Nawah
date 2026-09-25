# Nawah consolidated dataset

Canonical files: `source_registry.json` (use as `data/sources.json` only after review), `balady_activities_with_rules.json`, `sbc_activities.json`, `service_knowledge.json`. `activity_crosswalk_candidates.json` contains **unverified** possible joins, not automatic merges. `quality_issues.json` lists unresolved matters. `deduplication_log.json` records exact repeated descriptions omitted from the canonical Balady activity records. Original seven files are preserved in `originals/` inside the ZIP.

Do not pass all canonical JSON to the current `sources.json` loader: it expects the source registry list only. Structured records need a separate ingestion adapter and provenance/review gate. Do not automatically index `raw_extracted_content`, `unverified_rules`, or conflicting fee values as authoritative.
