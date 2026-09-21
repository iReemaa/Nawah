import json
from pathlib import Path

path = Path("data/sources.json")

new_urls = [
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-009.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-225.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-234.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-048.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-001.aspx",
    "https://zatca.gov.sa/ar/eServices/UserManual/Pages/UserManual-001.aspx?service=eServices-001",
    "https://zatca.gov.sa/ar/E-Invoicing/Pages/default.aspx",
    "https://zatca.gov.sa/ar/E-Invoicing/Introduction/Pages/Roll-out-phases.aspx",
    "https://zatca.gov.sa/ar/E-Invoicing/Introduction/Guidelines/Pages/default.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-273.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/Product-Identification.aspx",
]

with path.open(encoding="utf-8") as file:
    sources = json.load(file)

existing = {
    item["url"].rstrip("/").replace("www.", "")
    for item in sources
}

added = 0

for url in dict.fromkeys(new_urls):
    normalized = url.rstrip("/").replace("www.", "")

    if normalized in existing:
        continue

    category = (
        "e_invoicing"
        if "/E-Invoicing/" in url
        else "zatca_services"
    )

    sources.append({
        "authority": "ZATCA",
        "category": category,
        "source_type": "webpage",
        "url": url,
        "language": "ar",
        "business_activity": "general",
        "region": "national",
        "enabled": True
    })

    existing.add(normalized)
    added += 1

with path.open("w", encoding="utf-8") as file:
    json.dump(sources, file, ensure_ascii=False, indent=2)

print(f"Added {added} new ZATCA sources.")
print(f"Total registered sources: {len(sources)}")