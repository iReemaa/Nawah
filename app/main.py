from pathlib import Path
import json
import hashlib
import shutil
from datetime import datetime, timezone

from app.ingestion import run_ingestion_pipeline, IngestionPipelineError, audit_documents


def main():

    try:
        # ==========================================
        # 1. RUN INGESTION PIPELINE
        # ==========================================

        documents, failures = run_ingestion_pipeline()

        print("\nNawah ingestion pipeline completed.")

        print(
            f"Documents created: {len(documents)}"
        )

        print(
            f"Failures: {len(failures)}"
        )

        if failures:

            print("\nFailures:")

            for failure in failures:
                print(f"- {failure}")

        # ==========================================
        # 2. DISPLAY DOCUMENT METADATA
        # ==========================================

        print("\nDocument metadata sample:")

        if documents:
            for key, value in documents[0].metadata.items():
                print(f"{key}: {value}")

        # ==========================================
        # 3. PREVIEW EXTRACTED CONTENT
        # ==========================================

        print("\n" + "=" * 80)
        print("EXTRACTED CONTENT PREVIEW")
        print("=" * 80)

        preview_count = min(5, len(documents))

        for index, document in enumerate(
            documents[:preview_count],
            start=1
        ):

            print(f"\nDOCUMENT {index}")
            print("-" * 80)

            print(
                f"Authority: "
                f"{document.metadata.get('authority')}"
            )

            print(
                f"Category: "
                f"{document.metadata.get('category')}"
            )

            print(
                f"Source type: "
                f"{document.metadata.get('source_type')}"
            )

            print(
                f"Source URL: "
                f"{document.metadata.get('source_url')}"
            )

            page_number = document.metadata.get(
                "page_number"
            )

            if page_number:
                print(f"Page: {page_number}")

            print("\nACTUAL EXTRACTED TEXT:")

            print(
                document.page_content[:1500]
            )

            print("\n" + "-" * 80)

        # ==========================================
        # 4. SAVE FULL EXTRACTED CONTENT
        # ==========================================

        output_directory = Path(
            "data/processed"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        preview_file = (
            output_directory / "ingestion_preview.txt"
        )

        with preview_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            for index, document in enumerate(
                documents,
                start=1
            ):

                file.write(
                    "\n" + "=" * 80 + "\n"
                )

                file.write(
                    f"DOCUMENT {index}\n"
                )

                file.write(
                    "=" * 80 + "\n"
                )

                for key, value in document.metadata.items():

                    file.write(
                        f"{key}: {value}\n"
                    )

                file.write(
                    "\nEXTRACTED TEXT:\n\n"
                )

                file.write(
                    document.page_content
                )

                file.write("\n")

        print(
            f"\nFull extracted content saved to: "
            f"{preview_file}"
        )

        # ==========================================
        # 5. RUN CONTENT AUDIT
        # ==========================================

        audit_results = audit_documents(
            documents
        )

        passed = [
            result
            for result in audit_results
            if result.status == "PASS"
        ]

        review = [
            result
            for result in audit_results
            if result.status == "REVIEW"
        ]

        print("\n" + "=" * 60)
        print("CONTENT AUDIT RESULTS")
        print("=" * 60)

        print(
            f"Documents audited: {len(audit_results)}"
        )

        print(
            f"Passed screening: {len(passed)}"
        )

        print(
            f"Require review: {len(review)}"
        )

        # ==========================================
        # 6. SAVE CONTENT AUDIT REPORT
        # ==========================================

        report_path = (
            output_directory / "content_audit_report.txt"
        )

        with report_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "NAWAH CONTENT AUDIT REPORT\n"
            )

            file.write(
                "=" * 70 + "\n"
            )

            file.write(
                f"Documents audited: {len(audit_results)}\n"
            )

            file.write(
                f"Passed screening: {len(passed)}\n"
            )

            file.write(
                f"Require review: {len(review)}\n"
            )

            for index, result in enumerate(
                audit_results,
                start=1
            ):

                document = result.document

                file.write(
                    "\n" + "=" * 70 + "\n"
                )

                file.write(
                    f"DOCUMENT {index}\n"
                )

                file.write(
                    f"STATUS: {result.status}\n"
                )

                file.write(
                    f"Authority: "
                    f"{document.metadata.get('authority')}\n"
                )

                file.write(
                    f"Category: "
                    f"{document.metadata.get('category')}\n"
                )

                file.write(
                    f"Source: "
                    f"{document.metadata.get('source_url')}\n"
                )

                file.write(
                    f"Page: "
                    f"{document.metadata.get('page_number', 'N/A')}\n"
                )

                file.write(
                    f"Reasons: "
                    f"{', '.join(result.reasons) or 'None'}\n"
                )

                file.write(
                    "\nCONTENT PREVIEW:\n\n"
                )

                file.write(
                    document.page_content[:1000]
                )

                file.write("\n")

        # Persist failures separately so inaccessible sources can be retried.
        failure_path = output_directory / "source_failures.txt"
        failure_path.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")

        # Screening is NOT regulatory approval. Every document stays in a
        # review queue; nothing is exported as approved/indexable automatically.
        queue_path = output_directory / "document_review_queue.jsonl"
        decisions_path = output_directory / "review_decisions.jsonl"
        # Preserve the previous queue and all human decisions before refreshing.
        backup_dir = output_directory / "review_backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_dir.mkdir(parents=True, exist_ok=False)
        for existing in (queue_path, decisions_path):
            if existing.is_file():
                shutil.copy2(existing, backup_dir / existing.name)

        # Write a complete new queue before replacing the previous queue.
        refreshed_ids = set()
        temp_queue = queue_path.with_suffix(".jsonl.tmp")
        try:
            with temp_queue.open("w", encoding="utf-8") as output:
                for result in audit_results:
                    metadata = result.document.metadata
                    identity = json.dumps({
                        "source_url": metadata.get("source_url"),
                        "page_number": metadata.get("page_number"),
                        "content": result.document.page_content,
                    }, sort_keys=True, ensure_ascii=False)
                    document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                    refreshed_ids.add(document_id)
                    output.write(json.dumps({
                        "document_id": document_id,
                        "status": result.status,
                        "approved_for_indexing": False,
                        "reasons": result.reasons,
                        "content": result.document.page_content,
                        "metadata": result.document.metadata,
                    }, ensure_ascii=False) + "\n")
            temp_queue.replace(queue_path)
        finally:
            if temp_queue.exists():
                temp_queue.unlink()

        previous_decisions = []
        if decisions_path.is_file():
            with decisions_path.open(encoding="utf-8") as stream:
                previous_decisions = [json.loads(line) for line in stream if line.strip()]
        preserved = sum(d.get("document_id") in refreshed_ids for d in previous_decisions)
        stale = len(previous_decisions) - preserved
        print(f"Review backup saved to: {backup_dir}")
        print(f"Decision records matching unchanged content: {preserved}")
        print(f"Decision records requiring re-review after content changes/removal: {stale}")
        print("Review decisions file was NOT overwritten. Changed content requires new approval.")

        print(f"\nFailures saved to: {failure_path}")
        print(f"Review queue saved to: {queue_path}")
        print("Indexing gate: CLOSED — no documents are automatically approved.")
        print("PASS means screening only; REVIEW and failed sources remain excluded.")
        print(
            f"\nAudit report saved to: {report_path}"
        )

        print(
            "\nContent audit completed."
        )

    except IngestionPipelineError as error:

        print(
            "\nNawah ingestion pipeline failed."
        )

        print(
            f"Error: {error}"
        )


if __name__ == "__main__":
    main()
