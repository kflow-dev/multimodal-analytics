from pathlib import Path

from src.search_ingestion import build_request, discover_candidates


def test_discover_candidates_filters_local_files(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "genai-report.pdf").write_text("x")
    (docs / "notes.txt").write_text("x")
    (docs / "image.png").write_text("x")

    request = build_request(
        keywords="genai",
        document_types="pdf,text",
        input_location=str(docs),
    )

    candidates = discover_candidates(request)

    assert len(candidates) == 1
    assert candidates[0].reference.endswith("genai-report.pdf")
    assert candidates[0].document_type == "pdf"


def test_discover_candidates_filters_url_manifest(tmp_path: Path):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "\n".join(
            [
                "https://docs.example.com/genai-whitepaper.pdf",
                "https://blog.example.com/other-note.txt",
            ]
        )
    )

    request = build_request(
        keywords="genai",
        extensions=".pdf",
        input_location=str(manifest),
        url_subdomain_pattern="docs.example.com",
    )

    candidates = discover_candidates(request)

    assert len(candidates) == 1
    assert candidates[0].source_type == "url"
    assert candidates[0].reference == "https://docs.example.com/genai-whitepaper.pdf"


def test_discover_candidates_supports_specific_office_types(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "q1-budget.xlsx").write_text("x")
    (docs / "client-proposal.docx").write_text("x")
    (docs / "team-deck.pptx").write_text("x")

    request = build_request(
        keywords="budget,proposal,deck",
        document_types="word,excel,powerpoint",
        input_location=str(docs),
    )

    candidates = discover_candidates(request)

    assert len(candidates) == 3
    assert {candidate.document_type for candidate in candidates} == {
        "word",
        "excel",
        "powerpoint",
    }
