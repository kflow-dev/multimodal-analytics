from pathlib import Path

from src.create_thesaurus import create_thesaurus


def test_create_thesaurus_summarizes_local_files(tmp_path: Path):
    root = tmp_path / "showcase"
    root.mkdir()
    (root / "a.md").write_text("# note")
    (root / "b.json").write_text("{}")

    summary = create_thesaurus(root)

    assert summary["total_files"] == 2
    assert "markdown" in summary["groups"]
    assert "code" in summary["groups"]
