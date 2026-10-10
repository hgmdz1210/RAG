import pytest

from rag import cli
from rag.generation.answer import GenerationError


@pytest.fixture
def use_services(monkeypatch):
    """Make the CLI use the given fake services instead of building real ones."""

    def install(services):
        monkeypatch.setattr(cli, "build_services", lambda settings, collection=None: services)
        return services

    return install


def test_ingest_indexes_a_folder(tmp_path, services, use_services, capsys):
    use_services(services)
    (tmp_path / "a.txt").write_text("Alpha document about retrieval.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Beta\n\nBeta document about ranking.", encoding="utf-8")

    exit_code = cli.main(["ingest", str(tmp_path)])

    assert exit_code == 0
    assert services.store.count() == 2
    assert "Indexed 2 documents as 2 chunks into 'test'." in capsys.readouterr().out


def test_ingest_reports_a_missing_path(tmp_path, services, use_services, capsys):
    use_services(services)

    exit_code = cli.main(["ingest", str(tmp_path / "missing")])

    assert exit_code == 2
    assert "Path not found" in capsys.readouterr().err


def test_ingest_reports_a_folder_with_nothing_to_index(tmp_path, services, use_services, capsys):
    use_services(services)
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    exit_code = cli.main(["ingest", str(tmp_path)])

    assert exit_code == 1
    assert "No readable" in capsys.readouterr().err


def test_search_prints_ranked_passages(indexed_services, use_services, capsys):
    use_services(indexed_services)

    exit_code = cli.main(["search", "reciprocal rank fusion", "--top-k", "2", "--no-rerank"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.splitlines()[0].startswith(" [1] fusion.md#0")
    assert " [2] " in output
    assert " [3] " not in output


def test_ask_prints_the_answer_and_marks_cited_sources(indexed_services, use_services, capsys):
    use_services(indexed_services)

    exit_code = cli.main(["ask", "How are two rankings merged?"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.startswith("Based on the first passage [1].")
    assert "*[1] " in output
    assert " [2] " in output


def test_ask_reports_generation_failures(indexed_services, use_services, capsys):
    use_services(indexed_services)
    indexed_services.generator.error = GenerationError("no credentials")

    exit_code = cli.main(["ask", "anything"])

    assert exit_code == 1
    assert "no credentials" in capsys.readouterr().err
