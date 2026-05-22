from pathlib import Path


def test_gitignore_ignores_local_generated_artifacts():
    gitignore = Path(".gitignore").read_text()

    assert "runs/" in gitignore
    assert ".superpowers/" in gitignore


def test_claude_md_mentions_app_commands_and_architecture():
    guidance = Path("CLAUDE.md").read_text()

    assert "uv run python main.py" in guidance
    assert "uv run pytest" in guidance
    assert "FastAPI" in guidance
    assert "diffusers" in guidance
    assert "Server-Sent Events" in guidance
