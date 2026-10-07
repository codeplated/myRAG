"""Load and format prompt templates from config directory."""
from __future__ import annotations

from .config import Settings


def get_template(settings: Settings, name: str | None = None) -> str:
    """Load template by name (filename without extension). Uses settings.prompt_template if name is None."""
    template_name = name or settings.prompt_template
    path = settings.prompts_dir / f"{template_name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def format_prompt(
    template: str,
    context: str,
    query: str,
    citation_instruction: str = "",
) -> str:
    """Fill template placeholders: context, query, citation_instruction."""
    return template.format(
        context=context,
        query=query,
        citation_instruction=citation_instruction or "",
    )
