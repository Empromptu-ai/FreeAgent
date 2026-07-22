"""Shared test helpers: build a ContextArchitect wired to the fake backend."""

from context_architect import Config, ContextArchitect, LLMConfig


def make_architect(tmp_path, num_full_text_turns=None):
    kwargs = dict(
        storage_root=str(tmp_path / "ca"),
        llm=LLMConfig(provider="fake"),
    )
    if num_full_text_turns is not None:
        kwargs["num_full_text_turns"] = num_full_text_turns
    return ContextArchitect(Config(**kwargs))
