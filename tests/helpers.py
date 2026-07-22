"""Shared test helpers: build a FreeAgent wired to the fake backend."""

from free_agent import Config, FreeAgent, LLMConfig


def make_agent(tmp_path, num_full_text_turns=None):
    kwargs = dict(
        storage_root=str(tmp_path / "fa"),
        llm=LLMConfig(provider="fake"),
    )
    if num_full_text_turns is not None:
        kwargs["num_full_text_turns"] = num_full_text_turns
    return FreeAgent(Config(**kwargs))
