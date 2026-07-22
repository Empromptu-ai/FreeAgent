# Empromptu FreeAgent - The free, local, entirely private agent coding system, by Empromptu!
# Copyright (C) 2025  Empromptu, Sean Robinson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of version 3 of the GNU General Public License as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

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
