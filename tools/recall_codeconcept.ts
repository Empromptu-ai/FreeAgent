// Empromptu FreeAgent - The free, local, entirely private agent coding system, by Empromptu!
// Copyright (C) 2025  Empromptu, Sean Robinson
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of version 3 of the GNU General Public License as published by
// the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

import { tool } from "@opencode-ai/plugin"

export default tool({
  description:
    "Look up code concepts by tag and get back a digest: each matching " +
    "concept's description plus the actual code of its members (and directly " +
    "related concepts). Tags come from the code-concept index in your context " +
    "(each line is '<tag>: <summary>'). Use this when you know which concept " +
    "you want; use query_codeconcept for free-text search.",
  args: {
    tags: tool.schema
      .array(tool.schema.string())
      .describe("One or more concept tags, e.g. ['rate-limiting', 'auth-flow']"),
  },
  async execute(args, _context) {
    const res = await fetch("http://127.0.0.1:49786/codegraph/recall", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tags: args.tags }),
    })
    const data = await res.json()
    return data.text ?? "No matching code concepts found."
  },
})
