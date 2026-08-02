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
    "Ask about a concept or system in free text and get a digest of the most " +
    "relevant code: matching concepts' descriptions plus the actual code of " +
    "their members (and directly related concepts). Semantic search over the " +
    "code-concept graph — use this when you don't know the exact tag.",
  args: {
    query: tool.schema
      .string()
      .describe("What you want to understand, e.g. 'how are outbound requests throttled?'"),
  },
  async execute(args, _context) {
    const res = await fetch("http://127.0.0.1:49786/codegraph/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: args.query }),
    })
    const data = await res.json()
    return data.text ?? "No relevant code concepts found."
  },
})
