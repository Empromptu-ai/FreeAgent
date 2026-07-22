import { tool } from "@opencode-ai/plugin"

export default tool({
  description:
    "Fetch the full, uncompressed detail of an earlier turn by its archive " +
    "key. Summaries in the history end with '(recall: turn-NNNN)' — pass that " +
    "key here when a summary isn't enough.",
  args: {
    key: tool.schema.string().describe("Archive key, e.g. 'turn-0003'"),
  },
  async execute(args, context) {
    // context.sessionID matches the x-session-id the proxy keyed the archive on.
    const res = await fetch("http://127.0.0.1:49786/recall", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session: context.sessionID, key: args.key }),
    })
    const data = await res.json()
    return data.text ?? "No archived turn found."
  },
})
