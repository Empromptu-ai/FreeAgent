# Empromptu FreeAgent - The free, local, entirely private agent coding system, by Empromptu!

## Setup — steps to take (TL:DR; run Setup_FreeAgent and then FreeAgent)

The proxy is an **Ollama device**: it forwards OpenCode's requests to an Ollama
server and uses that same server for its own summary/label calls. Pick one of
the two flows below. Both end with `./FreeAgent`, which is idempotent —
re-run it any time.

### First time setup -  

#### Set up an Ollama server, either on the current computer or remotely.  This is pretty quick:
Ubuntu:
bash
curl -fsSL https://ollama.com/install.sh | sh

Mac:
bash
brew install ollama
(or download the app from https://ollama.com/download)

#### Then run ollama serve (if not auto-started) and ollama run llama3 (or any model) to start using it.


### Run `./Setup_FreeAgent` to put in whether you want a local or remote server. It's a friendly, interactive
configurator (no prior setup needed) that updates your
`.env` for you. Chances are, you want a standard local install, or a remote install
if you have an Ollama server somewhere else already.  If you're not sure, just double-click on
Setup_Freeagent and hit L for "Local".  You'll get a suggested model size that should fit and 
work on your computer (just hit enter to use it), or you can change it to any other model name in the Ollama format. 
Everything in .env can also be changed by hand, but `Setup_FreeAgent` saves you from editing the file directly. `./Setup_FreeAgent --help` for details.

### A. Local Ollama (everything gets set up on this machine)

Optional - add/change anything in .env.example, and/or copy it to .env (but the main and setup scripts also handle this). Then:
```sh
./FreeAgent
```
(Or just doubleclick on it). That should start everything.

That single command will: create the venv, install the Python deps, install
Ollama (if missing) and start it, pull the model in `FA_MODEL`, install
OpenCode, write `~/.config/opencode/opencode.json`, place the `recall_turn`
tool, and launch the proxy on `127.0.0.1:49786`.

This should launch a session of OpenCode that is set up to use the FreeAgent proxy (if it doesn't, try running FreeAgent again). You can also install the OpenCode widget in VSCode, and it should be automatically set up.  You can tell by the purple background!

OpenCode should already be set up to use your model. If it isn't, then type `/models` and pick **"Ollama/ (via free_agent)" / your-model-name (via free_agent)**.

### B. Remote Ollama (model runs on another machine)

On the **remote** machine, run Ollama so it listens on the network and pull the
model:

```sh
OLLAMA_HOST=0.0.0.0:11434 ollama serve      # bind all interfaces
ollama pull qwen3.6:35b                      # the model you'll use
```

On **this** machine, edit `.env` or `.env.example` before running the installer:
OLLAMA_BASE_URL=http://localhost:11434 (change this to be wherever your ollama server is)
Then:
```sh
cp .env.example .env
# then set these in .env:
#   OLLAMA_BASE_URL=http://<remote-host>:11434
#   FA_SKIP_OLLAMA=1        # don't install/start a local Ollama
#   FA_MODEL=qwen3.6:35b    # must already be pulled on the remote
./FreeAgent
```

More options:
`FA_SKIP_OLLAMA=1` skips the local Ollama install/serve/pull but still installs
the Python deps and everything else. The proxy still runs locally on
`127.0.0.1:49786`; only the model calls go to the remote.

> **Note:** the proxy sends no auth header, so the remote Ollama must be
> reachable without a token (LAN / VPN / SSH tunnel). Hosted APIs that require a
> key (OpenAI, Anthropic) are **not** reachable through the proxy today.

## Changing the port on the proxy (this code)

Set `FA_PROXY_PORT` (and/or `FA_PROXY_HOST`) in `.env` and re-run
`./FreeAgent`. The generated `opencode.json` and every existing
`recall_turn.ts` are re-synced to the new address automatically — only the
`/recall` URL is rewritten, the tool definition itself is left untouched.



## License

Empromptu FreeAgent — the free, local, entirely private agent coding system, by
Empromptu!

Copyright (C) 2025 Empromptu, Sean Robinson

This program is free software: you can redistribute it and/or modify it under
the terms of version 3 of the GNU General Public License as published by the
Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program (see [LICENSE](LICENSE)). If not, see
<https://www.gnu.org/licenses/>.
