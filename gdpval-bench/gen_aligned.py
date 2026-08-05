#!/usr/bin/env python3
"""AA-aligned generation: thinking-ON, N samples/task.
Web: Brave web_search (if key set) + web_fetch. Model/endpoint from aligned.yaml.
"""
import asyncio, os, glob, shutil, sys, concurrent.futures
sys.path.insert(0, os.path.expanduser("~/gdpval-bench"))
from stirrup import Agent
from stirrup.clients import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider, WebToolProvider
import aligned_config

_argv, _cfg_path = aligned_config.pop_config_args()
CFG = aligned_config.load(_cfg_path)
TR = os.path.join(CFG["_root"], "tasks")
MODEL = CFG["candidate"]["model"]
BASE = CFG["candidate"]["base_url"]
API_KEY = CFG["candidate"]["api_key"]
BRAVE_KEY = CFG.get("brave_api_key")
N_SAMPLES = int(CFG["gen"]["n_samples"])
WORKERS = int(CFG["gen"]["workers"])
MAX_TURNS = int(CFG["gen"]["max_turns"])
MAX_TOKENS = int(CFG["gen"]["max_tokens"])
SAMPLING = dict(CFG["gen"].get("sampling") or {})

def _build_client_kwargs(sampling: dict):
    """Map yaml sampling to OpenAI/vLLM chat.completions args."""
    kw = {}
    if "temperature" in sampling:
        kw["temperature"] = float(sampling["temperature"])
    if "top_p" in sampling:
        kw["top_p"] = float(sampling["top_p"])
    extra = {}
    if "top_k" in sampling:
        extra["top_k"] = int(sampling["top_k"])
    if "repetition_penalty" in sampling:
        extra["repetition_penalty"] = float(sampling["repetition_penalty"])
    if extra:
        kw["extra_body"] = extra
    return kw or None

CLIENT_KWARGS = _build_client_kwargs(SAMPLING)
_client_kwargs = lambda: CLIENT_KWARGS  # noqa: E731 — keep old name for probes

if BRAVE_KEY:
    SYS = ("You are an experienced professional completing a real work task. "
           "Use web_search (Brave) to find sources, then web_fetch to read promising pages. "
           "Produce the requested deliverable as a file named 'deliverable.EXT' in your working directory; "
           "do genuine, well-formatted work; call finish with paths when done.")
else:
    SYS = ("You are an experienced professional completing a real work task. You have web_fetch (there is no "
           "separate search tool): to search the web, call web_fetch on a results URL such as "
           "https://html.duckduckgo.com/html/?q=YOUR+QUERY , then web_fetch the promising result pages. "
           "Produce the requested deliverable as a file named 'deliverable.EXT' in your working directory; "
           "do genuine, well-formatted work; call finish with paths when done.")

def gold_ext(idx):
    g = glob.glob(f"{TR}/{idx}/gold/*")
    return g[0].rsplit(".", 1)[-1].lower() if g else None

def sample_wd(idx, s):
    return f"{TR}/{idx}/cand_{MODEL}_aligned/s{s}"

async def _run(idx, s, ext):
    wd = sample_wd(idx, s); sb = wd + "/sb"
    os.makedirs(sb, exist_ok=True)
    prompt = open(f"{TR}/{idx}/prompt.txt").read()
    refs = glob.glob(f"{TR}/{idx}/refs/*")
    client = ChatCompletionsClient(
        model=MODEL, base_url=BASE, api_key=API_KEY, max_tokens=MAX_TOKENS, kwargs=CLIENT_KWARGS)
    tools = [LocalCodeExecToolProvider(temp_base_dir=sb),
             WebToolProvider(brave_api_key=BRAVE_KEY)]
    agent = Agent(client, name="w", max_turns=MAX_TURNS, system_prompt=SYS.replace("EXT", ext), tools=tools)
    async with agent.session(output_dir=wd, input_files=refs, clear_cache_on_success=False) as sess:
        finish, msgs, meta = await sess.run(prompt)
        found = [os.path.join(r, f) for r, d, ff in os.walk(wd) for f in ff
                 if f.lower().startswith("deliverable.")]
        for src in found:
            dst = f"{wd}/deliverable.{src.rsplit('.', 1)[-1].lower()}"
            if os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(dst):
                try:
                    shutil.copy(src, dst)
                except FileNotFoundError:
                    pass
    ok = bool(glob.glob(wd + "/deliverable.*"))
    shutil.rmtree(sb, ignore_errors=True)
    return ok

def gen_one(idx, s, ext):
    wd = sample_wd(idx, s)
    if glob.glob(f"{wd}/deliverable.*"):
        return "cached"
    if os.path.exists(wd + "/_EMPTY"):
        return "cached-empty"
    if os.path.exists(wd):
        shutil.rmtree(wd)
    try:
        ok = asyncio.run(_run(idx, s, ext))
        if not ok:
            open(wd + "/_EMPTY", "w").write("no deliverable produced")
        return "OK" if ok else "EMPTY"
    except Exception as e:
        return "ERR " + str(e)[:60]

def main():
    tasks = aligned_config.tasks_list(CFG)
    jobs = []
    for s in range(N_SAMPLES):
        for idx in tasks:
            ext = gold_ext(idx)
            if not ext:
                continue
            jobs.append((idx, s, ext))
    todo = [j for j in jobs
            if not glob.glob(f"{sample_wd(j[0], j[1])}/deliverable.*")
            and not os.path.exists(f"{sample_wd(j[0], j[1])}/_EMPTY")]
    print(f"aligned-gen config={CFG['_path']}", flush=True)
    print(f"aligned-gen model={MODEL} base={BASE} brave_search={'on' if BRAVE_KEY else 'off'}", flush=True)
    print(f"aligned-gen sampling={SAMPLING or 'server-default'}", flush=True)
    print(f"aligned-gen: {len(jobs)} total ({N_SAMPLES} samples x {len(jobs)//max(N_SAMPLES,1)} tasks), "
          f"{len(todo)} to run, workers={WORKERS}", flush=True)
    done = 0
    from collections import Counter
    c = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(gen_one, *j): j for j in todo}
        for fut in concurrent.futures.as_completed(futs):
            done += 1; j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = "ERR " + str(e)[:50]
            c[r.split()[0]] += 1
            if done % 25 == 0 or r.startswith("ERR"):
                print(f"  {done}/{len(todo)} t{j[0]}s{j[1]}: {r} | {dict(c)}", flush=True)
    print(f"ALIGNED-GEN-DONE {dict(c)}", flush=True)

if __name__ == "__main__":
    main()
