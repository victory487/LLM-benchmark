#!/usr/bin/env python3
"""GDPval-AA bench using the REAL Stirrup harness for generation.

Generation: Stirrup Agent with LocalCodeExecToolProvider + WebToolProvider (Brave) — AA's actual
harness, no hand-rolled forcing. Deliverables are collected from the (preserved) code sandbox.
Judging: rotating panel of VLM judges. Aggregation: Bradley-Terry -> Elo anchored human=1000.

Env: BRAVE_API_KEY enables web_search. Config: ~/gdpval-bench/config.json.
"""
import os, sys, glob, json, shutil, random, math, itertools, contextlib, asyncio
sys.path.insert(0, os.path.expanduser("~/gdpval-bench"))
from gdpval_validate import render, b64url
from openai import OpenAI
from stirrup import Agent
from stirrup.clients import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider, WebToolProvider

TASKS_ROOT = os.path.expanduser("~/gdpval-bench/tasks")
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
OFFICE_EXTS = {"docx", "xlsx", "pptx", "pdf", "xls", "csv", "png"}
STIRRUP_SYS = ("You are an experienced professional completing a real work task. Use the code "
               "execution and web tools as needed to research and build the work. Produce the "
               "requested deliverable as a file in your working directory in the correct office "
               "format. Save the FINAL file as 'deliverable.EXT'. When finished, call finish and "
               "pass the deliverable file path(s) in paths.")

# ---------------- Stirrup generation ----------------
async def _stirrup_run(model, base, sysmsg, prompt, wd, sb, refs, brave_key, max_turns, ref_names, expected_ext):
    client = ChatCompletionsClient(model=model, base_url=base, api_key="x", max_tokens=12000,
                                   kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    tools = [LocalCodeExecToolProvider(temp_base_dir=sb)]
    if brave_key:
        tools.append(WebToolProvider(brave_api_key=brave_key))
    agent = Agent(client, name="gdpval_worker", max_turns=max_turns, system_prompt=sysmsg, tools=tools)
    async with agent.session(output_dir=wd, input_files=refs, clear_cache_on_success=False) as session:
        finish, msgs, meta = await session.run(prompt)
        produced = _collect(wd, ref_names, expected_ext)   # collect BEFORE the session context cleans up
    return finish, produced

def _delivs(wd, ref_names):
    out = [os.path.basename(f) for f in glob.glob(wd + "/*")
           if os.path.basename(f) not in ref_names and not os.path.basename(f).startswith("_")
           and not os.path.isdir(f)]
    out.sort(key=lambda b: (not b.lower().startswith("deliverable"), b))
    return out

def _has_deliv(wd, ref_names):
    return any(b.lower().startswith("deliverable") for b in _delivs(wd, ref_names))

def _collect(wd, ref_names, expected_ext):
    # Stirrup surfaces the model's declared finish.paths into output_dir (wd) — prefer that.
    top = sorted(f for f in os.listdir(wd)
                 if f.lower().startswith("deliverable.") and os.path.isfile(os.path.join(wd, f)))
    if top:
        return [top[0]]
    # fallback: any office file the model wrote in the sandbox but didn't declare in finish.paths
    cands = []
    for root, dd, ff in os.walk(wd):
        for f in ff:
            if f in ref_names or f.startswith("_"):
                continue
            ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
            if ext in OFFICE_EXTS:
                cands.append(os.path.join(root, f))
    if not cands:
        return []
    cands.sort(key=lambda p: (p.rsplit(".", 1)[-1].lower() != expected_ext, -os.path.getmtime(p)))
    chosen = cands[0]
    dst = os.path.join(wd, "deliverable." + chosen.rsplit(".", 1)[-1].lower())
    shutil.copy(chosen, dst)
    return [os.path.basename(dst)]

def generate(idx, model, base_url, expected_ext, max_turns=20):
    td = f"{TASKS_ROOT}/{idx}"
    wd = f"{td}/cand_{model}"
    ref_names = [os.path.basename(f) for f in glob.glob(td + "/refs/*")]
    if os.path.exists(wd) and _has_deliv(wd, ref_names):
        print(f"[gen] task{idx} {model}: cached", flush=True)
        return wd, _delivs(wd, ref_names)
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    sb = wd + "/_sandbox"; os.makedirs(sb)
    prompt = open(td + "/prompt.txt").read()
    refs = glob.glob(td + "/refs/*")
    sysmsg = STIRRUP_SYS.replace("EXT", expected_ext)
    brave_key = os.environ.get("BRAVE_API_KEY")
    finish = None; produced = []
    try:
        with open(wd + "/_stirrup.log", "w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            finish, produced = asyncio.run(_stirrup_run(model, base_url, sysmsg, prompt, wd, sb, refs, brave_key, max_turns, ref_names, expected_ext))
    except Exception as e:
        print(f"[gen] task{idx} {model}: STIRRUP-ERROR {str(e)[:140]}", flush=True)
    try:
        shutil.rmtree(sb)
    except Exception:
        pass
    fp = getattr(finish, "paths", None) if finish else None
    print(f"[gen] task{idx} {model}: finish.paths={fp} produced={produced}", flush=True)
    return wd, produced

# ---------------- panel judge ----------------
JUDGE_SYS = ("You are an expert occupational grader. You will see a work TASK and two candidate "
             "deliverables A and B (rendered as page images). Decide which better fulfills the task, "
             "weighing accuracy, completeness, instruction-following and presentation. Be decisive. "
             'Respond ONLY with JSON: {"winner":"A" or "B" or "tie","reason":"<=30 words"}.')

def judge_pair(prompt, imgA, imgB, judge):
    cli = OpenAI(base_url=judge["base"], api_key="x")
    content = [{"type": "text", "text": JUDGE_SYS + f"\n\nTASK:\n{prompt[:3000]}\n\n=== Deliverable A ==="},
               {"type": "image_url", "image_url": {"url": b64url(imgA)}},
               {"type": "text", "text": "=== Deliverable B ==="},
               {"type": "image_url", "image_url": {"url": b64url(imgB)}},
               {"type": "text", "text": "Which is better? JSON only."}]
    kw = {"extra_body": NOTHINK} if judge.get("nothink") else {}
    # instructions go in the user message (not a system message): some VLM chat templates
    # (e.g. InternVL) 400 on "system + multimodal user content".
    r = cli.chat.completions.create(model=judge["model"],
        messages=[{"role": "user", "content": content}],
        temperature=0.0, max_tokens=400, **kw)
    raw = r.choices[0].message.content or ""
    try:
        j = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]); return j.get("winner"), j.get("reason")
    except Exception:
        return None, "parse-fail"

# ---------------- Bradley-Terry -> Elo ----------------
def bradley_terry_elo(items, comps, iters=500, anchor="human", anchor_elo=1000.0):
    W = {i: 0.0 for i in items}; N = {i: {} for i in items}
    for a, b, wa in comps:
        W[a] += wa; W[b] += (1 - wa)
        N[a][b] = N[a].get(b, 0) + 1; N[b][a] = N[b].get(a, 0) + 1
    s = {i: 1.0 for i in items}
    for _ in range(iters):
        ns = {}
        for i in items:
            den = sum(n / (s[i] + s[j]) for j, n in N[i].items())
            ns[i] = (W[i] / den) if (den > 0 and W[i] > 0) else 1e-6
        gm = math.exp(sum(math.log(max(v, 1e-12)) for v in ns.values()) / len(ns))
        s = {i: ns[i] / gm for i in items}
    elo = {i: 400 * math.log10(max(s[i], 1e-12)) for i in items}
    shift = anchor_elo - elo.get(anchor, 0.0)
    return {i: round(elo[i] + shift, 1) for i in items}

# ---------------- main ----------------
def main():
    cfg = json.load(open(os.path.expanduser("~/gdpval-bench/config.json")))
    tasks, cands, judges = cfg["tasks"], cfg["candidates"], cfg["judges"]
    samples = cfg.get("samples", 1)
    rng = random.Random(cfg.get("seed", 7)); jcount = 0; results = []
    for idx in tasks:
        td = f"{TASKS_ROOT}/{idx}"
        gold = glob.glob(td + "/gold/*")[0]
        gext = gold.rsplit(".", 1)[-1].lower()
        prompt = open(td + "/prompt.txt").read()
        renders = {}
        try:
            renders["human"] = render(gold, td + "/render_gold")
        except Exception as e:
            print(f"task{idx} gold render fail: {e}"); continue
        for name, base in cands.items():
            try:
                wd, produced = generate(idx, name, base, gext)
            except Exception as e:
                print(f"task{idx} {name} GEN-FAIL: {str(e)[:150]}"); continue
            if not produced:
                print(f"task{idx} {name}: NO DELIVERABLE (skip)"); continue
            cand = next((os.path.join(wd, b) for b in produced if b.rsplit(".", 1)[-1].lower() == gext), os.path.join(wd, produced[0]))
            try:
                renders[name] = render(cand, wd + "/render_cand")
            except Exception as e:
                print(f"task{idx} {name} render fail: {e}")
        players = list(renders.keys())
        for a, b in itertools.combinations(players, 2):
            for _s in range(samples):
                judge = judges[jcount % len(judges)]; jcount += 1
                pair = [(a, renders[a]), (b, renders[b])]; rng.shuffle(pair)
                w, reason = judge_pair(prompt, pair[0][1], pair[1][1], judge)
                winner = pair[0][0] if w == "A" else pair[1][0] if w == "B" else "tie"
                results.append({"task": idx, "a": a, "b": b, "judge": judge["name"], "winner": winner, "reason": reason})
                print(f"  [judge {judge['name']}] task{idx}: {a} vs {b} -> {winner}", flush=True)
    items = sorted({p for r in results for p in (r["a"], r["b"])})
    comps = [(r["a"], r["b"], 1.0 if r["winner"] == r["a"] else 0.0 if r["winner"] == r["b"] else 0.5) for r in results]
    elo = bradley_terry_elo(items, comps) if comps else {}
    json.dump({"results": results, "elo": elo}, open(os.path.expanduser("~/gdpval-bench/bench_results_stirrup.json"), "w"), indent=1)
    print("\n==== GDPval-AA (Stirrup harness) ELO  (human anchored = 1000) ====")
    for it, e in sorted(elo.items(), key=lambda x: -x[1]):
        wins = sum(1 for r in results if r["winner"] == it)
        n = sum(1 for r in results if it in (r["a"], r["b"]))
        print(f"  {e:7.1f}  {it:32s}  ({wins}/{n} wins)")
    print(f"\ntotal comparisons: {len(results)} | judges: {[j['name'] for j in judges]}")

if __name__ == "__main__":
    main()
