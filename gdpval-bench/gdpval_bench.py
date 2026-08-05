#!/usr/bin/env python3
"""GDPval-AA-fidelity harness: web-enabled agentic generation + judge panel + Bradley-Terry Elo.

- Candidates get run_python + web_search (ddgs) + web_fetch + finish (Stirrup-style capability set).
- Each blind pairwise comparison (among candidates AND the human gold) is decided by a judge
  ROTATED across a panel of different judges, to reduce same-family bias (AA's approach).
- Pairwise outcomes are fit with Bradley-Terry (MM algorithm) -> Elo, anchored so human = 1000.

Config at bottom. Deliverables + renders are cached on disk so judging/Elo can be re-run cheaply.
"""
import os, sys, json, glob, shutil, subprocess, random, re, math, itertools, html
sys.path.insert(0, os.path.expanduser("~/gdpval-bench"))
from gdpval_validate import render, b64url          # reuse rendering
from openai import OpenAI
import requests

TASKS_ROOT = os.path.expanduser("~/gdpval-bench/tasks")
PYBIN = os.path.expanduser("~/gdpval-bench/.venv/bin/python")
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
WEB_BUDGET = 5   # after this many web_search/web_fetch calls, web tools are removed to force production

# ---------------- generation tools ----------------
def run_python(code, wd):
    open(os.path.join(wd, "_step.py"), "w").write(code)
    try:
        p = subprocess.run([PYBIN, "_step.py"], cwd=wd, capture_output=True, text=True, timeout=200)
        return f"stdout:\n{(p.stdout or '')[-2800:]}\nstderr:\n{(p.stderr or '')[-1800:]}"
    except subprocess.TimeoutExpired:
        return "ERROR: python timed out"

def web_search(query, k=5):
    try:
        from ddgs import DDGS
        rows = list(DDGS().text(query, max_results=k))
        return "\n".join(f"- {r.get('title','')}: {r.get('href','')}\n  {r.get('body','')[:200]}" for r in rows) or "(no results)"
    except Exception as e:
        return f"search error: {e}"

def web_fetch(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        t = r.text
        try:
            from bs4 import BeautifulSoup
            t = BeautifulSoup(t, "lxml").get_text(" ", strip=True)
        except Exception:
            t = re.sub(r"<[^>]+>", " ", t)
        return html.unescape(t)[:4000]
    except Exception as e:
        return f"fetch error: {e}"

RUN_PY = {"type": "function", "function": {"name": "run_python",
    "description": "Execute Python in the working dir to build the deliverable. pandas/openpyxl/xlsxwriter/python-docx(docx)/python-pptx(pptx)/matplotlib available. Files persist.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
WEB_SEARCH = {"type": "function", "function": {"name": "web_search",
    "description": "Search the web; returns top results with titles, URLs, snippets.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
WEB_FETCH = {"type": "function", "function": {"name": "web_fetch",
    "description": "Fetch a URL and return its main text content.",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}}
FINISH = {"type": "function", "function": {"name": "finish",
    "description": "Call once the final deliverable file exists in the working directory.",
    "parameters": {"type": "object", "properties": {"deliverable_files": {"type": "array", "items": {"type": "string"}}}, "required": ["deliverable_files"]}}}

def _delivs(wd, refs):
    out = [os.path.basename(f) for f in glob.glob(wd + "/*")
           if os.path.basename(f) not in refs and not os.path.basename(f).startswith("_") and not os.path.isdir(f)]
    out.sort(key=lambda b: (not b.lower().startswith("deliverable"), b))
    return out

def _has_deliv(wd, refs):
    return any(b.lower().startswith("deliverable") for b in _delivs(wd, refs))

def generate(idx, model, base_url, expected_ext, max_turns=18, force=False):
    td = f"{TASKS_ROOT}/{idx}"
    wd = f"{td}/cand_{model}"
    if os.path.exists(wd) and not force and _has_deliv(wd, [os.path.basename(f) for f in glob.glob(td+'/refs/*')]):
        print(f"[gen] task{idx} {model}: cached", flush=True)
        return wd, _delivs(wd, [os.path.basename(f) for f in glob.glob(td+'/refs/*')])
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    prompt = open(td + "/prompt.txt").read()
    refs = []
    for rf in glob.glob(td + "/refs/*"):
        shutil.copy(rf, wd); refs.append(os.path.basename(rf))
    cli = OpenAI(base_url=base_url, api_key="x")
    sysmsg = ("You are an experienced professional completing a real work task and producing a polished "
              f"deliverable file. SAVE the final deliverable as a relative file 'deliverable.{expected_ext}' "
              "in the current working directory (never /tmp or absolute paths). "
              f"Reference files already present: {refs}. You may use web_search / web_fetch to research, "
              "and run_python (pandas/openpyxl/xlsxwriter/python-docx/python-pptx/matplotlib) to build the "
              "file. Write a COMPLETE deliverable early, then refine. Do genuine, well-formatted work — "
              "never a stub. Call finish when done.")
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}]
    tools = [RUN_PY, WEB_SEARCH, WEB_FETCH, FINISH]
    final = None; used = 0; empties = 0; web_calls = 0; web_cut = False
    for t in range(max_turns):
        used = t + 1
        cur_tools = tools if web_calls < WEB_BUDGET else [RUN_PY, FINISH]
        if web_calls >= WEB_BUDGET and not web_cut:
            web_cut = True
            msgs.append({"role": "user", "content":
                         f"Web research budget reached ({web_calls} calls). Do NOT search the web further. "
                         f"Now write a COMPLETE 'deliverable.{expected_ext}' with run_python, then call finish."})
        elif t == max_turns - 3 and not _has_deliv(wd, refs):
            msgs.append({"role": "user", "content":
                         f"Few turns left. In your next run_python call write a COMPLETE 'deliverable.{expected_ext}' now, then finish."})
        r = cli.chat.completions.create(model=model, messages=msgs, tools=cur_tools, tool_choice="auto",
                                        temperature=0.3, max_tokens=8000, extra_body=NOTHINK)
        m = r.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        called = []
        if not m.tool_calls:
            code = re.search(r"```(?:python)?\s*\n(.*?)```", m.content or "", re.S)
            if code:
                called = ["run_python(fallback)"]; res = run_python(code.group(1), wd)
                msgs.append({"role": "user", "content": f"(auto-ran your code)\n{res[:2500]}"})
            else:
                msgs.append({"role": "user", "content": f"Call run_python to create 'deliverable.{expected_ext}'."})
        else:
            for tc in m.tool_calls:
                fn = tc.function.name; called.append(fn)
                try: args = json.loads(tc.function.arguments or "{}")
                except Exception: args = {}
                if fn == "run_python":
                    _code = args.get("code", "")
                    if not _code.strip():
                        empties += 1
                        res = "ERROR: empty code argument. Send the COMPLETE python code that builds the deliverable."
                    else:
                        empties = 0; res = run_python(_code, wd)
                elif fn == "web_search":
                    if web_calls >= WEB_BUDGET: res = "Web budget exhausted — web is disabled. Write the deliverable now with run_python."
                    else: web_calls += 1; res = web_search(args.get("query", ""))
                elif fn == "web_fetch":
                    if web_calls >= WEB_BUDGET: res = "Web budget exhausted — web is disabled. Write the deliverable now with run_python."
                    else: web_calls += 1; res = web_fetch(args.get("url", ""))
                elif fn == "finish":
                    if _has_deliv(wd, refs): final = args.get("deliverable_files"); res = "acknowledged"
                    else: res = f"No deliverable.{expected_ext} yet; create it first."
                else: res = "unknown tool"
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res[:3500]})
        print(f"  [gen t{idx}/{model}] turn{used}: {called} deliv={_has_deliv(wd, refs)}", flush=True)
        if empties >= 3 and not _has_deliv(wd, refs):
            print(f"[gen] task{idx} {model}: abort after {empties} empty-code calls", flush=True); break
        if final is not None: break
    json.dump(msgs, open(wd + "/_transcript.json", "w"), default=str, indent=1)
    produced = _delivs(wd, refs)
    print(f"[gen] task{idx} {model}: turns={used} produced={produced}", flush=True)
    return wd, produced

# ---------------- panel judge ----------------
JUDGE_SYS = ("You are an expert occupational grader. You will see a work TASK and two candidate "
             "deliverables A and B (rendered as page images). Decide which better fulfills the task, "
             "weighing accuracy, completeness, instruction-following and presentation. Be decisive. "
             'Respond ONLY with JSON: {"winner":"A" or "B" or "tie","reason":"<=30 words"}.')

def judge_pair(prompt, imgA, imgB, judge):
    cli = OpenAI(base_url=judge["base"], api_key="x")
    content = [{"type": "text", "text": f"TASK:\n{prompt[:3000]}\n\n=== Deliverable A ==="},
               {"type": "image_url", "image_url": {"url": b64url(imgA)}},
               {"type": "text", "text": "=== Deliverable B ==="},
               {"type": "image_url", "image_url": {"url": b64url(imgB)}},
               {"type": "text", "text": "Which is better? JSON only."}]
    kw = {"extra_body": NOTHINK} if judge.get("nothink") else {}
    r = cli.chat.completions.create(model=judge["model"],
        messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": content}],
        temperature=0.0, max_tokens=400, **kw)
    raw = r.choices[0].message.content or ""
    try:
        j = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]); return j.get("winner"), j.get("reason")
    except Exception:
        return None, "parse-fail: " + raw[:80]

# ---------------- Bradley-Terry -> Elo ----------------
def bradley_terry_elo(items, comps, iters=500, anchor="human", anchor_elo=1000.0):
    # comps: list of (a, b, wa) with wa in {1,0.5,0} = a's win fraction vs b
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
    return {i: round(elo[i] + shift, 1) for i in items}, s

# ---------------- main pipeline ----------------
def main():
    cfg = json.load(open(os.path.expanduser("~/gdpval-bench/config.json")))
    tasks = cfg["tasks"]; cands = cfg["candidates"]; judges = cfg["judges"]; samples = cfg.get("samples", 1)
    results = []
    rng = random.Random(cfg.get("seed", 7))
    jcount = 0
    for idx in tasks:
        td = f"{TASKS_ROOT}/{idx}"
        gold = glob.glob(td + "/gold/*")[0]
        gext = gold.rsplit(".", 1)[-1].lower()
        prompt = open(td + "/prompt.txt").read()
        # 1) generate + render every candidate; render gold
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
        # 2) all-pairs blind pairwise, rotating judge
        players = list(renders.keys())
        for a, b in itertools.combinations(players, 2):
            for _s in range(samples):
                judge = judges[jcount % len(judges)]; jcount += 1
                pair = [(a, renders[a]), (b, renders[b])]; rng.shuffle(pair)
                w, reason = judge_pair(prompt, pair[0][1], pair[1][1], judge)
                if w == "A": winner = pair[0][0]
                elif w == "B": winner = pair[1][0]
                else: winner = "tie"
                results.append({"task": idx, "a": a, "b": b, "shownA": pair[0][0], "shownB": pair[1][0],
                                "judge": judge["name"], "winner": winner, "reason": reason})
                print(f"  [judge {judge['name']}] task{idx}: {a} vs {b} -> {winner}", flush=True)
    # 3) Bradley-Terry Elo
    items = sorted({p for r in results for p in (r["a"], r["b"])})
    comps = []
    for r in results:
        wa = 1.0 if r["winner"] == r["a"] else 0.0 if r["winner"] == r["b"] else 0.5
        comps.append((r["a"], r["b"], wa))
    elo, strengths = bradley_terry_elo(items, comps) if comps else ({}, {})
    json.dump({"results": results, "elo": elo}, open(os.path.expanduser("~/gdpval-bench/bench_results.json"), "w"), indent=1)
    print("\n==== GDPval-AA (local) ELO leaderboard  (human anchored = 1000) ====")
    for it, e in sorted(elo.items(), key=lambda x: -x[1]):
        wins = sum(1 for r in results if r["winner"] == it)
        n = sum(1 for r in results if it in (r["a"], r["b"]))
        print(f"  {e:7.1f}  {it:32s}  ({wins}/{n} pairwise wins)")
    print(f"\ntotal comparisons: {len(results)}  | judges: {[j['name'] for j in judges]}")

if __name__ == "__main__":
    main()
