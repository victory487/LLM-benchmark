#!/usr/bin/env python3
"""Re-judge ONLY topk050 with a 3-judge panel (Gemma3-12B + Qwen3-VL-30B + InternVL3-78B).
Reuses the cached rendered deliverables from the full run (no re-generation). Judges topk050
against the human gold and against the other two candidates, each matchup scored by all 3
panel judges. Reports per-judge and majority-vote win rates + a focused Elo (human anchored=1000).
"""
import os, glob, json, random, concurrent.futures
from gdpval_bench_stirrup import judge_pair, bradley_terry_elo, TASKS_ROOT

FOCUS = "sce397b_nex-ornith_topk050"
OTHERS = ["Nex-N2-Pro", "sce397b_nex-ornith_topk100"]
CFG = json.load(open(os.path.expanduser("~/gdpval-bench/config.json")))
PANEL = CFG["panel_judges"]          # list of {name, base, model, nothink}
TASKS = CFG["tasks"]

def gold_render(idx):
    p = f"{TASKS_ROOT}/{idx}/render_gold/stitched.png"
    return p if os.path.exists(p) else None

def cand_render(idx, name):
    p = f"{TASKS_ROOT}/{idx}/cand_{name}/render_cand/stitched.png"
    return p if os.path.exists(p) else None

def build_jobs():
    jobs = []; rng = random.Random(7)
    for idx in TASKS:
        fr = cand_render(idx, FOCUS)
        if not fr:
            continue
        prompt = open(f"{TASKS_ROOT}/{idx}/prompt.txt").read()
        opps = {}
        g = gold_render(idx)
        if g:
            opps["human"] = g
        for o in OTHERS:
            r = cand_render(idx, o)
            if r:
                opps[o] = r
        for oppname, oppimg in opps.items():
            for judge in PANEL:
                pair = [(FOCUS, fr), (oppname, oppimg)]; rng.shuffle(pair)
                jobs.append((idx, prompt, pair, judge))
    return jobs

def do(job):
    idx, prompt, pair, judge = job
    try:
        w, _ = judge_pair(prompt, pair[0][1], pair[1][1], judge)
    except Exception:
        w = None
    winner = pair[0][0] if w == "A" else pair[1][0] if w == "B" else "tie"
    return {"task": idx, "a": pair[0][0], "b": pair[1][0], "judge": judge["name"], "winner": winner}

def matchup(results, opp):
    rs = [r for r in results if {r["a"], r["b"]} == {FOCUS, opp}]
    by_judge = {}
    for jn in sorted(set(r["judge"] for r in rs)):
        jrs = [r for r in rs if r["judge"] == jn]
        w = sum(1 for r in jrs if r["winner"] == FOCUS)
        by_judge[jn] = f"{w}/{len(jrs)}"
    tasks = set(r["task"] for r in rs); mw = mn = 0
    for t in tasks:
        trs = [r for r in rs if r["task"] == t]
        votes = sum(1 for r in trs if r["winner"] == FOCUS)
        mn += 1
        if votes * 2 > len(trs):
            mw += 1
    return by_judge, mw, mn

def main():
    jobs = build_jobs()
    print(f"panel: {[j['name'] for j in PANEL]} | rejudge jobs: {len(jobs)}", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(do, jobs)):
            results.append(r)
            if i % 100 == 0:
                print(f"  judged {i}/{len(jobs)}", flush=True)
    json.dump(results, open(os.path.expanduser("~/gdpval-bench/rejudge_topk050_results.json"), "w"), indent=1)
    print(f"\n==== topk050 re-judged by 3-panel {[j['name'] for j in PANEL]} ====")
    for opp in ["human"] + OTHERS:
        bj, mw, mn = matchup(results, opp)
        if mn:
            print(f"  topk050 vs {opp:28s}: majority {mw}/{mn} ({100*mw/mn:.0f}%)  | per-judge {bj}")
        else:
            print(f"  topk050 vs {opp:28s}: no shared tasks")
    items = sorted({p for r in results for p in (r["a"], r["b"])})
    comps = [(r["a"], r["b"], 1.0 if r["winner"] == r["a"] else 0.0 if r["winner"] == r["b"] else 0.5) for r in results]
    elo = bradley_terry_elo(items, comps) if comps else {}
    print("\n  focused Elo (topk050 matchups only, human anchored=1000):")
    for it, e in sorted(elo.items(), key=lambda x: -x[1]):
        print(f"    {e:8.1f}  {it}")

if __name__ == "__main__":
    main()
