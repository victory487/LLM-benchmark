#!/usr/bin/env python3
"""Full 220-task GDPval-AA run with the real Stirrup harness, parallelized.
Phase 1: generation via ProcessPoolExecutor (each candidate x task in its own process — isolated
         stdout/sandbox, resumable via on-disk deliverable cache).
Phase 2: render deliverables (thread pool). Phase 3: panel judging (thread pool). Then Elo.
Env: BRAVE_API_KEY. Config: ~/gdpval-bench/config.json (tasks list, candidates, judges, samples).
"""
import os, glob, json, itertools, random, concurrent.futures
from gdpval_bench_stirrup import generate, judge_pair, bradley_terry_elo, _has_deliv, TASKS_ROOT
from gdpval_validate import render

CFG = json.load(open(os.path.expanduser("~/gdpval-bench/config.json")))
CANDS, JUDGES = CFG["candidates"], CFG["judges"]
SAMPLES = CFG.get("samples", 1)
TASKS = CFG["tasks"]
GEN_WORKERS = CFG.get("gen_workers", 12)
JUDGE_WORKERS = CFG.get("judge_workers", 8)

def _task_ext(idx):
    g = glob.glob(f"{TASKS_ROOT}/{idx}/gold/*")
    return (g[0].rsplit(".", 1)[-1].lower(), g[0]) if g else (None, None)

def _refnames(idx):
    return [os.path.basename(f) for f in glob.glob(f"{TASKS_ROOT}/{idx}/refs/*")]

# ---------------- phase 1: parallel generation ----------------
def phase1():
    jobs = []
    for idx in TASKS:
        ext, gold = _task_ext(idx)
        if not gold:
            continue
        rn = _refnames(idx)
        for name, base in CANDS.items():
            wd = f"{TASKS_ROOT}/{idx}/cand_{name}"
            if os.path.exists(wd) and _has_deliv(wd, rn):
                continue
            jobs.append((str(idx), name, base, ext))
    print(f"PHASE1 generation jobs (uncached): {len(jobs)}", flush=True)
    if not jobs:
        return
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=GEN_WORKERS) as ex:
        futs = {ex.submit(generate, *j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            done += 1; j = futs[fut]
            try:
                _, prod = fut.result()
                print(f"  gen {done}/{len(jobs)} t{j[0]} {j[1]}: {'OK' if prod else 'EMPTY'}", flush=True)
            except Exception as e:
                print(f"  gen {done}/{len(jobs)} t{j[0]} {j[1]}: ERR {str(e)[:70]}", flush=True)
    print("PHASE1 done", flush=True)

# ---------------- phase 2: render ----------------
def render_task(idx):
    ext, gold = _task_ext(idx)
    if not gold:
        return idx, {}
    rn = _refnames(idx); rmap = {}
    try:
        rmap["human"] = render(gold, f"{TASKS_ROOT}/{idx}/render_gold")
    except Exception:
        return idx, {}
    for name in CANDS:
        wd = f"{TASKS_ROOT}/{idx}/cand_{name}"
        if not (os.path.exists(wd) and _has_deliv(wd, rn)):
            continue
        dl = [b for b in os.listdir(wd) if b.lower().startswith("deliverable.")]
        if not dl:
            continue
        try:
            rmap[name] = render(os.path.join(wd, dl[0]), wd + "/render_cand")
        except Exception:
            pass
    return idx, rmap

def phase2():
    renders = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for idx, rmap in ex.map(render_task, TASKS):
            renders[idx] = rmap
    ok = sum(1 for r in renders.values() if len(r) >= 2)
    print(f"PHASE2 render done: {ok}/{len(TASKS)} tasks have >=2 deliverables to compare", flush=True)
    return renders

# ---------------- phase 3: judging ----------------
def phase3(renders):
    rng = random.Random(CFG.get("seed", 7)); jcount = 0; jobs = []
    for idx in TASKS:
        rmap = renders.get(idx, {}); players = list(rmap)
        if len(players) < 2:
            continue
        prompt = open(f"{TASKS_ROOT}/{idx}/prompt.txt").read()
        for a, b in itertools.combinations(players, 2):
            for _s in range(SAMPLES):
                judge = JUDGES[jcount % len(JUDGES)]; jcount += 1
                pair = [(a, rmap[a]), (b, rmap[b])]; rng.shuffle(pair)
                jobs.append((idx, prompt, pair, judge))
    print(f"PHASE3 judge jobs: {len(jobs)}", flush=True)

    def do_judge(job):
        idx, prompt, pair, judge = job
        try:
            w, reason = judge_pair(prompt, pair[0][1], pair[1][1], judge)
        except Exception:
            w = None
        winner = pair[0][0] if w == "A" else pair[1][0] if w == "B" else "tie"
        return {"task": idx, "a": pair[0][0], "b": pair[1][0], "judge": judge["name"], "winner": winner}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        for i, r in enumerate(ex.map(do_judge, jobs)):
            results.append(r)
            if i % 100 == 0:
                print(f"  judged {i}/{len(jobs)}", flush=True)
    print("PHASE3 done", flush=True)
    return results

def main():
    phase1()
    renders = phase2()
    results = phase3(renders)
    items = sorted({p for r in results for p in (r["a"], r["b"])})
    comps = [(r["a"], r["b"], 1.0 if r["winner"] == r["a"] else 0.0 if r["winner"] == r["b"] else 0.5) for r in results]
    elo = bradley_terry_elo(items, comps) if comps else {}
    json.dump({"results": results, "elo": elo},
              open(os.path.expanduser("~/gdpval-bench/bench_full_results.json"), "w"), indent=1)
    print("\n==== GDPval-AA FULL (Stirrup harness, 220-task gold) ELO  (human anchored = 1000) ====", flush=True)
    for it, e in sorted(elo.items(), key=lambda x: -x[1]):
        wins = sum(1 for r in results if r["winner"] == it)
        n = sum(1 for r in results if it in (r["a"], r["b"]))
        print(f"  {e:8.1f}  {it:34s} ({wins}/{n})", flush=True)
    print(f"\ntotal comparisons: {len(results)} | judges: {[j['name'] for j in JUDGES]}", flush=True)

if __name__ == "__main__":
    main()
