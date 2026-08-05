#!/usr/bin/env python3
"""Round-1 snapshot: judge ONLY sample s0 of each settled task. Config from aligned.yaml.
"""
import os, sys, glob, json, random, concurrent.futures
sys.path.insert(0, os.path.expanduser("~/gdpval-bench"))
import aligned_config
_argv, _cfg_path = aligned_config.pop_config_args()
# Import after config path is in env so judge module loads the same yaml.
from judge_aligned_gpt55 import render_pages, xlsx_text, judge, gold_file, bootstrap_ci, TR, MODEL, WORKERS, CFG

RESULTS_FILE = aligned_config.results_path(CFG, "round1")

def s0_state(idx):
    wd = f"{TR}/{idx}/cand_{MODEL}_aligned/s0"
    if glob.glob(wd + "/deliverable.*"):
        return "produced"
    if os.path.isdir(wd) and not os.path.isdir(wd + "/sb"):
        return "empty"
    return "pending"

def score_task(idx):
    gold = gold_file(idx)
    if not gold:
        return None
    st = s0_state(idx)
    if st == "pending":
        return None
    if st == "empty":
        return (idx, 0.0, "empty")
    grd = f"{TR}/{idx}/render_gold_full"
    g_imgs = render_pages(gold, grd); g_txt = xlsx_text(gold)
    if not g_imgs and not g_txt:
        return None
    sf = glob.glob(f"{TR}/{idx}/cand_{MODEL}_aligned/s0/deliverable.*")[0]
    s_imgs = render_pages(sf, f"{TR}/{idx}/cand_{MODEL}_aligned/s0/render_full")
    s_txt = xlsx_text(sf)
    if not s_imgs and not s_txt:
        return (idx, 0.0, "unrenderable")
    prompt = open(f"{TR}/{idx}/prompt.txt").read()
    pair = [("cand", s_imgs, s_txt), ("human", g_imgs, g_txt)]
    random.Random(int(idx) * 10).shuffle(pair)
    w, _reason = judge(prompt, pair[0][1], pair[0][2], pair[1][1], pair[1][2])
    win = pair[0][0] if w == "A" else pair[1][0] if w == "B" else "tie"
    return (idx, 1.0 if win == "cand" else 0.0 if win == "human" else 0.5, "judged")

def main():
    tasks = aligned_config.tasks_list(CFG)
    settled = [t for t in tasks if gold_file(t) and s0_state(t) != "pending"]
    print(f"round-1 config={CFG['_path']} model={MODEL}", flush=True)
    print(f"round-1 snapshot: s0 settled on {len(settled)} tasks "
          f"(of {sum(1 for t in tasks if gold_file(t))} gold tasks)", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(score_task, settled)):
            if r:
                results.append(r)
            if i % 20 == 0:
                print(f"  judged {i}/{len(settled)}", flush=True)
    vals = [r[1] for r in results]
    n_empty = sum(1 for r in results if r[2] in ("empty", "unrenderable"))
    wr = sum(vals) / len(vals) if vals else 0
    lo, hi = bootstrap_ci(vals)
    prod_vals = [r[1] for r in results if r[2] == "judged"]
    pwr = sum(prod_vals) / len(prod_vals) if prod_vals else 0
    print(f"\n==== ROUND-1 (s0 only) {MODEL} vs HUMAN — INTERIM, {len(results)} settled tasks ====")
    print(f"  overall win-rate (empty=loss): {wr*100:.0f}%  (95% CI {lo*100:.0f}-{hi*100:.0f}%)")
    print(f"  produced-only win-rate:        {pwr*100:.0f}%  ({len(prod_vals)} tasks)")
    print(f"  empty/unrenderable: {n_empty}/{len(results)}")
    print(f"  results -> {RESULTS_FILE}", flush=True)
    json.dump([{"task": r[0], "score": r[1], "kind": r[2]} for r in results],
              open(RESULTS_FILE, "w"), indent=1)

if __name__ == "__main__":
    main()
