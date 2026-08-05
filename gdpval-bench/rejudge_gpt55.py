#!/usr/bin/env python3
"""Re-judge topk050 with GPT-5.5 (an actual AA panel judge), AA-aligned + credibility fixes:
- Judge = gpt-5.5 via router (frontier, multimodal) — replaces the weak local VLMs.
- Full coverage + non-production = loss (AA/OpenAI口径): on a gold task, if the candidate produced
  no deliverable it LOSES to human (no survivorship bias); candidate-vs-candidate handled symmetrically.
- Bootstrap 95% CI on the vs-human win rate.
Reports win rate vs human / vs each candidate + CI + a focused Elo.
"""
import os, glob, json, base64, random, concurrent.futures
from openai import OpenAI

FOCUS = "sce397b_nex-ornith_topk050"
OTHERS = ["Nex-N2-Pro", "sce397b_nex-ornith_topk100"]
TR = os.path.expanduser("~/gdpval-bench/tasks")
CFG = json.load(open(os.path.expanduser("~/gdpval-bench/config.json")))
TASKS = CFG["tasks"]
JUDGE_BASE = "https://agrouter-ng-test.kiwiar.com/v1"
JUDGE_KEY = "sk-bfU2MvaVuZ0jW7wB53qJKw"
JUDGE_MODEL = "gpt-5.5"
WORKERS = 5

JUDGE_INSTR = ("You are an expert occupational grader. You are given a work TASK and two candidate "
               "deliverables A and B (rendered as page images). Decide which better fulfills the task, "
               "weighing accuracy, completeness, instruction-following and presentation. Be decisive. "
               'Respond ONLY with JSON: {"winner":"A" or "B" or "tie","reason":"<=25 words"}.')

def b64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

def gold_render(idx):
    p = f"{TR}/{idx}/render_gold/stitched.png"
    return p if os.path.exists(p) else None

def cand_render(idx, name):
    p = f"{TR}/{idx}/cand_{name}/render_cand/stitched.png"
    return p if os.path.exists(p) else None

_cli = OpenAI(base_url=JUDGE_BASE, api_key=JUDGE_KEY, timeout=120, max_retries=3)

def gpt55_judge(prompt, imgA, imgB):
    content = [{"type": "text", "text": JUDGE_INSTR + f"\n\nTASK:\n{prompt[:3000]}\n\n=== Deliverable A ==="},
               {"type": "image_url", "image_url": {"url": b64(imgA)}},
               {"type": "text", "text": "=== Deliverable B ==="},
               {"type": "image_url", "image_url": {"url": b64(imgB)}},
               {"type": "text", "text": "Which is better? JSON only."}]
    r = _cli.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": content}],
                                     max_tokens=900)
    raw = r.choices[0].message.content or ""
    try:
        j = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]); return j.get("winner")
    except Exception:
        return None

# ---- build matchups with non-production = loss ----
def build(opp):
    """Return list of (idx, kind) where kind in {'judge','auto_focus_win','auto_focus_loss'}."""
    items = []
    for idx in TASKS:
        fr = cand_render(idx, FOCUS)
        opp_r = gold_render(idx) if opp == "human" else cand_render(idx, opp)
        if opp == "human":
            if opp_r is None:
                continue  # no gold to anchor against -> skip
            if fr is None:
                items.append((idx, "auto_focus_loss"))       # focus produced nothing -> loses to human
            else:
                items.append((idx, "judge"))
        else:
            if fr is None and opp_r is None:
                continue
            elif fr is not None and opp_r is None:
                items.append((idx, "auto_focus_win"))         # focus produced, opp didn't
            elif fr is None and opp_r is not None:
                items.append((idx, "auto_focus_loss"))
            else:
                items.append((idx, "judge"))
    return items

def score_matchup(opp):
    items = build(opp)
    # judge the 'judge' ones (blind, randomized order) concurrently
    def do(item):
        idx, kind = item
        if kind == "auto_focus_win":
            return (idx, 1.0)
        if kind == "auto_focus_loss":
            return (idx, 0.0)
        fr = cand_render(idx, FOCUS)
        opp_r = gold_render(idx) if opp == "human" else cand_render(idx, opp)
        prompt = open(f"{TR}/{idx}/prompt.txt").read()
        pair = [(FOCUS, fr), (opp, opp_r)]; random.Random(idx).shuffle(pair)
        w = gpt55_judge(prompt, pair[0][1], pair[1][1])
        if w == "A":
            win = pair[0][0]
        elif w == "B":
            win = pair[1][0]
        else:
            win = "tie"
        return (idx, 1.0 if win == FOCUS else 0.0 if win == opp else 0.5)
    scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, s in enumerate(ex.map(do, items)):
            scores.append(s)
            if i % 40 == 0:
                print(f"  [{opp}] {i}/{len(items)}", flush=True)
    return scores  # list of (idx, focus_score in {1,0.5,0})

def bootstrap_ci(vals, iters=2000):
    if not vals:
        return (0, 0)
    n = len(vals); rng = random.Random(13); means = []
    for _ in range(iters):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])

def main():
    print(f"judge = {JUDGE_MODEL} (via router) | non-production = loss | full coverage", flush=True)
    all_scores = {}
    for opp in ["human"] + OTHERS:
        sc = score_matchup(opp)
        all_scores[opp] = sc
        vals = [s for _, s in sc]
        wr = sum(vals) / len(vals) if vals else 0
        lo, hi = bootstrap_ci(vals)
        wins = sum(1 for v in vals if v == 1.0)
        print(f"\n  topk050 vs {opp:28s}: win-rate {wr*100:.0f}%  (95% CI {lo*100:.0f}-{hi*100:.0f}%)  "
              f"[{wins}W/{sum(1 for v in vals if v==0.5)}T/{sum(1 for v in vals if v==0.0)}L of {len(vals)}]", flush=True)
    # focused Elo (BT) over all focus matchups
    comps = []
    for opp, sc in all_scores.items():
        for _, s in sc:
            comps.append((FOCUS, opp, s))
    import math
    items = sorted({p for c in comps for p in (c[0], c[1])})
    W = {i: 0.0 for i in items}; N = {i: {} for i in items}
    for a, b, wa in comps:
        W[a] += wa; W[b] += 1 - wa; N[a][b] = N[a].get(b, 0) + 1; N[b][a] = N[b].get(a, 0) + 1
    s = {i: 1.0 for i in items}
    for _ in range(500):
        ns = {}
        for i in items:
            den = sum(n / (s[i] + s[j]) for j, n in N[i].items())
            ns[i] = W[i] / den if den > 0 and W[i] > 0 else 1e-6
        gm = math.exp(sum(math.log(max(v, 1e-12)) for v in ns.values()) / len(ns))
        s = {i: ns[i] / gm for i in items}
    elo = {i: round(400 * math.log10(max(s[i], 1e-12)) + (1000 - 400 * math.log10(max(s["human"], 1e-12))), 1) for i in items}
    print("\n  focused Elo (GPT-5.5 judge, human anchored=1000):")
    for it, e in sorted(elo.items(), key=lambda x: -x[1]):
        print(f"    {e:8.1f}  {it}")
    json.dump({"scores": {k: v for k, v in all_scores.items()}, "elo": elo},
              open(os.path.expanduser("~/gdpval-bench/rejudge_gpt55_results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
