#!/usr/bin/env python3
"""Validate the JUDGE half of the GDPval-AA-style pipeline.
For each task dir: render the real gold deliverable, build a deliberately-bad stub,
render it, then ask the local Gemma3 VLM judge to do a blind pairwise pick.
Success = pipeline runs end-to-end AND judge prefers the real gold over the stub.
"""
import os, sys, glob, json, base64, random, subprocess, tempfile
from PIL import Image
from openai import OpenAI

JUDGE_BASE  = os.environ.get("JUDGE_BASE",  "http://10.0.3.52:8010/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma3-judge")
TASKS_ROOT  = os.path.expanduser("~/gdpval-bench/tasks")

# ---------- rendering: any office file -> one stitched tall PNG ----------
def to_pdf(src, outdir):
    ext = src.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return src
    prof = f"file://{tempfile.mkdtemp(prefix='lo_')}"
    subprocess.run(["soffice", "--headless", f"-env:UserInstallation={prof}",
                    "--convert-to", "pdf", "--outdir", outdir, src],
                   check=True, timeout=240,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = os.path.splitext(os.path.basename(src))[0]
    pdf = os.path.join(outdir, base + ".pdf")
    if not os.path.exists(pdf):
        raise RuntimeError(f"soffice produced no pdf for {src}")
    return pdf

def render(src, workdir, maxpages=4, dpi=110, maxw=1000):
    os.makedirs(workdir, exist_ok=True)
    pdf = to_pdf(src, workdir)
    prefix = os.path.join(workdir, "pg")
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-l", str(maxpages), pdf, prefix],
                   check=True, timeout=240)
    pngs = sorted(glob.glob(prefix + "*.png"))
    if not pngs:
        raise RuntimeError(f"no pages rendered for {src}")
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    imgs = [i.resize((maxw, int(i.height * maxw / i.width))) if i.width > maxw else i for i in imgs]
    W = max(i.width for i in imgs); H = sum(i.height for i in imgs)
    canvas = Image.new("RGB", (W, H), "white"); y = 0
    for i in imgs:
        canvas.paste(i, (0, y)); y += i.height
    out = os.path.join(workdir, "stitched.png")
    canvas.save(out, "PNG")
    return out

def b64url(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

# ---------- judge ----------
SYS = ("You are an expert occupational grader. You will see a work TASK and two candidate "
       "deliverables, A and B, each rendered as an image of its pages. Decide which deliverable "
       "better fulfills the task, weighing accuracy, completeness, instruction-following and "
       "presentation/formatting. Be decisive. Respond with ONLY a JSON object: "
       '{"winner":"A" or "B" or "tie","reason":"<= 40 words"}.')

def judge(prompt, imgA, imgB):
    cli = OpenAI(base_url=JUDGE_BASE, api_key="x")
    content = [
        {"type": "text", "text": f"TASK:\n{prompt[:3500]}\n\n=== Deliverable A ==="},
        {"type": "image_url", "image_url": {"url": b64url(imgA)}},
        {"type": "text", "text": "=== Deliverable B ==="},
        {"type": "image_url", "image_url": {"url": b64url(imgB)}},
        {"type": "text", "text": "Which deliverable is better? Reply JSON only."},
    ]
    r = cli.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": content}],
        temperature=0.0, max_tokens=300)
    return r.choices[0].message.content

def parse(raw):
    try:
        j = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return j.get("winner"), j.get("reason")
    except Exception as e:
        return None, f"PARSE-FAIL {e}"

# ---------- driver ----------
def main():
    task_dirs = [d for d in sorted(glob.glob(TASKS_ROOT + "/*")) if os.path.isdir(d)]
    print(f"judge endpoint: {JUDGE_BASE} model={JUDGE_MODEL}")
    ok = 0; total = 0
    for td in task_dirs:
        idx = os.path.basename(td)
        golds = glob.glob(td + "/gold/*")
        if not golds:
            continue
        gold = golds[0]
        prompt = open(td + "/prompt.txt").read()
        print(f"\n===== task {idx} | gold={os.path.basename(gold)} =====")
        # build a clearly-inferior stub deliverable
        stub_txt = td + "/stub.txt"
        open(stub_txt, "w").write("DRAFT PLACEHOLDER\n\nThe deliverable was not completed. "
                                  "No analysis, data, or formatting was produced.")
        try:
            img_gold = render(gold, td + "/render_gold")
            img_stub = render(stub_txt, td + "/render_stub")
            print(f"  rendered gold+stub OK")
        except Exception as e:
            print(f"  RENDER-FAIL: {e}"); continue
        # blind pairwise: randomize which physical file is shown as A vs B
        files = {"gold": img_gold, "stub": img_stub}
        labels = list(files.items()); random.shuffle(labels)
        shownA, shownB = labels[0], labels[1]
        try:
            raw = judge(prompt, shownA[1], shownB[1])
        except Exception as e:
            print(f"  JUDGE-CALL-FAIL: {e}"); continue
        w, reason = parse(raw)
        total += 1
        if w == "A":
            picked = shownA[0]
        elif w == "B":
            picked = shownB[0]
        else:
            picked = w
        correct = (picked == "gold")
        ok += correct
        print(f"  RAW: {raw[:200]}")
        print(f"  -> shownA={shownA[0]} shownB={shownB[0]} | winner={w} => picked={picked} "
              f"| {'CORRECT (prefers gold)' if correct else 'WRONG/tie'}")
        print(f"  reason: {reason}")
    print(f"\n==== JUDGE VALIDATION: {ok}/{total} tasks judge correctly preferred gold ====")

if __name__ == "__main__":
    main()
