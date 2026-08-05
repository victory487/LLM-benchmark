#!/usr/bin/env python3
"""Full-chain single run: generate a deliverable with a candidate model (agentic tool loop),
then blind-judge candidate vs human gold with the local Gemma3 VLM judge.
Usage: run_one.py <task_idx> <model_name> <base_url>
"""
import os, sys, json, glob, shutil, subprocess, random, re
sys.path.insert(0, os.path.expanduser("~/gdpval-bench"))
from gdpval_validate import render, judge, parse          # reuse rendering + judge
from openai import OpenAI

TASKS_ROOT = os.path.expanduser("~/gdpval-bench/tasks")
PYBIN = os.path.expanduser("~/gdpval-bench/.venv/bin/python")

RUN_PY = {"type": "function", "function": {
    "name": "run_python",
    "description": ("Execute Python in the working directory to build the deliverable. Available: "
                    "pandas, numpy, openpyxl, xlsxwriter, python-docx (docx), python-pptx (pptx), "
                    "matplotlib. Files you write persist. Returns stdout+stderr."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
RUN_SH = {"type": "function", "function": {
    "name": "run_shell",
    "description": "Run a bash command in the working dir (e.g. `soffice --headless --convert-to pdf x.docx`, `ls`).",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}
FINISH = {"type": "function", "function": {
    "name": "finish",
    "description": "Call when the final deliverable file(s) are complete in the working directory.",
    "parameters": {"type": "object", "properties": {"deliverable_files": {"type": "array", "items": {"type": "string"}}},
                   "required": ["deliverable_files"]}}}

def run_python(code, wd):
    open(os.path.join(wd, "_step.py"), "w").write(code)
    try:
        p = subprocess.run([PYBIN, "_step.py"], cwd=wd, capture_output=True, text=True, timeout=200)
        return f"stdout:\n{(p.stdout or '')[-3000:]}\nstderr:\n{(p.stderr or '')[-2000:]}"
    except subprocess.TimeoutExpired:
        return "ERROR: python timed out (200s)"

def run_shell(cmd, wd):
    try:
        p = subprocess.run(cmd, cwd=wd, shell=True, capture_output=True, text=True, timeout=200)
        return f"stdout:\n{(p.stdout or '')[-2500:]}\nstderr:\n{(p.stderr or '')[-1500:]}"
    except subprocess.TimeoutExpired:
        return "ERROR: shell timed out"

def _deliverable_files(wd, refs):
    out = []
    for f in glob.glob(wd + "/*"):
        b = os.path.basename(f)
        if b in refs or b.startswith("_") or os.path.isdir(f):
            continue
        out.append(b)
    # prefer a file literally named deliverable.*
    out.sort(key=lambda b: (not b.lower().startswith("deliverable"), b))
    return out

def _has_deliverable(wd, refs, ext):
    return any(b.lower().startswith("deliverable") for b in _deliverable_files(wd, refs))

def _extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return m.group(1) if m else ""

def generate(idx, model, base_url, expected_ext, max_turns=12):
    td = f"{TASKS_ROOT}/{idx}"
    prompt = open(td + "/prompt.txt").read()
    wd = f"{td}/cand_{model}"
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    refs = []
    for rf in glob.glob(td + "/refs/*"):
        shutil.copy(rf, wd); refs.append(os.path.basename(rf))
    cli = OpenAI(base_url=base_url, api_key="x")
    sysmsg = ("You are an experienced professional completing a real work task, producing a polished "
              "deliverable file. IMPORTANT CONSTRAINTS:\n"
              "- You have NO internet access. Do not attempt web requests / urllib / requests; they will fail. "
              "Rely only on the reference files and your own expertise.\n"
              f"- Your VERY FIRST run_python call must write a COMPLETE 'deliverable.{expected_ext}' "
              "(relative path, in the current working directory). Never write to /tmp or absolute paths.\n"
              "- After the complete file exists you may refine it, then call finish.\n"
              f"Reference files already in the working directory: {refs}. "
              "Tools available: run_python (pandas/openpyxl/xlsxwriter/python-docx=docx/python-pptx=pptx/"
              "matplotlib installed). Do genuine, well-formatted work — never a stub or placeholder.")
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}]
    tools = [RUN_PY, FINISH]
    final = None
    used = 0
    for t in range(max_turns):
        used = t + 1
        if t >= 2 and not _has_deliverable(wd, refs, expected_ext):
            msgs.append({"role": "user", "content":
                         f"You still have not created the file. In your NEXT run_python call, write a "
                         f"COMPLETE 'deliverable.{expected_ext}' now using your own knowledge + the reference "
                         f"files. Do NOT search the web."})
        r = cli.chat.completions.create(model=model, messages=msgs, tools=tools,
                                        tool_choice="auto", temperature=0.3, max_tokens=8000,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        m = r.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        called = []
        if not m.tool_calls:
            code = _extract_code(m.content or "")
            if code:
                called = ["run_python(fallback)"]
                res = run_python(code, wd)
                msgs.append({"role": "user", "content":
                             f"(auto-executed the code from your message)\n{res[:3000]}\n"
                             f"If 'deliverable.{expected_ext}' now exists and is complete, call finish."})
            else:
                msgs.append({"role": "user", "content":
                             f"You must CALL the run_python tool (do not answer in prose). Create "
                             f"'deliverable.{expected_ext}' now with run_python."})
        else:
            for tc in m.tool_calls:
                fn = tc.function.name; called.append(fn)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                if fn == "run_python":
                    res = run_python(args.get("code", ""), wd)
                elif fn == "finish":
                    if _has_deliverable(wd, refs, expected_ext):
                        final = args.get("deliverable_files"); res = "acknowledged"
                    else:
                        res = (f"Cannot finish yet: no 'deliverable.{expected_ext}' file exists. "
                               "Create it with run_python first.")
                else:
                    res = "tool not available; use run_python"
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res[:3500]})
        print(f"  turn{used}: tools={called} deliverable_exists={_has_deliverable(wd, refs, expected_ext)}", flush=True)
        if final is not None:
            break
    json.dump(msgs, open(wd + "/_transcript.json", "w"), default=str, indent=1)
    produced = _deliverable_files(wd, refs)
    print(f"[gen] task{idx} {model}: turns={used} finish={final} produced={produced}")
    return wd, produced

def main():
    idx, model, base = sys.argv[1], sys.argv[2], sys.argv[3]
    td = f"{TASKS_ROOT}/{idx}"
    gold = glob.glob(td + "/gold/*")[0]
    expected_ext = gold.rsplit(".", 1)[-1].lower()
    wd, produced = generate(idx, model, base, expected_ext)
    if not produced:
        print("RESULT: NO DELIVERABLE PRODUCED"); return
    prompt = open(td + "/prompt.txt").read()
    goldext = gold.rsplit(".", 1)[-1].lower()
    cand = next((os.path.join(wd, b) for b in produced if b.rsplit(".", 1)[-1].lower() == goldext), None)
    if not cand:
        cand = os.path.join(wd, produced[0])
    print(f"[judge] gold={os.path.basename(gold)}  vs  cand={os.path.basename(cand)}")
    try:
        img_gold = render(gold, td + "/render_gold")
        img_cand = render(cand, wd + "/render_cand")
    except Exception as e:
        print(f"RENDER-FAIL: {e}"); return
    files = {"gold": img_gold, "cand": img_cand}
    labels = list(files.items()); random.shuffle(labels)
    raw = judge(prompt, labels[0][1], labels[1][1])
    w, reason = parse(raw)
    picked = labels[0][0] if w == "A" else (labels[1][0] if w == "B" else w)
    verdict = ("candidate BEATS human gold" if picked == "cand"
               else "candidate LOSES to human gold" if picked == "gold" else "TIE")
    print(f"RAW: {raw[:250]}")
    print(f"shownA={labels[0][0]} shownB={labels[1][0]} winner={w} => picked={picked}")
    print(f"RESULT: {verdict}")
    print(f"reason: {reason}")

if __name__ == "__main__":
    main()
