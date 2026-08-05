from datasets import load_dataset
import os, urllib.request, concurrent.futures
ds=load_dataset("openai/gdpval",split="train")
base=os.path.expanduser("~/gdpval-bench/tasks")
def dl(a):
    url,path=a
    if os.path.exists(path) and os.path.getsize(path)>0: return "cached"
    try: urllib.request.urlretrieve(url,path); return "ok"
    except Exception as e: return "fail"
jobs=[]
for i,r in enumerate(ds):
    d="%s/%d"%(base,i); os.makedirs(d+"/gold",exist_ok=True); os.makedirs(d+"/refs",exist_ok=True)
    open(d+"/prompt.txt","w").write(r["prompt"])
    for u,fn in zip(r.get("deliverable_file_urls") or [], r.get("deliverable_files") or []):
        jobs.append((u, d+"/gold/"+os.path.basename(fn)))
    for u,fn in zip(r.get("reference_file_urls") or [], r.get("reference_files") or []):
        jobs.append((u, d+"/refs/"+os.path.basename(fn)))
print("tasks:",len(ds),"files to fetch:",len(jobs),flush=True)
from collections import Counter
c=Counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for res in ex.map(dl,jobs): c[res]+=1
print("DL RESULT:",dict(c),flush=True)
