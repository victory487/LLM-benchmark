import os, glob, shutil, asyncio
from stirrup import Agent
from stirrup.clients import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider

idx="0"; model="sce397b_nex-ornith_topk100"; base="http://10.0.3.56:8000/v1"
td=os.path.expanduser("~/gdpval-bench/tasks/"+idx)
wd=td+"/stirrup_"+model
if os.path.exists(wd): shutil.rmtree(wd)
os.makedirs(wd)
sb=wd+"/sb"; os.makedirs(sb)
prompt=open(td+"/prompt.txt").read()
refs=glob.glob(td+"/refs/*")
client=ChatCompletionsClient(model=model, base_url=base, api_key="x", max_tokens=12000,
        kwargs={"extra_body":{"chat_template_kwargs":{"enable_thinking":False}}})
agent=Agent(client, name="gdpval_worker", max_turns=16,
        system_prompt="You are an experienced professional. Read the reference file(s) in your working directory, produce the requested deliverable, save it as deliverable.xlsx in the working directory, then call finish with paths=[\"deliverable.xlsx\"].",
        tools=[LocalCodeExecToolProvider(temp_base_dir=sb)])
async def main():
    async with agent.session(output_dir=wd, input_files=refs) as session:
        return await session.run(prompt)
finish, msgs, meta = asyncio.run(main())
print("=== RESULT ===", flush=True)
print("FINISH.paths:", getattr(finish,"paths",None), flush=True)
seen=False
for label,base_dir in [("OUTDIR",wd),("SANDBOX_BASE",sb)]:
    for root,dd,ff in os.walk(base_dir):
        for f in ff:
            print("FILE[%s]:"%label, os.path.join(root,f).replace(base_dir,"."), os.path.getsize(os.path.join(root,f)), flush=True); seen=True
if not seen: print("NO FILES ANYWHERE", flush=True)
