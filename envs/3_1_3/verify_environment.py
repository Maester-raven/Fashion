import importlib,json
mods=['torch','numpy','PIL']
rows=[]
for m in mods:
    try:
        x=importlib.import_module(m); rows.append({'module':m,'ok':True,'version':getattr(x,'__version__',None),'file':getattr(x,'__file__',None)})
    except Exception as e: rows.append({'module':m,'ok':False,'error':repr(e)})
print(json.dumps({'passed':all(r['ok'] for r in rows),'modules':rows},indent=2))
