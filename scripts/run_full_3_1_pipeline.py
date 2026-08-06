#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

def run_template(template: str, **values):
    return subprocess.check_call([part.format(**values) for part in template.split()])

def main(argv=None):
    p=argparse.ArgumentParser(description='Run 3.1.1 -> 3.1.2 -> 3.1.3 through module CLIs and JSON files.')
    p.add_argument('--image',required=True); p.add_argument('--query',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--cmd-311',required=True); p.add_argument('--cmd-312',required=True); p.add_argument('--cmd-313',required=True)
    a=p.parse_args(argv); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    garment_json=out/'3_1_1_garments.json'; local_json=out/'3_1_2_locals.json'; attr_json=out/'3_1_3_attributes.json'
    run_template(a.cmd_311,image=a.image,query=a.query,output_dir=str(out),garment_json=str(garment_json))
    garments=json.loads(garment_json.read_text()); instances=garments.get('instances',garments if isinstance(garments,list) else [])
    if not instances:
        final={'garment_instances':[],'warnings':['3.1.1 returned no garment'],'design_attributes':[]}; (out/'full_3_1_result.json').write_text(json.dumps(final,indent=2)); return 0
    selected=max(instances,key=lambda r:float(r.get('confidence',r.get('score',0.0))))
    run_template(a.cmd_312,image=a.image,query=a.query,output_dir=str(out),garment_json=str(garment_json),local_json=str(local_json))
    locals_=json.loads(local_json.read_text()); parts=locals_.get('instances',[])
    if not parts:
        final={'garment_instances':instances,'selected_garment':selected,'query':a.query,'local_parts':[],'warnings':['3.1.2 returned no local part'],'design_attributes':[]}; (out/'full_3_1_result.json').write_text(json.dumps(final,indent=2)); return 0
    run_template(a.cmd_313,image=a.image,query=a.query,output_dir=str(out),local_json=str(local_json),attribute_json=str(attr_json))
    attrs=json.loads(attr_json.read_text())
    final={'garment_instances':instances,'selected_garment':selected,'query':a.query,'local_parts':parts,'selected_local_part':parts[0],'design_attributes':attrs.get('predictions',[]),'warnings':[],'runtime_metadata':{'orchestrator':'subprocess_json_v1'}}
    (out/'full_3_1_result.json').write_text(json.dumps(final,indent=2,ensure_ascii=False)); print(json.dumps(final,indent=2,ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
