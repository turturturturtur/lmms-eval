import ast
import json
from pathlib import Path
from transformers import AutoTokenizer
from lmms_eval.tasks import TaskManager, get_task_dict

work = Path(__file__).resolve().parents[1]
repo = work.parents[1]
path = repo/'.venv/lib/python3.11/site-packages/vllm/transformers_utils/processors/internvl.py'
names = {'find_closest_aspect_ratio','get_internvl_target_ratios','calculate_internvl_targets'}
tree = ast.parse(path.read_text())
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
scope = {}; exec(compile(tree,str(path),'exec'),scope)
tok = AutoTokenizer.from_pretrained('/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_model_views/internvl35_row07_nothink_20261018',trust_remote_code=True)
manager = TaskManager(verbosity='ERROR', model_name='vllm_backend')
report = {}
for task in ['superchem_en','superchem_cn']:
    obj = get_task_dict([task],task_manager=manager)[task]
    entries = []
    for i,doc in enumerate(obj.test_docs()):
        visuals = obj.doc_to_visual(doc)
        estimates = {}
        for cap in [12,11,10,8,6,4,2,1]:
            patches = sum(scope['calculate_internvl_targets'](orig_width=im.width,orig_height=im.height,
                image_size=448,target_ratios=scope['get_internvl_target_ratios'](1,cap),use_thumbnail=True)[0] for im in visuals)
            estimates[str(cap)] = patches*256 + len(tok.encode(obj.doc_to_text(doc))) + 2*len(visuals)+30
        entries.append({'doc_id':i,'images':len(visuals),'sizes':[im.size for im in visuals],'estimated_input_tokens_by_patch_cap':estimates})
    report[task] = entries
    largest = sorted(entries,key=lambda x:x['estimated_input_tokens_by_patch_cap']['12'],reverse=True)[:5]
    print(json.dumps({'task':task,'largest':largest,'over_context':sum(x['estimated_input_tokens_by_patch_cap']['12']>=40960 for x in entries)}),flush=True)
(work/'diagnosis/input_audit.json').write_text(json.dumps(report,indent=2)+'\n')
