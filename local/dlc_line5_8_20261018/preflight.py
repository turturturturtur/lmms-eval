"""Validate actual model artifacts, task data, and metric assets before CreateJob."""
import hashlib
import json
import math
from pathlib import Path
import sys

WORK = Path('/mnt/cpfsA/user/tianleniu/Innovator-Tune/lmms-eval/local/dlc_line5_8_20261018')
SOURCE = WORK / 'source'
sys.path.insert(0, str(SOURCE))


def leaves(tree):
    for name, obj in tree.items():
        if isinstance(obj, dict):
            yield from leaves(obj)
        else:
            yield name, obj[-1] if isinstance(obj, tuple) else obj


def main():
    from transformers import AutoTokenizer
    from vllm.model_executor.models import ModelRegistry
    from lmms_eval.storage_policy import validate_config, validate_paths
    from lmms_eval.runtime_preflight import check_runtime
    print('Runtime dependencies:', check_runtime(vllm=True, imports=True), flush=True)
    manifest = json.loads((WORK / 'manifest.json').read_text())
    models = []
    for row in manifest['jobs']:
        config = json.loads(Path(row['runtime_config']).read_text())
        validate_paths(config)
        validate_config(json.loads((Path(row['directory'])/'config_dlc.json').read_text()))
        m = config['model']; path = Path(m['path'])
        c = json.loads((path/'config.json').read_text())
        arch = c['architectures'][0]
        assert arch in ModelRegistry.get_supported_archs(), arch
        native = (c.get('text_config') or c.get('llm_config') or c)['max_position_embeddings']
        assert m['max_model_len'] <= native and m['enable_thinking'] is False
        assert config['eval']['limit'] == -1 and config['eval']['gen_kwargs'] == ''
        index = path/'model.safetensors.index.json'
        shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        assert all((path/f).is_file() and (path/f).stat().st_size > 0 for f in shards)
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        assert tokenizer.eos_token_id in m['stop_token_ids']
        rendered = tokenizer.apply_chat_template([{'role':'user','content':'Reply with OK.'}],
            chat_template=Path(m['chat_template']).read_text(), tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        if '<|channel>thought' in rendered:
            assert rendered.endswith('<|channel>thought\n<channel|>'), repr(rendered[-150:])
        assert '<|think|>' not in rendered
        models.append({'row':row['row'],'architecture':arch,'native_context':native,
            'weight_shards':len(shards),'thinking':False,'prompt_suffix':rendered[-150:]})
        print(json.dumps(models[-1]),flush=True)

    from lmms_eval.tasks import TaskManager, get_task_dict
    manager = TaskManager(verbosity='ERROR', model_name='vllm_backend')
    benchmarks = []
    for task in ['superchem_en','superchem_cn','SmolInstruct_v2']:
        members = []
        for name,obj in leaves(get_task_dict([task], task_manager=manager)):
            docs = obj.test_docs() if obj.has_test_docs() else obj.validation_docs()
            count = len(docs)
            assert count > 0
            for i in sorted({0,count//2,count-1}):
                doc = docs[i]
                assert obj.doc_to_text(doc) and obj.doc_to_target(doc) is not None
                obj.doc_to_visual(doc)
            members.append({'task':name,'samples':count})
        if task.startswith('superchem_'):
            assert len(members)==1 and members[0]['samples']==500
        else:
            assert len(members)==14 and sum(x['samples'] for x in members)==33061
        benchmarks.append({'benchmark':task,'members':members})
        print(json.dumps({'benchmark':task,'leaf_tasks':len(members),'samples':sum(x['samples'] for x in members)}),flush=True)

    from lmms_eval.tasks.SmolInstruct_v2.utils import smol_aggregate_results
    fixtures = [('forward_synthesis',['CCO','CC'],['CCO','CC'],1.0),
                ('property_prediction-bbbp',['Yes','No'],['Yes','No'],1.0),
                ('molecule_captioning',['A primary alcohol.','A simple alkane.'],
                    ['A primary alcohol.','A simple alkane.'],1.0)]
    for task,preds,golds,expected in fixtures:
        score=smol_aggregate_results([{'task':task,'pred':p,'gold':g} for p,g in zip(preds,golds)])
        assert math.isclose(score,expected,abs_tol=1e-10),(task,score)
    report={'passed':True,'models':models,'benchmarks':benchmarks,'metric_smokes':len(fixtures),
            'source_manifest_sha256':hashlib.sha256((SOURCE/'source_manifest.json').read_bytes()).hexdigest()}
    (WORK/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':
    main()
