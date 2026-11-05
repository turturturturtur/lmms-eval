"""Check WebUI parity, complete task loading, and serialized requests before submission."""
import asyncio
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parent
SOURCE = WORK / 'source'
REPO = WORK.parents[2]


def main():
    from lmms_eval.tui.server import get_defaults
    from lmms_eval.llm_judge.task_policy import resolve_post_eval_judge_tasks
    from lmms_eval.runtime_preflight import check_runtime
    from lmms_eval.storage_policy import validate_config, validate_paths
    from transformers import AutoTokenizer
    from vllm.model_executor.models import ModelRegistry
    sys.path.insert(0,str(SOURCE / 'tools'))
    from gpu_gate_webui import prepare_requests, make_backend
    defaults = asyncio.run(get_defaults())
    manifest = json.loads((WORK / 'manifest.json').read_text())
    report = {'runtime':check_runtime(vllm=True,imports=True),'jobs':[]}
    for row in manifest['jobs']:
        cfg = json.loads(Path(row['runtime_config']).read_text())
        m, e = cfg['model'], cfg['eval']
        for key, field in [('tp','model_tp'),('max_model_len','max_model_len'),
                           ('gpu_memory_utilization','gpu_memory_utilization'),
                           ('max_num_seqs','max_num_seqs'),('base_port','base_port'),
                           ('enable_thinking','enable_thinking')]:
            assert m[key] == getattr(defaults,field), key
        for key in ['concurrency','gen_kwargs','limit','debug','batch_size']:
            assert e[key] == getattr(defaults,key), key
        validate_paths(cfg)
        validate_config(json.loads((Path(row['directory']) / 'config_dlc.json').read_text()))
        assert resolve_post_eval_judge_tasks(row['tasks']) == []
        path = Path(m['path'])
        model = json.loads((path / 'config.json').read_text())
        assert model['architectures'][0] in ModelRegistry.get_supported_archs()
        native = (model.get('text_config') or model.get('llm_config') or model)['max_position_embeddings']
        assert m['max_model_len'] <= native
        index = path / 'model.safetensors.index.json'
        shards = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else ['model.safetensors']
        assert all((path / x).is_file() and (path / x).stat().st_size for x in shards)
        tokenizer = AutoTokenizer.from_pretrained(str(path),trust_remote_code=True,local_files_only=True)
        assert tokenizer.eos_token_id in m['stop_token_ids']
        requests, selected, task_dict = prepare_requests(cfg)
        assert len(requests) == (14 if row['row'] == 6 else 8)
        assert sum(next(iter(v.values())).num_rows for v in task_dict.values()) == (33061 if row['row'] == 6 else 1000)
        backend = make_backend(cfg,'http://127.0.0.1:9/v1')
        backend.task_dict = task_dict
        payloads = []
        def capture(payload, url):
            payloads.append({k:v for k,v in payload.items() if k != 'messages'})
            assert payload['chat_template_kwargs'] == {'enable_thinking':False}
            expected_caps = {x['generation_kwargs']['max_new_tokens'] for x in selected}
            assert payload['max_tokens'] in expected_caps, payload['max_tokens']
            assert payload['temperature'] == (0.0 if row['row'] == 6 else 1.0)
            if row['row'] != 6:
                assert payload['top_p'] == 1.0
            assert payload['messages']
            return {'choices':[{'message':{'content':'A'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':1,'completion_tokens':1}}
        backend._make_request = capture
        results = backend.generate_until(requests)
        assert len(payloads) == len(requests) and all(r.text == 'A' for r in results)
        assert all(x['repeats'] == 1 for x in selected)
        report['jobs'].append({'row':row['row'],'model':model['architectures'][0],
            'request_count':len(payloads),'tasks':selected,'serialized_generation_parameters':payloads,
            'judge_required':False,'webui_defaults_match':True})
        print('Verified row',row['row'],flush=True)
    # Benchmark implementations and model request construction must remain the
    # current checkout's versions; the only backend delta is transport receipts/recovery.
    for folder in ['superchem_v2','SmolInstruct_v2']:
        for p in (REPO / 'lmms_eval/tasks' / folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.yaml'}:
                assert p.read_bytes() == (SOURCE / p.relative_to(REPO)).read_bytes(),str(p)
    from lmms_eval.tasks.SmolInstruct_v2.utils import smol_aggregate_results
    for task, values in [('forward_synthesis',['CCO','CC']),('property_prediction-bbbp',['Yes','No']),
                         ('molecule_captioning',['A primary alcohol.','A simple alkane.'])]:
        score = smol_aggregate_results([{'task':task,'pred':x,'gold':x} for x in values])
        assert math.isclose(score,1.0,abs_tol=1e-10),(task,score)
    report.update(passed=True,source_manifest_sha256=manifest['source_manifest_sha256'])
    (WORK / 'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PREFLIGHT PASSED',flush=True)


if __name__ == '__main__':
    main()
