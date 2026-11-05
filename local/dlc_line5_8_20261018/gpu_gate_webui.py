"""Validate actual chat requests using the same native task settings as evaluation."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def leaves(tree):
    for name, obj in tree.items():
        if isinstance(obj, dict):
            yield from leaves(obj)
        else:
            yield name, obj[-1] if isinstance(obj, tuple) else obj


def prepare_requests(config):
    from lmms_eval.tasks import TaskManager, get_task_dict
    manager = TaskManager(verbosity='ERROR', model_name='vllm_backend')
    tree = get_task_dict(config['eval']['tasks'].split(','), task_manager=manager, task_type='chat')
    requests, selected, task_dict = [], [], {}
    for name, obj in leaves(tree):
        split = obj.config.test_split if obj.has_test_docs() else obj.config.validation_split
        docs = obj.test_docs() if obj.has_test_docs() else obj.validation_docs()
        task_dict[name] = {split: docs}
        indices = [0,34,313,390] if name.startswith('superchem_') else [0]
        for index in indices:
            doc = docs[index]
            assert callable(obj.doc_to_messages)
            messages = obj.doc_to_messages(doc)
            assert messages and obj.doc_to_text(doc)
            requests.append(SimpleNamespace(args=(obj.doc_to_text(doc), obj.doc_to_messages,
                dict(obj.config.generation_kwargs), index, name, split)))
            selected.append({'task':name,'doc_id':index,'images':len(obj.doc_to_visual(doc) or []),
                             'generation_kwargs':dict(obj.config.generation_kwargs),
                             'samples':len(docs),'repeats':obj.config.repeats})
    return requests, selected, task_dict


def make_backend(config, urls):
    from lmms_eval.models.chat.vllm_backend import VLLMBackend
    m, e = config['model'], config['eval']
    return VLLMBackend(base_url=urls,model=m['served_model_name'],timeout=e['vllm_request_timeout_seconds'],
        num_concurrent=e['concurrency'],adaptive_max_concurrency=e['concurrency'],max_new_tokens=32768,
        max_pixels=4014080,min_pixels=78400,is_qwen3_vl=m['is_qwen3_vl'],enable_thinking=m['enable_thinking'],
        stop_token_ids=m['stop_token_ids'],max_retries=1,shuffle_requests=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--urls', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    logdir = Path(config['log']['dir'])
    os.environ['LMMS_RESPONSE_JOURNAL'] = str(logdir / 'gpu_gate_responses')
    requests, selected, task_dict = prepare_requests(config)
    backend = make_backend(config, args.urls)
    backend.task_dict = task_dict
    print('[GPU_GATE] Starting '+str(len(requests))+' real benchmark requests.',flush=True)
    results = backend.generate_until(requests)
    assert len(results) == len(requests)
    for item, result in zip(selected, results):
        assert result.text and not result.text.startswith(('[LMMS_','[PREPROC_FAILED]')), item
        item.update(input_tokens=result.token_counts.input_tokens, output_tokens=result.token_counts.output_tokens)
    receipts = [json.loads(p.read_text()) for p in (logdir / 'gpu_gate_responses').rglob('*.json')]
    assert len(receipts) == len(requests)
    report = {'passed':True,'samples':selected,'adaptations':[
        {'identity':r['identity'],'adaptation':r['response']['_lmms_context_recovery']}
        for r in receipts if '_lmms_context_recovery' in r['response']]}
    (logdir / 'gpu_gate_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('[GPU_GATE] PASSED; starting full evaluation.',flush=True)


if __name__ == '__main__':
    main()
