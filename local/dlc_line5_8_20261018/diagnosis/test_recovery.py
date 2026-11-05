import json
import os
import threading
import time
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from lmms_eval.models.chat.vllm_backend import VLLMBackend
from lmms_eval.models.chat import vllm_backend as backend_module


@contextmanager
def api():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(payload)
            if payload.get('slow'):
                time.sleep(0.12)
            cap = payload.get('mm_processor_kwargs', {}).get('max_dynamic_patch', 12)
            error = payload.get('oversize') and cap > 10
            status = 400 if error or payload.get('invalid') else 200
            result = ({'error': {'message': "Input length (43917) exceeds model's maximum context length (40960)."}}
                if error else {'error': {'message': 'Unknown model'}} if payload.get('invalid')
                else {'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 4}})
            self.send_response(status); self.send_header('Content-Type','application/json'); self.end_headers()
            try:
                self.wfile.write(json.dumps(result).encode())
            except BrokenPipeError:
                pass
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1', seen
    server.shutdown(); server.server_close(); thread.join()


def backend(timeout=1):
    b = VLLMBackend.__new__(VLLMBackend)
    b.session = requests.Session(); b.headers = {}; b.timeout = timeout
    return b


def timeout_reproduction_and_larger_budget(api):
    url, seen = api
    with unittest.TestCase().assertRaises(requests.exceptions.ReadTimeout):
        backend(0.03)._make_request({'slow':True}, url)
    assert backend(1)._make_request({'slow':True}, url)['choices'][0]['message']['content']=='answer'


def overlong_multimodal_request_retains_all_inputs(api):
    url, seen = api
    payload={'oversize':True,'model':'internvl','messages':[{'role':'user','content':[
        {'type':'image_url','image_url':{'url':'original-image-1'}},
        {'type':'image_url','image_url':{'url':'original-image-2'}},
        {'type':'text','text':'Complete question with every option'}]}],
        'max_tokens':32768,'chat_template_kwargs':{'enable_thinking':False}}
    result = backend()._make_request(payload,url)
    assert result['choices'][0]['message']['content']=='answer'
    assert [p.get('mm_processor_kwargs',{}).get('max_dynamic_patch',12) for p in seen]==[12,11,10]
    for submitted in seen:
        assert submitted['messages']==payload['messages']
        assert submitted['max_tokens']==32768
        assert submitted['chat_template_kwargs']=={'enable_thinking':False}


def unrelated_400_is_not_changed(api):
    url,seen=api
    with unittest.TestCase().assertRaises(requests.HTTPError):
        backend()._make_request({'invalid':True},url)
    assert len(seen)==1


class RecoveryTests(unittest.TestCase):
    def test_timeout(self):
        with api() as fixture:
            timeout_reproduction_and_larger_budget(fixture)
    def test_context(self):
        with api() as fixture, patch.dict(os.environ, {'LMMS_INTERNVL_CONTEXT_RECOVERY':'1'}):
            overlong_multimodal_request_retains_all_inputs(fixture)
    def test_invalid(self):
        with api() as fixture, patch.dict(os.environ, {'LMMS_INTERNVL_CONTEXT_RECOVERY':'1'}):
            unrelated_400_is_not_changed(fixture)
    def test_journal_keeps_repeats_separate_and_resumes(self):
        with api() as (url, seen), TemporaryDirectory() as directory, patch.dict(os.environ, {'LMMS_RESPONSE_JOURNAL':directory}), patch.object(
            backend_module,'Accelerator',return_value=SimpleNamespace(num_processes=1,local_process_index=0,device='cpu')):
            b=VLLMBackend(base_url=url,model='model',num_concurrent=2,adaptive_max_concurrency=2,
                prefix_aware_queue=False,max_retries=1,enable_thinking=False)
            b._rank=1
            b.task_dict={'task':{'test':[{'question':'Same question for eight independent trials'}]}}
            request=SimpleNamespace(args=('context',lambda d:[{'role':'user','content':[{'type':'text','text':d['question']}]}],
                {'max_new_tokens':16},0,'task','test'))
            assert len(b.generate_until([request]*8))==8
            assert len(seen)==8
            receipts=[json.loads(p.read_text()) for p in Path(directory).rglob('*.json')]
            assert len(receipts)==8
            assert {r['identity']['request_index'] for r in receipts}==set(range(8))
            assert all(r['response']['choices'][0]['finish_reason']=='stop' for r in receipts)
            assert len(b.generate_until([request]*8))==8
            assert len(seen)==8, 'Resume must use each saved trial without resubmitting'


if __name__ == '__main__':
    unittest.main(verbosity=2)
