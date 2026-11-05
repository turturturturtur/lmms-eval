import json
from collections import Counter
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from lmms_eval.tui import server
import asyncio

url='http://127.0.0.1:8000'
defaults=asyncio.run(server.get_defaults()).model_dump()
defaults.update(user='reviewer', model='/tmp/fake-model', tasks=['ai2d'], env_vars='', judge_api_key='', api_key='')
counts=Counter()
state={'status':'Running', 'retry_failure':True, 'retry_ready':False}
errors=[]

def route_api(route):
    req=route.request
    path=urlparse(req.url).path
    if not path.startswith(('/auth/','/eval/','/dlc/')) and path not in ('/health','/defaults','/tasks','/storage'):
        route.continue_(); return
    counts[path]+=1
    payload={}
    status=200
    if path=='/auth/me': payload={'username':'reviewer','display_name':'reviewer','role':'user','access_key_id':'fake-ak','expires_at':9999999999}
    elif path=='/health': payload={'version':'review','git':{},'system':{}}
    elif path=='/defaults': payload=defaults
    elif path=='/tasks': payload=[{'id':'ai2d','name':'AI2D','group':False,'requires_llm_judge':False}]
    elif path=='/storage': payload={'quota':'fake','paths':[]}
    elif path=='/dlc/pool-usage': payload={'detail':'offline review'}; status=503
    elif path=='/eval/preview': payload={'command':'safe preview'}
    elif path=='/eval/start': payload={'job_id':'fake-session','command':'safe preview'}
    elif path=='/eval/fake-session/stop': payload={'detail':'simulated stop failure'}; status=500
    elif path=='/dlc/jobs/dlc-fake/retry-judge':
        if state['retry_failure']:
            payload={'detail':'Source sample coverage is incomplete'}; status=400
        else:
            payload={'job_id':'retry-workflow','job_name':'eval_judge_retry_fixture'}
    elif path=='/eval/retry-workflow': payload={'status':'running','finished':False,'dlc_jobs':{'eval':'dlcretried'} if state['retry_ready'] else {}}
    elif path=='/dlc/jobs':
        job=server.DlcJobSummary(job_id='dlc-fake',name='eval_review',status=state['status'], job_stage='eval',lmms_tasks=[],requires_llm_judge=False,create_time='2026-10-01T00:00:00Z').model_dump()
        payload={'jobs':[job],'total':1,'start_time':'2026-09-05T00:00:00Z','end_time':'2026-10-05T00:00:00Z','fetched_at':'now','source':'offline review'}
    elif path=='/dlc/jobs/dlc-fake': payload={'job':{'Status':state['status']},'result_root':None,'result_status':state['status'],'requires_llm_judge':True,'llm_judge_tasks':['mmmu_val_qwen3_official']}
    elif path=='/dlc/jobs/dlc-fake/metrics':
        rows=[] if state['status']=='Running' else [server.DlcMetricRow(metric_id='1',display_name='AI2D',lmms_tasks='ai2d',status='success',value=0.95,value_text='0.95',metric_name='accuracy',sample_jsonls=['fixture_samples_ai2d.jsonl']).model_dump()]
        payload={'job_id':'dlc-fake','result_root':None,'metrics':rows,'summary_files':[]}
    elif path=='/dlc/jobs/dlc-fake/metrics/1/samples':
        payload={'job_id':'dlc-fake','metric_id':'1','columns':['doc_id'],
                 'rows':[{'doc_id':0,'_media':[{'url':'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Crect width="100" height="100" fill="blue"/%3E%3C/svg%3E','label':'fixture image','source':'fixture','media_type':'image/svg+xml'}]}],
                 'total':1,'offset':0,'limit':50,'sample_files':['fixture_samples_ai2d.jsonl'],
                 'answer_stats':{'is_multiple_choice':False,'correct_answers':[],'target_answers':[],'total':1,'filtered_total':1,'wrong_total':1}}
    else:
        status=404; payload={'detail':'offline route'}
    route.fulfill(status=status, content_type='application/json',body=json.dumps(payload))

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/root/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1800,'height':1000})
    page.route('**/*',route_api)
    page.add_init_script('window.reviewStreams=[]; window.EventSource=class {constructor(url){this.url=url;window.reviewStreams.push(this);} close(){this.closed=true;}};')
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(url)
    page.get_by_role('button',name='Start',exact=True).wait_for()
    page.wait_for_function("document.body.innerText.includes('No local metrics found.')")
    page.get_by_role('button',name='View Logs',exact=True).click()
    before=dict(counts)
    state['status']='Succeeded'
    page.get_by_role('button',name='Sync DLC',exact=True).click()
    page.wait_for_function("document.body.innerText.includes('Succeeded')")
    page.wait_for_timeout(250)
    print('SYNC_COUNTERS',{'jobs_before':before.get('/dlc/jobs'),'jobs_after':counts['/dlc/jobs'],'metrics_before':before.get('/dlc/jobs/dlc-fake/metrics'),'metrics_after':counts['/dlc/jobs/dlc-fake/metrics']})
    assert counts['/dlc/jobs/dlc-fake/metrics'] >= 2
    assert page.get_by_text('No local metrics found.',exact=True).count() == 0
    page.locator('tr[title="Open sample rows"]').first.click()
    page.locator('td img').first.click()
    close = page.get_by_role('button',name='Close',exact=True)
    close.wait_for()
    box = close.bounding_box()
    assert box and 0 <= box['x'] <= 1800 - box['width'] and 0 <= box['y'] <= 1000 - box['height'], box
    close.click()
    print('IMAGE_MODAL_IN_VIEWPORT', box)
    page.get_by_role('button',name='Back to job',exact=True).click()
    page.get_by_role('button',name='Retry judge (8 GPUs)',exact=True).click()
    page.get_by_text('Source sample coverage is incomplete',exact=True).wait_for()
    state['retry_failure']=False
    page.get_by_role('button',name='Retry judge (8 GPUs)',exact=True).click()
    page.get_by_text('Submitting eval_judge_retry_fixture on 8 GPUs. Completed inference will be reused.',exact=True).wait_for()
    page.reload()
    page.get_by_role('button',name='View Logs',exact=True).click()
    state['retry_ready']=True
    page.get_by_text('Judge retry submitted: dlcretried. Results are in eval_judge_retry_fixture.',exact=True).wait_for()
    assert counts['/dlc/jobs/dlc-fake/retry-judge'] == 2
    print('JUDGE_RETRY_VALIDATION_RELOAD_AND_RECEIPT_PASS')
    page.get_by_role('button',name='Evaluate',exact=True).click()
    page.get_by_role('button',name='Start',exact=True).click()
    page.wait_for_function('window.reviewStreams.length===1')
    page.evaluate("window.reviewStreams[0].onmessage({data:JSON.stringify({type:'done',exit_code:0})})")
    page.get_by_text('Evaluation completed (exit: 0)',exact=True).wait_for()
    print('DONE_LABEL',page.get_by_text('Evaluation completed (exit: 0)',exact=True).inner_text(),'stop_disabled',page.get_by_role('button',name='Stop',exact=True).is_disabled())
    page.get_by_role('button',name='Start',exact=True).click()
    page.wait_for_function('window.reviewStreams.length===2')
    page.get_by_role('button',name='Stop',exact=True).click()
    page.wait_for_function("document.body.innerText.includes('simulated stop failure')")
    assert page.get_by_role('button',name='Stop',exact=True).is_enabled()
    print('STOP_FAILURE_DISPLAYED',{'start_disabled':page.get_by_role('button',name='Running...',exact=True).is_disabled(),'stop_disabled':page.get_by_role('button',name='Stop',exact=True).is_disabled(),'error_visible':'simulated stop failure' in page.locator('body').inner_text()})
    page.get_by_role('button',name='Task Builder',exact=True).click()
    py=page.get_by_label('Python editor').input_value()
    compile(py,'generated_utils.py','exec')
    ns={}; exec(py,ns)
    print('TEMPLATE_REGEX_LINE',next(line for line in py.splitlines() if 're.sub' in line))
    print('TEMPLATE_SCORE', ns['custom_vqa_task_process_results']({'answer':'New York'},['New  York']))
    assert ns['custom_vqa_task_process_results']({'answer':'New York'},['New\tYork'])['custom_vqa_task_exact_match'] == 1.0
    assert ns['custom_vqa_task_process_results']({'answer':'New York'},['New  York'])['custom_vqa_task_exact_match'] == 1.0
    assert not errors, errors
    print('BROWSER_REGRESSIONS_PASS')
    browser.close()
