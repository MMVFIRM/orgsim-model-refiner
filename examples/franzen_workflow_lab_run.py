"""Offline-only synthetic ERP workflow exercise; no real Dolibarr connectivity."""
from pathlib import Path
import json
from orgsim.engine import SimulationEngine, SimError
from orgsim.spec import load_spec

spec = load_spec(Path(__file__).with_name('franzen_workflow_lab.yaml'))

def seed(eng, *, stock: int):
    for tool, data in [
      ('erp.customers.create', {'id':'1','name':'Fictional QA Customer'}),
      ('erp.products.create', {'id':'2','ref':'QA-SKU-001','stock':stock}),
      ('erp.orders.create',{'id':'3','customer_id':'1','product_id':'2','qty':4,'status':'draft'})
    ]:
        eng.call(tool,data,role='test_agent',actor='scenario_setup')

for stock in [10,2]:
    eng=SimulationEngine(spec,allow_draft=True)
    try:
        seed(eng,stock=stock)
        try:
            outcome=eng.call('erp.workflow.validate_order',{'id':'3'},role='test_agent')
            result={'ok':outcome['ok'],'evidence':outcome['evidence']}
        except SimError as exc:
            result={'ok':False,'error':exc.code}
        report={'stock_before':stock,'workflow':result,
                'order_status_after':eng.get('erp','orders','3')['status'],
                'stock_after':eng.get('erp','products','2')['stock'],
                'failed_calls_logged':sum(not x['ok'] for x in eng.tool_trace())}
        print(json.dumps(report,sort_keys=True))
    finally:
        eng.close()
