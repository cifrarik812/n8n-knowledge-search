import json,os
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
url=os.environ.get('PER05_SEARCH_URL','http://localhost:5678/webhook/per05-chroma-search')
cases=[
 ('diagnosis','Сколько стоит диагностика ноутбука, если я откажусь от ремонта?','01-diagnostics-and-intake.txt','700'),
 ('warranty','Как долго действует гарантия и покрывает ли она залитие ноутбука?','02-warranty-and-pickup.txt','90'),
 ('paraphrase','Сколько заплачу за проверку неисправности, если решу не чинить ноутбук?','01-diagnostics-and-intake.txt','700'),
 ('faq','Что нужно сообщить, чтобы узнать готовность ремонта?','03-faq.txt','квитанц'),
 ('missing','Есть ли выезд мастера на дом и сколько он стоит?',None,None),
]
results=[]
for name,query,source,fact in cases:
    response=requests.post(url,json={'query':query,'top_k':3,'generate':True},timeout=120)
    response.raise_for_status()
    result=response.json()
    assert result['ok'] and result['count']==6 and result['model']=='GigaChat-2-Pro',result
    if source:
        assert result['matches'][0]['source']==source,(name,result['matches'])
        assert fact in result['answer'].lower(),(name,result['answer'])
    else:
        answer=result['answer'].lower()
        assert (('нет' in answer and any(w in answer for w in ['информац','сведени','документ']))
                or any(w in answer for w in ['отсутствует','не содерж','не указ','не наш'])),answer
        assert '700' not in answer,answer
    results.append({'case':name,**result})
    (ROOT/'search-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name+': PASS; first_source='+result['matches'][0]['source']+'; answer='+result['answer'],flush=True)
print('5/5 n8n webhook + ChromaDB + GigaChat checks passed.')
