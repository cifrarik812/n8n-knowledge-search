import json,os
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
url=os.environ.get('PER05_UPLOAD_URL','http://localhost:5678/webhook/per05-chroma-upload')
documents=[{'filename':p.name,'text':p.read_text(encoding='utf-8-sig')} for p in sorted((ROOT/'data').glob('*.txt'))]
response=requests.post(url,json={'documents':documents},timeout=120)
response.raise_for_status()
result=response.json()
assert result['ok'] and result['files']==3 and result['count']==6 and result['chunks_uploaded']==6,result
(ROOT/'upload-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
