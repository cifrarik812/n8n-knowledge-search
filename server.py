"""Loopback-only service called by the two new n8n workflows."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from rag_service import RagStore

store=None
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,status,payload):
        body=json.dumps(payload,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def permitted(self):
        return self.headers.get('Host') in ('127.0.0.1:8796','localhost:8796','host.docker.internal:8796') and not self.headers.get('Origin')
    def do_GET(self):
        if not self.permitted(): self.reply(403,{'error':'Недопустимый источник запроса.'}); return
        if self.path!='/health': self.reply(404,{'error':'Не найдено.'}); return
        self.reply(200,{'ok':True,'collection':'per05_service','count':store.count(),'interface':'n8n local adapter'})
    def do_POST(self):
        if not self.permitted(): self.reply(403,{'error':'Недопустимый источник запроса.'}); return
        try:
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                raise ValueError('Требуется JSON.')
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=100000: raise ValueError('Недопустимый размер запроса.')
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict): raise ValueError('Ожидается объект JSON.')
            if self.path=='/ingest': result=store.ingest(body.get('documents'))
            elif self.path=='/search': result=store.search(body.get('query'),body.get('top_k',3),body.get('generate',True))
            else: self.reply(404,{'error':'Не найдено.'}); return
            self.reply(200,result)
        except (ValueError,TypeError,KeyError): self.reply(400,{'error':'Не удалось обработать данные или ответ провайдера. Проверь формат и доступ к модели.'})
        except Exception: self.reply(503,{'error':'Локальный RAG или модель временно недоступны.'})

if __name__=='__main__':
    store=RagStore()
    print('PEr05 RAG ready: 127.0.0.1:8796; collection=per05_service; count='+str(store.count()),flush=True)
    ThreadingHTTPServer(('127.0.0.1',8796),Handler).serve_forever()

