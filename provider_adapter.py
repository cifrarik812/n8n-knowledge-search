"""Provider adapter with TLS verification, cached OAuth and sanitized errors."""
import math
import threading
import time
import uuid
import requests


class ServiceError(ValueError):
    pass


class GigaChat:
    def __init__(self, settings):
        self.settings = settings
        self.embedding_model = settings.embedding_model
        self._token, self._expiry = None, 0
        self._lock = threading.RLock()

    def _call(self, url, **kwargs):
        try:
            response = requests.request('POST', url, verify=self.settings.ca_bundle or True,
                timeout=self.settings.timeout, allow_redirects=False, **kwargs)
            if response.status_code != 200:
                raise ServiceError(f'GigaChat: HTTP {response.status_code}. Проверь доступ и лимит API.')
            return response.json()
        except requests.RequestException:
            raise ServiceError('GigaChat недоступен: проверь сеть и сертификаты TLS.') from None
        except (ValueError, TypeError) as exc:
            if isinstance(exc, ServiceError):
                raise
            raise ServiceError('GigaChat вернул некорректный ответ.') from None

    def _headers(self):
        if not self.settings.key:
            raise ServiceError('Не настроен GigaChat: укажи RAG_PROVIDER_ENV или .env.')
        if not self._token or time.time() >= self._expiry:
            payload = self._call('https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
                headers={'Authorization':'Basic '+self.settings.key.removeprefix('Basic '), 'RqUID':str(uuid.uuid4())},
                data={'scope':self.settings.scope})
            try:
                token = payload['access_token']
                expiry = float(payload['expires_at'])/1000-30
                if not isinstance(token,str) or not token:
                    raise ValueError()
                self._token, self._expiry = token, expiry
            except (KeyError,ValueError,TypeError):
                raise ServiceError('Некорректный ответ авторизации GigaChat.') from None
        return {'Authorization':'Bearer '+self._token}

    def chat(self, messages):
        with self._lock:
            payload = self._call(self.settings.api_base+'/v1/chat/completions', headers=self._headers(),
                json={'model':self.settings.model,'messages':messages, 'temperature':0.1,'max_tokens':1800,'stream':False})
            try:
                content = payload['choices'][0]['message']['content']
                if not isinstance(content,str) or not content.strip():
                    raise ValueError()
                return content.strip()
            except (KeyError,IndexError,TypeError,ValueError):
                raise ServiceError('Пустой или некорректный ответ модели.') from None

    def embed(self, texts):
        if not texts:
            return []
        with self._lock:
            payload = self._call(self.settings.api_base+'/v1/embeddings', headers=self._headers(),
                json={'model':self.embedding_model,'input':texts})
            try:
                data = sorted(payload['data'], key=lambda row:row['index'])
                if [row['index'] for row in data] != list(range(len(texts))):
                    raise ValueError()
                vectors = [row['embedding'] for row in data]
                if not vectors[0] or any(len(v)!=len(vectors[0]) or
                    not all(isinstance(x,(float,int)) and math.isfinite(x) for x in v) or
                    not any(x!=0 for x in v) for v in vectors):
                    raise ValueError()
                return vectors
            except (KeyError,IndexError,TypeError,ValueError):
                raise ServiceError('Некорректные эмбеддинги GigaChat.') from None

