import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
COLLECTION = 'per05_service'
ENV_FILE = os.environ.get('RAG_PROVIDER_ENV', str(ROOT / '.env'))
os.environ.setdefault('FASTEMBED_CACHE_PATH',str(ROOT / 'model-cache'))
os.environ.setdefault('ANONYMIZED_TELEMETRY','False')



