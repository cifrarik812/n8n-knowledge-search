"""Real-vector integration checks; no LLM or provider key required."""
import tempfile
import unittest
from pathlib import Path
from rag_service import RagStore

DATA = Path(__file__).resolve().parent / 'data'

class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp(prefix='per05-store-test-'))
        cls.store = RagStore(cls.temp)
        cls.documents = [{'filename':p.name,'text':p.read_text(encoding='utf-8-sig')}
                         for p in sorted(DATA.glob('*.txt'))]

    def test_ingest_persist_search_and_reject_invalid(self):
        self.assertEqual(self.store.ingest(self.documents)['count'], 6)
        self.assertEqual(self.store.ingest(self.documents)['count'], 6)
        reopened = RagStore(self.temp, embedder=self.store.embedder)
        self.assertEqual(reopened.count(), 6)
        diagnosis = reopened.search('Сколько стоит диагностика при отказе от ремонта?',generate=False)
        self.assertEqual(diagnosis['matches'][0]['source'], '01-diagnostics-and-intake.txt')
        self.assertIn('700',diagnosis['matches'][0]['text'])
        warranty = reopened.search('Какова гарантия и входит ли залитие?',generate=False)
        self.assertEqual(warranty['matches'][0]['source'],'02-warranty-and-pickup.txt')
        self.assertIn('90 календарных',warranty['matches'][0]['text'])
        with self.assertRaises(ValueError):
            reopened.ingest([{'filename':'empty.txt','text':''}])
        self.assertEqual(reopened.count(),6)
        with self.assertRaises(ValueError):
            reopened.search('',generate=False)

if __name__ == '__main__': unittest.main(verbosity=2)

