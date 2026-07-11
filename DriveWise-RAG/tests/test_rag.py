import unittest
import os
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from retrieval.retriever import DriveWiseRetriever
from app import app, ChatRequest, ChatMessage
from fastapi.testclient import TestClient

class TestDriveWiseRAG(unittest.TestCase):
    
    def setUp(self):
        self.retriever = DriveWiseRetriever()
        self.client = TestClient(app)

    def test_local_target_extraction(self):
        # 1. Test explicit brand/model mention in current query
        targets = self.retriever.extract_targets_locally("Does the Tata Nexon have a sunroof?")
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], ("tata", "nexon"))
        
        # 2. Test multiple targets for comparison query
        targets = self.retriever.extract_targets_locally("Compare the Hyundai Creta and Maruti Suzuki Swift.")
        self.assertEqual(len(targets), 2)
        self.assertIn(("hyundai", "creta"), targets)
        self.assertIn(("maruti-suzuki", "swift"), targets)
        
        # 3. Test resolution using history
        history = [{"role": "user", "content": "What is the boot space of the Tata Nexon?"}]
        targets = self.retriever.extract_targets_locally("What about its mileage?", history)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], ("tata", "nexon"))

    def test_local_keyword_extraction(self):
        keywords = self.retriever.extract_keywords_locally("Does the Creta have a smart panoramic sunroof?")
        # Should extract key nouns like 'panoramic', 'sunroof', excluding stopwords like 'does', 'the', 'have'
        self.assertIn("sunroof", keywords)
        self.assertIn("panoramic", keywords)
        self.assertNotIn("does", keywords)

    def test_mock_chat_routing(self):
        os.environ["MOCK_LLM"] = "true"
        
        # 1. Test sunroof mock response
        response = self.client.post("/api/chat", json={
            "query": "Does the Hyundai Creta have a sunroof?",
            "brand": "hyundai",
            "model": "creta"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["confidence"], "high")
        self.assertIn("sunroof", data["answer"].lower())
        self.assertEqual(data["citations"][0]["source_file"], "hyundai_creta_2026.pdf")
        
        # 2. Test boot space variant disambiguation mock response
        response = self.client.post("/api/chat", json={
            "query": "What is the boot space of the Tata Nexon?",
            "brand": "tata",
            "model": "nexon"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["variant_disambiguation"]), 2)
        self.assertEqual(data["variant_disambiguation"][0]["value"], "382 litres")
        self.assertEqual(data["variant_disambiguation"][1]["value"], "321 litres")
        
        # 3. Test unmatched query mock fallback
        response = self.client.post("/api/chat", json={
            "query": "What is the tyre size of the Alcazar?",
            "brand": "hyundai",
            "model": "alcazar"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["confidence"], "not_found")
        self.assertIn("no matching demo scenario was found", data["answer"].lower())

    def test_catalog_query(self):
        # Test catalog query detection and response
        response = self.client.post("/api/chat", json={
            "query": "What all cars are in your database?"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["confidence"], "high")
        self.assertIn("Our database contains the following brochures", data["answer"])
        self.assertIn("Hyundai", data["answer"])
        self.assertIn("Tata", data["answer"])
        self.assertIn("Maruti Suzuki", data["answer"])
        self.assertEqual(len(data["citations"]), 0)

if __name__ == "__main__":
    unittest.main()
