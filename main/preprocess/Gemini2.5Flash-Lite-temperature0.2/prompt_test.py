import unittest
import requests
import json

GEMINI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"
MODEL_NAME = "gemini-2.5-flash-lite"   # 若失败可先改成 gemini-2.0-flash
API_URL = f"https://api.zhizengzeng.com/google/v1beta/models/{MODEL_NAME}:generateContent"
HEADERS = {"Content-Type": "application/json"}

SYSTEM_INSTRUCTION = (
    "You are a precise clinical biomedical writing assistant. "
    "Generate formal, clinically grounded, machine-readable medical text "
    "as one continuous paragraph."
)

DRUG_PROMPT_TMPL = (
    "You are a distinguished clinical pharmacologist and expert in cheminformatics. "
    "Generate a comprehensive machine-readable description for the medication with "
    "ATC code {atc3_code} (representative NDC: {ndc_code}). "
    "Write one continuous paragraph covering therapeutic context, related diagnoses, "
    "possible procedures, mechanism of action, co-prescriptions, and important DDIs when appropriate."
)

def call_gemini(prompt: str) -> str:
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 800,
            "topP": 0.8,
            "topK": 10
        }
    }

    resp = requests.post(
        API_URL,
        headers=HEADERS,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=90
    )

    print("HTTP status:", resp.status_code)
    print("Response text:", resp.text)

    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise AssertionError(f"API error: {json.dumps(data, ensure_ascii=False)}")

    candidates = data.get("candidates", [])
    if not candidates:
        raise AssertionError(f"No candidates returned: {json.dumps(data, ensure_ascii=False)}")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise AssertionError(f"No content parts returned: {json.dumps(data, ensure_ascii=False)}")

    result = "".join(part.get("text", "") for part in parts).strip()
    if not result:
        raise AssertionError(f"Empty text returned: {json.dumps(data, ensure_ascii=False)}")

    return result

class TestGeminiDrugA02A(unittest.TestCase):
    def test_a02a_description(self):
        prompt = DRUG_PROMPT_TMPL.format(atc3_code="A02A", ndc_code="TEST-NDC")
        result = call_gemini(prompt)
        self.assertTrue(result)
        print("\nGenerated text:\n", result)

if __name__ == "__main__":
    unittest.main()