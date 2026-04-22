import requests
import base64
from .base import LLMProvider

class LocalLLMProvider(LLMProvider):
    """
    LLM Provider for Local models using Ollama.
    Expects Ollama running locally at http://localhost:11434.
    Requires a model (e.g. 'mistral' or 'llama2') for text.
    For vision, a multimodal model like 'llava' is recommended.
    """

    def __init__(self, host: str = "http://localhost:11434", text_model: str = "mistral", vision_model: str = "llava"):
        self.host = host
        self.text_model = text_model
        self.vision_model = vision_model
        
        # Test connection briefly
        try:
            requests.get(f"{self.host}/api/tags", timeout=2)
            print(f'✅ LocalLLMProvider Ready! Host: {self.host} | Text: {self.text_model} | Vision: {self.vision_model}')
        except requests.exceptions.RequestException:
            print(f'⚠️  Warning: Could not connect to locally running Ollama at {self.host}')

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.host}/api/generate"
        
        # Combine system and user prompt for Ollama generate endpoint
        full_prompt = f"System: {system_prompt}\nUser: {user_prompt}"
        
        payload = {
            "model": self.text_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1500
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f'  ⚠️  Local text generation failed: {e}')
            raise RuntimeError(f"Ollama text exception: {e}")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_vision(self, prompt: str, image_path: str) -> str:
        url = f"{self.host}/api/generate"
        base64_image = self._encode_image(image_path)
        
        payload = {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1500
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f'  ⚠️  Local vision generation failed: {e}')
            raise RuntimeError(f"Ollama vision exception: {e}")
