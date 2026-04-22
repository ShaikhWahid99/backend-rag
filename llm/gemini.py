import time
from google import genai
from google.genai import types
from PIL import Image as PILImage
from .base import LLMProvider

class GeminiProvider(LLMProvider):
    """
    LLM Provider for Google's Gemini models.
    Supports free tier quota management (throttling).
    """

    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-flash'):
        self.api_key = api_key
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not defined.")
            
        self.client = genai.Client(api_key=self.api_key)
        self._last_call = 0
        self._call_count = 0
        
        print(f'✅ GeminiProvider Ready! Model: {self.model_name}')

    def _throttle(self):
        """Wait so we never exceed 10 req/min (free tier limit)."""
        elapsed = time.time() - self._last_call
        if elapsed < 6:
            wait = 6 - elapsed
            print(f'  ⏱️  Throttling {wait:.1f}s...')
            time.sleep(wait)
        self._last_call = time.time()
        self._call_count += 1

    def generate_text(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        for attempt in range(1, retries + 1):
            try:
                self._throttle()
                r = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=1500,
                        temperature=0.1
                    )
                )
                if r.text and len(r.text.strip()) > 20:
                    print(f'  ✅ {self.model_name} text (call #{self._call_count})')
                    return r.text.strip()
                print(f'  ⚠️  Empty response attempt {attempt}')
            except Exception as e:
                err = str(e)
                if '429' in err or 'quota' in err.lower() or 'RESOURCE_EXHAUSTED' in err:
                    print(f'  ⏳ Rate limit hit! Aborting Gemini to trigger instant fallback...')
                    raise RuntimeError("Gemini Rate Limited! Redirecting...")
                else:
                    print(f'  ⚠️  Attempt {attempt}: {e}')
                    time.sleep(5)
                    
        raise RuntimeError('❌ API query failed after retries.')

    def generate_vision(self, prompt: str, image_path: str, retries: int = 3) -> str:
        pil_img = PILImage.open(image_path)
        for attempt in range(1, retries + 1):
            try:
                self._throttle()
                r = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, pil_img],
                    config=types.GenerateContentConfig(
                        max_output_tokens=1500,
                        temperature=0.1
                    )
                )
                if r.text and len(r.text.strip()) > 20:
                    print(f'  ✅ {self.model_name} vision (call #{self._call_count})')
                    return r.text.strip()
                print(f'  ⚠️  Empty vision response attempt {attempt}')
            except Exception as e:
                err = str(e)
                if '429' in err or 'quota' in err.lower() or 'RESOURCE_EXHAUSTED' in err:
                    print(f'  ⏳ Rate limit hit! Aborting Gemini vision to trigger instant fallback...')
                    raise RuntimeError("Gemini Vision Rate Limited! Redirecting...")
                else:
                    print(f'  ⚠️  Attempt {attempt}: {e}')
                    time.sleep(5)
                    
        raise RuntimeError('❌ Vision model API query failed after retries.')
