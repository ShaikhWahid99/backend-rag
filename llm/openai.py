import base64
from openai import OpenAI
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    """
    LLM Provider for OpenAI's models.
    Default models are gpt-4o as they support multimodality efficiently.
    """

    def __init__(self, api_key: str, model_name: str = 'gpt-4o'):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not defined.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f'✅ OpenAIProvider Ready! Model: {self.model_name}')

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            text = response.choices[0].message.content
            if text:
                return text.strip()
            raise RuntimeError("Empty response from OpenAI.")
        except Exception as e:
            print(f'  ⚠️  OpenAI text generation failed: {e}')
            raise

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_vision(self, prompt: str, image_path: str) -> str:
        try:
            base64_image = self._encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            text = response.choices[0].message.content
            if text:
                return text.strip()
            raise RuntimeError("Empty response from OpenAI Vision.")
        except Exception as e:
            print(f'  ⚠️  OpenAI vision generation failed: {e}')
            raise
