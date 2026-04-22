from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract Base Class for Multimodal LLM Providers."""

    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a text response based on system and user prompts.
        
        Args:
            system_prompt (str): The system prompt detailing behavior and instructions.
            user_prompt (str): The actual user query or structured input.
            
        Returns:
            str: Generated text.
        """
        pass

    @abstractmethod
    def generate_vision(self, prompt: str, image_path: str) -> str:
        """
        Generate a text response analyzing an image.
        
        Args:
            prompt (str): The prompt detailing how to analyze the image.
            image_path (str): The local path to the image file.
            
        Returns:
            str: Generated text explanation.
        """
        pass
