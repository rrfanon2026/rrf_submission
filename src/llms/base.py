from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLM(ABC):
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate_founder_insight(self, 
                               founder_id: str,
                               founder_name: str,
                               university_degrees: list,
                               work_history: list,
                               previous_companies_founded: list,
                               startup_success: bool,
                               professional_background: str,
                               company_location: str) -> Dict[str, Any]:
        """
        Generate insights about a founder's success or failure.
        
        Args:
            founder_id: Unique identifier for the founder
            founder_name: Name of the founder
            university_degrees: List of university degrees
            work_history: List of previous work experiences
            previous_companies_founded: List of companies previously founded
            startup_success: Boolean indicating if the startup was successful
            professional_background: String describing professional experience and achievements
            company_location: Location where the company was founded
            
        Returns:
            Dict containing the generated insight and metadata
        """
        pass

    @abstractmethod
    def test_question(self, prompt: str) -> str:
        """
        Test a question against founder data.
        
        Args:
            prompt: Prompt containing the question and founder data
            
        Returns:
            str: LLM response in JSON format
        """
        pass 