from openai import OpenAI
import time
from typing import Dict, Any

from src.core.settings import settings
from src.llms.base import BaseLLM
from src.utils.logging_utils import setup_logger, log_rules_generation

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    VALID_MODELS = [
        'gpt-4o-mini',  # Cost-effective general use
        'gpt-4o',       # High-quality, general-purpose
        'o3',
        'o3-mini',       # Highest quality (expensive, no temperature control)
        'gpt-4-turbo-preview',
        'gpt-3.5-turbo',
        'gpt-4',
        'o1-mini'
    ]

    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        if model_name not in self.VALID_MODELS:
            valid_models_str = '\n- '.join([''] + self.VALID_MODELS)
            raise ValueError(f"Invalid OpenAI model name: {model_name}. Valid models are:{valid_models_str}")
        
        no_temperature_models = ['o3', 'o3-mini']  # <- Add o3 here
        if model_name in no_temperature_models and temperature != 1:
            print(f"⚠️ Warning: {model_name} does not support temperature settings. Using default (1.0).")
            self.temperature = 1.0
        
        # Initialize OpenAI client
        # self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY, organization=settings.OPENAI_ORG_ID)
        
        # Set up logger
        self.logger = setup_logger(f"openai_{model_name.replace('-', '_')}")

    def send_prompt(self, system_message: str, user_prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            system_message (str): High-level context or instructions for the LLM.
            user_prompt (str): The specific prompt or query to be processed.
            
        Returns:
            str: The LLM's response as a string.
        """
        try:
            no_system_role_models = ['o1-mini']
            no_temperature_role_models = ['o3-mini']
            
            if self.model_name in no_system_role_models:
                # Combine system and user messages for models that don't support system role
                combined_prompt = f"{system_message}\n\n{user_prompt}"
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": combined_prompt}]
                )
            else:
                # Use system role for models that support it
                if self.model_name in no_temperature_role_models:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature
                    )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error sending prompt: {e}")
            raise Exception(f"OpenAI API error: {str(e)}")

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
        Generate insights about a founder's success or failure using OpenAI's API.
        """
        status = "successful" if startup_success else "unsuccessful"
        outcome = "succeeded" if startup_success else "failed"

        # Format the lists into readable strings
        degrees_str = "\n- " + "\n- ".join(university_degrees) if university_degrees else "No university degrees listed"
        work_str = "\n- " + "\n- ".join(work_history) if work_history else "No work history listed"
        companies_str = "\n- " + "\n- ".join(previous_companies_founded) if previous_companies_founded else "No previous companies founded"

        prompt = f"""
        Founder Name: {founder_name}
        Location: {company_location}
        
        Education:
        {degrees_str}
        
        Work History:
        {work_str}
        
        Previous Companies Founded:
        {companies_str}
        
        Professional Background and Achievements:
        {professional_background}
        
        This startup was eventually {status}.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a VC analyst. Analyze the given data of successful or unsuccessful founders and identify common features or patterns. Provide a concise summary of these common characteristics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )

            reflection = response.choices[0].message.content.strip()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            return {
                "Founder ID": founder_id,
                "Reflection (Natural Language)": reflection,
                "Outcome": "Successful" if startup_success else "Unsuccessful",
                "Timestamp": timestamp,
                "Model": self.model_name,
                "Temperature": self.temperature,
                "Provider": "OpenAI"
            }

        except Exception as e:
            print(f"Error generating insight for founder {founder_id}: {e}")
            raise

    def test_question(self, prompt: str) -> str:
        """
        Test a question against founder data.
        
        Args:
            prompt: Prompt containing the question and founder data
            
        Returns:
            str: LLM response in JSON format
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a VC analyst evaluating founders. Answer questions about founders with 'Yes' or 'No' in the specified JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
