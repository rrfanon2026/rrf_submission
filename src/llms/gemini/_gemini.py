import google.generativeai as genai
import time
from typing import Dict, Any, List
from ..base import BaseLLM
from src.utils.logging_utils import setup_logger, log_rules_generation
from src.core.settings import settings

# Configure the client with your Gemini API key
genai.configure(api_key=settings.GEMINI_API_KEY)

class GeminiLLM(BaseLLM):
    """Gemini LLM implementation."""
    
    VALID_MODELS = [
        'gemini-pro',
        'gemini-2.0-flash',      # For complex reasoning
        'gemini-2.0-flash-lite', # For balanced responses
        'gemini-1.5-flash'       # For fastest responses
    ]
    
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        if model_name not in self.VALID_MODELS:
            valid_models_str = '\n- '.join([''] + self.VALID_MODELS)
            raise ValueError(f"Invalid Gemini model name: {model_name}. Valid models are:{valid_models_str}")
        # Store the original model name for API calls
        self.api_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name
        
        # Set up logger
        self.logger = setup_logger(f"gemini_{model_name.replace('-', '_')}")
    
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
        Generate insights about a founder using Gemini's API.
        
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
        status = "successful" if startup_success else "unsuccessful"
        outcome = "succeeded" if startup_success else "failed"

        # Format the lists into readable strings
        degrees_str = "\n- " + "\n- ".join(university_degrees) if university_degrees else "No university degrees listed"
        work_str = "\n- " + "\n- ".join(work_history) if work_history else "No work history listed"
        companies_str = "\n- " + "\n- ".join(previous_companies_founded) if previous_companies_founded else "No previous companies founded"

        # Construct the prompt with system message
        system_prompt = "You are a VC analyst. Analyze the given data of successful or unsuccessful founders and identify common features or patterns. Provide a concise summary of these common characteristics."
        
        user_prompt = f"""
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

        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Update this section to use the current Gemini API
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(full_prompt)
        
        reflection = response.text if response.text else "No reflection generated."
        
        # Return in the same format as before
        return {
            "Founder ID": founder_id,
            "Reflection (Natural Language)": reflection,
            "Outcome": "Success" if startup_success else "Failure",
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Model": self.model_name,
            "Temperature": self.temperature,
            "Provider": "gemini"
        }

    def generate_simple_logical_rules(self, reflections_text: str) -> str:
        """
        Generate simple logical rules from startup reflections (alternative to generate_logical_rules).
        This version generates simpler, more direct rules without detailed patterns.
        
        Args:
            reflections_text: Text containing startup reflections
            
        Returns:
            str: Generated logical rules
        """
        prompt = f"""
        Given the following reflections explaining why startups succeeded or failed:

        {reflections_text}

        Based on these reflections, clearly derive a concise set of logical rules in the structured format below:

        IF [condition A] AND [condition B] AND [condition C] THEN likelihood_of_success = HIGH (or LOW)

        Limit the number of rules to 3–5 concise, actionable, and generalizable conditions.
        """

        try:
            response = genai.generate_text(
                model=self.api_model_name,
                prompt=prompt,
                temperature=self.temperature,
            )
            return response.result if response.result else "No rules generated."
        except Exception as e:
            print(f"Error generating logical rules: {e}")
            raise

    def generate_logical_rules(self, reflections_text: str, use_validation: bool = False) -> str:
        """
        Generate logical rules from founder reflections.
        
        Args:
            reflections_text: Text containing founder reflections
            use_validation: Whether to use a second pass for validation (default: False)
            
        Returns:
            str: Generated logical rules
        """
        example_questions = """
        Example High-Quality Questions:
        1. Has the founder previously founded or co-founded a startup that raised more than $10M?
        2. Did the founder study in a top 20 ranked university based on QS World University Rankings 2023?
        3. Did the founder work as a product manager, software engineer, or researcher?
        4. Has the founder's previous venture achieved significant traction through acquisitions or IPOs?
        5. Has the founder received significant press and media coverage?
        6. Has the founder held a leadership position in a public tech company (NASDAQ)?
        """

        # First pass: Generate initial rules without examples
        initial_prompt = f"""
        Given the following reflections explaining why startups succeeded or failed:

        {reflections_text}

        Based on these reflections, derive a set of logical rules. For each rule:
        1. Use the IF-THEN format
        2. Assign a specific category
        3. Identify the success/failure pattern it represents
        4. Make conditions specific and measurable

        Format each rule as:

        Category: [Education/Experience/Technical/Market/Team/etc.]
        IF [specific condition A] AND [specific condition B] THEN likelihood_of_success = HIGH (or LOW)
        Success Pattern: [Description of what successful founders typically show in this area]
        Failure Pattern: [Description of what unsuccessful founders typically show in this area]

        Example format (but generate your own content based on the reflections):
        Category: Technical Background
        IF [founder has PhD in relevant field] AND [has 5+ years industry experience] THEN likelihood_of_success = HIGH
        Success Pattern: Deep technical expertise combined with practical industry application
        Failure Pattern: Lack of formal technical education or purely academic background

        Limit to 3-5 rules, ensuring each is specific, measurable, and backed by the reflections data.
        """

        system_prompt = "You are a VC analyst specializing in pattern recognition. Your task is to identify specific, actionable patterns that distinguish successful founders from unsuccessful ones."
        full_initial_prompt = f"{system_prompt}\n\n{initial_prompt}"

        try:
            # First pass: Generate initial rules
            self.logger.info("\nGenerating initial rules...")
            response = self.client.generate_content(
                full_initial_prompt,
                generation_config={"temperature": self.temperature}
            )
            initial_rules = response.text.strip()
            log_rules_generation(self.logger, "initial", initial_rules, "gemini", self.model_name)

            # Return initial rules if validation is disabled
            if not use_validation:
                return initial_rules

            # Second pass: Validate and improve rules using examples
            validation_prompt = f"""
            Below are some example high-quality questions and the generated rules. Please evaluate if the rules meet the quality standards shown in the examples.

            {example_questions}

            Generated Rules:
            {initial_rules}

            If the rules meet or exceed the quality standards of the examples, return them as is.
            If the rules need improvement, please provide an improved version that better matches the quality and specificity of the examples.
            Focus on:
            1. Specific, measurable criteria
            2. Clear decision paths
            3. Quantifiable thresholds where possible
            4. Professional experience indicators
            5. Educational background requirements
            """

            validation_system_prompt = "You are a VC analyst specializing in evaluating founder assessment criteria. Your task is to ensure the rules meet high quality standards."
            full_validation_prompt = f"{validation_system_prompt}\n\n{validation_prompt}"

            self.logger.info("\nValidating rules against quality standards...")
            validation_response = self.client.generate_content(
                full_validation_prompt,
                generation_config={"temperature": self.temperature}
            )
            
            final_rules = validation_response.text.strip()
            log_rules_generation(self.logger, "validation", final_rules, "gemini", self.model_name)
            
            return final_rules

        except Exception as e:
            self.logger.error(f"Error generating logical rules: {e}")
            raise

    def consolidate_simple_rules(self, all_rules_text: str) -> str:
        """
        Consolidate multiple sets of logical rules into a single set (alternative to consolidate_rules).
        This version uses a simpler prompt but maintains the required formatting for compatibility.
        
        Args:
            all_rules_text: Text containing multiple sets of rules
            
        Returns:
            str: Consolidated rules
        """
        system_prompt = "You are a VC analyst consolidating founder evaluation rules. You must create exactly 5 HIGH success rules (numbered 1-5) and 5 LOW success rules (numbered 6-10)."
        
        user_prompt = f"""
        Below are multiple sets of logical rules derived from startup reflections:

        {all_rules_text}

        Combine and consolidate these into one concise, clearly structured final set of logical rules. 
        Remove redundancy and ensure each rule is distinct and actionable. 

        IMPORTANT: You must create EXACTLY:
        - 5 rules for HIGH likelihood of success (numbered 1-5)
        - 5 rules for LOW likelihood of success (numbered 6-10)

        FORMATTING REQUIREMENTS:
        1. Start with "HIGH LIKELIHOOD OF SUCCESS RULES:" header
        2. List exactly 5 HIGH success rules, numbered 1-5
        3. Then add "LOW LIKELIHOOD OF SUCCESS RULES:" header
        4. List exactly 5 LOW success rules, numbered 6-10
        5. Each rule must be on its own line
        6. Each rule must start with its number followed by a period
        
        Example format:
        HIGH LIKELIHOOD OF SUCCESS RULES:
        1. Category: Technical Background
           IF [founder has PhD in relevant field] AND [has 5+ years industry experience] THEN likelihood_of_success = HIGH
        2. Category: Another High Success Rule...
        (and so on until rule 5)
        
        LOW LIKELIHOOD OF SUCCESS RULES:
        6. Category: Market Timing
           IF [market is highly saturated] AND [no clear differentiation] THEN likelihood_of_success = LOW
        7. Category: Another Low Success Rule...
        (and so on until rule 10)

        REMEMBER: You MUST output exactly 5 rules for each category, numbered 1-5 for HIGH and 6-10 for LOW.
        """

        prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature
                )
            )
            return response.text if response.text else "No consolidated rules generated."
        except Exception as e:
            print(f"Error consolidating rules: {e}")
            raise

    def consolidate_rules(self, all_rules_text: str) -> str:
        """
        Consolidate multiple sets of logical rules into a single coherent set.
        
        Args:
            all_rules_text: Text containing multiple sets of rules
            
        Returns:
            Consolidated set of logical rules as a string
        """
        prompt = f"""
        Below are multiple sets of logical rules derived from startup reflections:

        {all_rules_text}

        Combine and consolidate these into one concise, clearly structured final set of logical rules. 
        Remove redundancy and ensure each rule is distinct and actionable. 
        Clearly separate rules into two sections: HIGH likelihood of success and LOW likelihood of success.
        
        IMPORTANT FORMATTING REQUIREMENTS:
        1. Number each rule sequentially starting with "1." for the first rule
        2. Each rule should be on its own line
        3. Start each rule with its number followed by a period (e.g., "1.", "2.", etc.)
        4. Include a clear section header for each likelihood category
        
        Example format:
        HIGH LIKELIHOOD OF SUCCESS RULES:
        1. Category: Technical Background
           IF [founder has PhD in relevant field] AND [has 5+ years industry experience] THEN likelihood_of_success = HIGH
           Success Pattern: Deep technical expertise combined with practical industry application
           Failure Pattern: Lack of formal technical education or purely academic background
        
        LOW LIKELIHOOD OF SUCCESS RULES:
        1. Category: Market Timing
           IF [market is highly saturated] AND [no clear differentiation] THEN likelihood_of_success = LOW
           Success Pattern: Clear market gap and unique value proposition
           Failure Pattern: Entering crowded market without clear advantages
        
        Limit the total number of rules to approximately 8–12.
        """

        response = genai.generate_text(
            model=self.api_model_name,
            prompt=prompt,
            temperature=self.temperature,
        )
        
        return response.result if response.result else "No consolidated rules generated."

    def evaluate_founder_against_rules(self, 
                                     founder_info: str,
                                     rules_text: str) -> List[int]:
        """
        Evaluate a founder against a set of rules.
        
        Args:
            founder_info: Founder's profile and startup description
            rules_text: Text containing the rules to evaluate against
            
        Returns:
            List[int]: List of rule evaluation scores (1 for pass, 0 for fail, -1 for low likelihood)
        """
        prompt = f"""
        Given the following founder's background and startup description:
        
        {founder_info}
        
        Evaluate whether the founder meets the following rules. The rules are divided into two sections:
        1. HIGH Likelihood of Success Rules (should return 1 if met, 0 if not met)
        2. LOW Likelihood of Success Rules (should return -1 if met, 0 if not met)
        
        Rules:
        {rules_text}
        
        Return ONLY a comma-separated list of scores, where:
        1 = The founder meets a HIGH likelihood rule's condition
        0 = The founder does not meet the rule's condition
        -1 = The founder meets a LOW likelihood rule's condition
        
        Example output format: 1,0,1,0,1,-1,0,-1,0,-1
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=1000,
                )
            )

            # Parse the response into a list of integers
            scores_text = response.text.strip()
            try:
                scores = [int(score.strip()) for score in scores_text.split(',')]
                return scores
            except (ValueError, AttributeError):
                print(f"Error parsing scores from response: {scores_text}")
                # Count the number of actual rules by looking for lines containing 'IF' and 'THEN'
                rule_count = len([r for r in rules_text.split('\n') 
                                if 'IF' in r and 'THEN' in r and 'likelihood_of_success' in r])
                return [0] * rule_count
                
        except Exception as e:
            print(f"Error evaluating founder against rules: {e}")
            # Count the number of actual rules by looking for lines containing 'IF' and 'THEN'
            rule_count = len([r for r in rules_text.split('\n') 
                            if 'IF' in r and 'THEN' in r and 'likelihood_of_success' in r])
            return [0] * rule_count

