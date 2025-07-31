from anthropic import Anthropic, APIError
import time
from typing import Dict, Any, List
import pandas as pd

from ..base import BaseLLM
from src.core.settings import settings
from src.utils.logging_utils import setup_logger, log_rules_generation

# Configure the client with your Anthropic API key
client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

class AnthropicLLM(BaseLLM):
    """Anthropic LLM implementation."""
    
    VALID_MODELS = [
        'claude-3-7-sonnet-latest',
        'claude-3-5-haiku-latest',
        'claude-3-haiku-20240307'
    ]
    
    def __init__(self, model_name: str, temperature: float):
        super().__init__(model_name, temperature)
        if model_name not in self.VALID_MODELS:
            valid_models_str = '\n- '.join([''] + self.VALID_MODELS)
            raise ValueError(f"Invalid Anthropic model name: {model_name}. Valid models are:{valid_models_str}")
        
        # Set up logger
        self.logger = setup_logger(f"anthropic_{model_name.replace('-', '_')}")
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

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
        Generate insights about a founder using Anthropic's Claude.
        
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
        
        Raises:
            APIError: If there's an error calling the Anthropic API
        """
        status = "successful" if startup_success else "unsuccessful"
        outcome = "succeeded" if startup_success else "failed"

        # Format the lists into readable strings
        degrees_str = "\n- " + "\n- ".join(university_degrees) if university_degrees else "No university degrees listed"
        work_str = "\n- " + "\n- ".join(work_history) if work_history else "No work history listed"
        companies_str = "\n- " + "\n- ".join(previous_companies_founded) if previous_companies_founded else "No previous companies founded"

        # Construct the prompt
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
            # Generate the reflection using Anthropic's API
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=self.temperature,
                system="You are a VC analyst. Analyze the given data of successful or unsuccessful founders and identify common features or patterns. Provide a concise summary of these common characteristics.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            reflection = response.content[0].text if response.content else "No reflection generated."
            
            # Return in the same format as other providers
            return {
                "Founder ID": founder_id,
                "Reflection (Natural Language)": reflection,
                "Outcome": "Success" if startup_success else "Failure",
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Model": self.model_name,
                "Temperature": self.temperature,
                "Provider": "anthropic"
            }
            
        except APIError as e:
            raise APIError(f"Anthropic API error: {str(e)}. Make sure you're using a valid model name from: {', '.join(self.VALID_MODELS)}")

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
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
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

        try:
            # First pass: Generate initial rules
            self.logger.info("\nGenerating initial rules...")
            response = self.client.messages.create(
                model=self.model_name,
                system="You are a VC analyst specializing in pattern recognition. Your task is to identify specific, actionable patterns that distinguish successful founders from unsuccessful ones.",
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=self.temperature,
                max_tokens=2000
            )
            initial_rules = response.content[0].text.strip()
            log_rules_generation(self.logger, "initial", initial_rules, "anthropic", self.model_name)

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

            self.logger.info("\nValidating rules against quality standards...")
            validation_response = self.client.messages.create(
                model=self.model_name,
                system="You are a VC analyst specializing in evaluating founder assessment criteria. Your task is to ensure the rules meet high quality standards.",
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            final_rules = validation_response.content[0].text.strip()
            log_rules_generation(self.logger, "validation", final_rules, "anthropic", self.model_name)
            
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
        prompt = f"""
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

        try:
            response = self.client.messages.create(
                model=self.model_name,
                system="You are a VC analyst consolidating founder evaluation rules. You must create exactly 5 HIGH success rules (numbered 1-5) and 5 LOW success rules (numbered 6-10).",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000
            )
            return response.content[0].text.strip()
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
            
        Raises:
            APIError: If there's an error calling the Anthropic API
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

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text if response.content else "No consolidated rules generated."
            
        except APIError as e:
            raise APIError(f"Anthropic API error: {str(e)}. Make sure you're using a valid model name from: {', '.join(self.VALID_MODELS)}")

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
        
        HIGH LIKELIHOOD OF SUCCESS RULES (Rules 1-5):
        - Return 1 if the founder meets the rule's conditions
        - Return 0 if the founder does not meet the rule's conditions
        
        LOW LIKELIHOOD OF SUCCESS RULES (Rules 6-10):
        - Return -1 if the founder meets the rule's conditions (indicating increased likelihood of failure)
        - Return 0 if the founder does not meet the rule's conditions
        
        Rules:
        {rules_text}
        
        Return ONLY a comma-separated list of scores. Make sure to return -1 for LOW likelihood rules that are met.
        
        Example output format: 1,0,1,1,0,-1,0,-1,0,-1
        
        IMPORTANT: The first 5 scores should be 0 or 1 (HIGH likelihood rules)
                  The last 5 scores should be 0 or -1 (LOW likelihood rules)
        """

        try:
            response = self.client.messages.create(
                model=self.model_name,
                system="You are a VC analyst evaluating founders against specific criteria. For HIGH likelihood rules (1-5), return 1 if met. For LOW likelihood rules (6-10), return -1 if met. Return 0 if any rule is not met.",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000
            )

            # Parse the response into a list of integers
            scores_text = response.content[0].text.strip()
            try:
                scores = [int(score.strip()) for score in scores_text.split(',')]
                
                # Ensure proper scoring
                for i in range(len(scores)):
                    if i < 5:  # HIGH likelihood rules (1-5)
                        scores[i] = 1 if scores[i] > 0 else 0
                    else:  # LOW likelihood rules (6-10)
                        scores[i] = -1 if scores[i] < 0 else 0
                        
                return scores
            except (ValueError, AttributeError):
                print(f"Error parsing scores from response: {scores_text}")
                rule_count = len([r for r in rules_text.split('\n') 
                                if 'IF' in r and 'THEN' in r and 'likelihood_of_success' in r])
                return [0] * rule_count
                
        except Exception as e:
            print(f"Error evaluating founder against rules: {e}")
            rule_count = len([r for r in rules_text.split('\n') 
                            if 'IF' in r and 'THEN' in r and 'likelihood_of_success' in r])
            return [0] * rule_count

    def generate_structured_founder_insight(self, 
                                          founder_id: str,
                                          founder_name: str,
                                          university_degrees: list,
                                          work_history: list,
                                          previous_companies_founded: list,
                                          startup_success: bool,
                                          professional_background: str,
                                          company_location: str) -> Dict[str, Any]:
        """
        Generate structured insights about a founder using Anthropic's Claude.
        This version provides a more focused, structured summary.
        
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

        # Construct the prompt
        prompt = f"""
        Given the following founder's background and startup details, generate a concise summary (50-100 words) that highlights key factors related to their success or failure.

        Do NOT include the founder's name—instead, use 'this founder' or 'the founder'.
        Summarize key traits: Education, work experience, industry expertise, entrepreneurial experience, market understanding, adaptability, and the startup's outcome (success or failure).
        Avoid overly specific numbers and university names unless the ranking/prestige is crucial. Instead, use 'a well-ranked university' or 'a strong academic background'.
        Ensure the summary is structured, concise, and clearly presents relevant success/failure factors.

        Founder Information:
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
            # Generate the reflection using Anthropic's API
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=self.temperature,
                system="You are a VC analyst specializing in founder analysis. Provide concise, structured insights about founder characteristics and their impact on startup outcomes.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            reflection = response.content[0].text if response.content else "No reflection generated."
            
            # Return in the same format as other providers
            return {
                "Founder ID": founder_id,
                "Reflection (Natural Language)": reflection,
                "Outcome": "Success" if startup_success else "Failure",
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Model": self.model_name,
                "Temperature": self.temperature,
                "Provider": "anthropic"
            }
            
        except APIError as e:
            raise APIError(f"Anthropic API error: {str(e)}. Make sure you're using a valid model name from: {', '.join(self.VALID_MODELS)}")

    def generate_dataframe_based_questions(self, dataframe: pd.DataFrame, founder_advice: str, num_questions: int = 10) -> List[str]:
        """
        Generate questions based on dataframe features and founder advice.
        
        Args:
            dataframe: DataFrame containing founder features
            founder_advice: Text containing advice about successful founders
            num_questions: Number of questions to generate (default: 10)
            
        Returns:
            List[str]: List of generated questions
        """
        # Convert dataframe columns to a string for the prompt
        columns_str = "\n".join([f"- {col}" for col in dataframe.columns])
        
        prompt = f"""You must generate EXACTLY {num_questions} Yes/No questions about founders.
Each question must be on its own line.
Do not include any numbering, formatting, or additional text.
Do not use markdown code blocks.
Each question must end with a question mark.
Each question must be answerable with only Yes or No.

Example output format:
Has the founder previously worked at a FAANG company?
Did the founder graduate from an Ivy League university?
Has the founder's previous startup achieved an exit?

Available features in the dataframe:
{columns_str}

Summarized trends of successful founders:
{founder_advice}

Requirements:
1. Questions must be objective and directly answerable from the data
2. Focus on concrete achievements and measurable facts
3. Avoid subjective assessments or opinions
4. Do not use the startup_success column

Generate exactly {num_questions} questions in the specified format."""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=self.temperature,
                system="You are a VC analyst generating founder evaluation questions. Return ONLY the questions, exactly one per line, no other text.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Clean response to get only questions
            response_text = response.content[0].text.strip()
            questions = []
            
            # Parse questions from response
            for line in response_text.split('\n'):
                line = line.strip()
                if line and '?' in line and not line.startswith(('```', '[', ']', '{', '}', '#', '1.', '2.', '3.')):
                    questions.append(line)
            
            # Ensure we return exactly num_questions
            return questions[:num_questions]
            
        except Exception as e:
            print(f"Error generating questions: {e}")
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
            # Create a very strict prompt that enforces JSON structure
            json_examples = r'''CORRECT FORMAT:
[{"Founder ID": "123", "Answer": "Yes"}, {"Founder ID": "456", "Answer": "No"}]

INCORRECT FORMATS (DO NOT USE):
❌ ```json
[{"Founder ID": "123", "Answer": "Yes"}]
```
❌ Here's the JSON array:
[{"Founder ID": "123", "Answer": "Yes"}]
❌ [{"Founder ID": "123", "Answer": "Yes"},]
❌ [{"Founder ID": "123", "Answer": "yes"}]
❌ [{"Founder ID": "123", "Answer": "Yes"},]
❌ [{"Founder ID": "123", "Answer": "Yes"}, {"Founder ID": "456", "Answer": "No"},]'''

            enhanced_prompt = f"""CRITICAL: You must return ONLY a JSON array with no additional text or formatting.

STRICT REQUIREMENTS:
1. Response must be a valid JSON array starting with [ and ending with ]
2. Each object in the array must have exactly these fields:
   - "Founder ID": string (the founder's ID)
   - "Answer": string (must be exactly "Yes" or "No")
3. No explanations, no markdown, no code blocks
4. No extra text before or after the array
5. No line breaks or formatting within the array
6. No trailing commas
7. All strings must be properly escaped

{json_examples}

Task:
{prompt}

Remember: Return ONLY the JSON array. Any additional text or formatting will cause errors."""

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=self.temperature,
                system="You are a JSON generator. Your ONLY task is to output valid JSON arrays. Any other output format is an error. Do not include any explanations, markdown, or additional text.",
                messages=[{"role": "user", "content": enhanced_prompt}]
            )
            
            # Get the response text and clean it
            response_text = response.content[0].text.strip()
            
            # Remove any potential markdown code block markers and clean up
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            response_text = response_text.replace('\n', '').replace('\r', '')
            
            # Try to parse as JSON first to validate structure
            import json
            try:
                # First try to parse the response as is
                try:
                    parsed_json = json.loads(response_text)
                except json.JSONDecodeError:
                    # If that fails, try to clean up common issues
                    # Remove any trailing commas
                    response_text = response_text.replace(',]', ']')
                    response_text = response_text.replace(',}', '}')
                    # Remove any extra whitespace
                    response_text = ' '.join(response_text.split())
                    # Try parsing again
                    parsed_json = json.loads(response_text)
                
                if not isinstance(parsed_json, list):
                    raise ValueError("Response is not a JSON array")
                
                # Validate each object in the array
                for item in parsed_json:
                    if not isinstance(item, dict):
                        raise ValueError("Each item in the array must be an object")
                    if "Founder ID" not in item or "Answer" not in item:
                        raise ValueError("Each object must have 'Founder ID' and 'Answer' fields")
                    if item["Answer"] not in ["Yes", "No"]:
                        raise ValueError("Answer must be exactly 'Yes' or 'No'")
                
                # If we got here, the JSON is valid - return the cleaned version
                return json.dumps(parsed_json)
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON structure: {str(e)}")
            
        except APIError as e:
            raise APIError(f"Anthropic API error: {str(e)}. Make sure you're using a valid model name from: {', '.join(self.VALID_MODELS)}")

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
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=self.temperature,
                system=system_message,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            self.logger.error(f"Error sending prompt: {e}")
            raise Exception(f"Anthropic API error: {str(e)}")