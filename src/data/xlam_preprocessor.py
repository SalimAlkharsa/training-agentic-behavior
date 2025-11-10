"""
xLAM Dataset Preprocessor

Formats xLAM function-calling data for training:
- Converts query + tools into instruction prompts
- Formats answers into expected completion format
- Handles JSON serialization for tools and function calls
"""

import json
from typing import Dict, List, Any, Optional
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XLAMPreprocessor:
    """
    Preprocessor for xLAM function-calling dataset.

    Converts raw xLAM examples into instruction-following format suitable
    for fine-tuning language models on function calling.
    """

    def __init__(
        self,
        prompt_template: str = "default",
        max_tools: Optional[int] = None
    ):
        """
        Initialize the preprocessor.

        Args:
            prompt_template: Which prompt template to use ('default', 'chat', 'qwen')
            max_tools: Maximum number of tools to include in context (None = all)
        """
        self.prompt_template = prompt_template
        self.max_tools = max_tools

    def format_tool_description(self, tool: Dict[str, Any]) -> str:
        """
        Format a single tool definition as a readable string.

        Args:
            tool: Tool dictionary with name, description, and parameters

        Returns:
            Formatted tool description string
        """
        tool_str = f"Function: {tool['name']}\n"
        tool_str += f"Description: {tool.get('description', 'No description')}\n"

        if 'parameters' in tool and tool['parameters']:
            tool_str += "Parameters:\n"
            for param_name, param_info in tool['parameters'].items():
                param_type = param_info.get('type', 'unknown')
                param_desc = param_info.get('description', 'No description')
                required = "required" if param_info.get('required', False) else "optional"
                tool_str += f"  - {param_name} ({param_type}, {required}): {param_desc}\n"

        return tool_str

    def format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format all available tools as a string.

        Args:
            tools: List of tool dictionaries

        Returns:
            Formatted string describing all tools
        """
        if self.max_tools and len(tools) > self.max_tools:
            tools = tools[:self.max_tools]
            logger.warning(f"Truncated tools list to {self.max_tools} tools")

        tools_str = "Available Functions:\n\n"
        for i, tool in enumerate(tools, 1):
            tools_str += f"{i}. "
            tools_str += self.format_tool_description(tool)
            tools_str += "\n"

        return tools_str.strip()

    def format_answer(self, answer: Dict[str, Any]) -> str:
        """
        Format a function call answer as a JSON string.

        Args:
            answer: Answer dictionary with function name and arguments

        Returns:
            JSON-formatted function call
        """
        # Create clean function call format
        function_call = {
            "name": answer.get('name', answer.get('function_name', '')),
            "arguments": answer.get('arguments', answer.get('parameters', {}))
        }
        return json.dumps(function_call, indent=2)

    def format_answers(self, answers: List[Dict[str, Any]]) -> str:
        """
        Format multiple function call answers.

        Args:
            answers: List of answer dictionaries

        Returns:
            Formatted string with all function calls
        """
        if len(answers) == 1:
            return self.format_answer(answers[0])
        else:
            # Multiple function calls
            calls = [self.format_answer(ans) for ans in answers]
            return json.dumps([json.loads(call) for call in calls], indent=2)

    def create_prompt(self, query: str, tools: List[Dict[str, Any]]) -> str:
        """
        Create the full instruction prompt from query and tools.

        Args:
            query: User's natural language query
            tools: List of available tools

        Returns:
            Formatted instruction prompt
        """
        tools_description = self.format_tools(tools)

        if self.prompt_template == "chat":
            prompt = f"""You are a helpful assistant with access to the following functions:

{tools_description}

To use a function, respond with a JSON object containing the function name and arguments.

User Query: {query}

Function Call:"""

        elif self.prompt_template == "qwen":
            # Qwen-style prompt format
            prompt = f"""<|im_start|>system
You are a helpful assistant with access to functions. Use them when appropriate.<|im_end|>
<|im_start|>functions
{tools_description}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

        else:  # default
            prompt = f"""Given the following functions and user query, select the appropriate function and provide the arguments.

{tools_description}

User Query: {query}

Function Call:"""

        return prompt

    def preprocess_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Preprocess a single xLAM example into instruction format.

        Args:
            example: Raw xLAM example with query, tools, and answers

        Returns:
            Dictionary with 'prompt' and 'completion' fields for training
        """
        query = example['query']
        tools = example['tools']
        answers = example['answers']

        # Create prompt (instruction)
        prompt = self.create_prompt(query, tools)

        # Create completion (expected output)
        completion = self.format_answers(answers)

        return {
            'prompt': prompt,
            'completion': completion,
            'query': query,  # Keep original for reference
            'num_tools': len(tools),
            'num_answers': len(answers)
        }

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess entire dataset.

        Args:
            dataset: HuggingFace Dataset with xLAM examples

        Returns:
            Processed dataset with prompt/completion pairs
        """
        logger.info(f"Preprocessing {len(dataset)} examples...")

        processed = dataset.map(
            self.preprocess_example,
            desc="Preprocessing xLAM examples"
        )

        logger.info(f"Preprocessing complete. Processed {len(processed)} examples")
        return processed

    def create_training_format(
        self,
        processed_dataset: Dataset,
        text_field: str = "text"
    ) -> Dataset:
        """
        Create final training format by combining prompt and completion.

        Args:
            processed_dataset: Dataset with separate prompt/completion fields
            text_field: Name of the combined text field for training

        Returns:
            Dataset with combined text field ready for training
        """
        def combine_prompt_completion(example):
            # Combine prompt and completion into single text field
            text = f"{example['prompt']}\n{example['completion']}"
            return {text_field: text}

        logger.info("Creating training format...")
        training_data = processed_dataset.map(combine_prompt_completion)

        return training_data


def main():
    """
    Example usage of the XLAMPreprocessor.
    """
    from xlam_loader import XLAMDatasetLoader

    # Load dataset
    loader = XLAMDatasetLoader()
    dataset = loader.load()

    # Handle DatasetDict
    if hasattr(dataset, 'keys'):
        dataset = dataset[list(dataset.keys())[0]]

    # Initialize preprocessor
    preprocessor = XLAMPreprocessor(prompt_template="default")

    # Preprocess first example
    example = dataset[0]
    processed = preprocessor.preprocess_example(example)

    print("\n=== Original Example ===")
    print(f"Query: {example['query'][:100]}...")
    print(f"Tools: {len(example['tools'])} functions")
    print(f"Answers: {len(example['answers'])} function calls")

    print("\n=== Processed Example ===")
    print(f"Prompt:\n{processed['prompt'][:300]}...")
    print(f"\nCompletion:\n{processed['completion']}")

    # Preprocess small subset
    small_dataset = dataset.select(range(10))
    processed_dataset = preprocessor.preprocess_dataset(small_dataset)

    print(f"\n=== Processed Dataset ===")
    print(f"Size: {len(processed_dataset)} examples")
    print(f"Fields: {list(processed_dataset.features.keys())}")


if __name__ == "__main__":
    main()
