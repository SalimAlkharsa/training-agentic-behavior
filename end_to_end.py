"""
End-to-end demo: Load example -> Format prompt -> Model generates -> Evaluate metrics
"""

import json
import sys
import argparse
from pathlib import Path
from datasets import load_dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.models.qwen_wrapper import Qwen2Wrapper


def format_prompt(query, tools):
    """Create prompt for the model."""
    prompt = "Given these functions:\n\n"

    for tool in tools:
        prompt += f"- {tool['name']}: {tool.get('description', 'No description')}\n"
        if 'parameters' in tool:
            prompt += f"  Parameters: {list(tool['parameters'].keys())}\n"

    prompt += f"\nUser query: {query}\n\n"
    prompt += "Respond with the function call in JSON format:\n"

    return prompt


def get_model_output(prompt, model=None, debug=False):
    """Generate function call prediction using Qwen model."""
    if debug:
        # Debug mode: return mock output without loading model
        generated = """{
  "function_call": {
    "name": "live_giveaways_by_type",
    "arguments": {
      "type": ["beta access", "games"]
    }
  }
}"""
    else:
        # Generate with low temperature for deterministic output
        output = model.generate(prompt, max_length=1024, temperature=0.1, do_sample=False)
        # Extract generated part (after prompt)
        generated = output[len(prompt):].strip()

    print("\nModel raw output:")
    print(generated[:500])

    try:
        # Find first JSON object in output
        start = generated.find("{")
        end = generated.find("}", start) + 1

        # Keep extending to find complete JSON object
        brace_count = 1
        for i in range(start + 1, len(generated)):
            if generated[i] == "{":
                brace_count += 1
            elif generated[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if start != -1 and end > start:
            parsed = json.loads(generated[start:end])

            # Handle nested function_call structure
            if "function_call" in parsed:
                return parsed["function_call"]
            return parsed

        return {"name": "parse_error", "arguments": {}}
    except Exception as e:
        print(f"Parse error: {e}")
        return {"name": "parse_error", "arguments": {}}


def evaluate_prediction(prediction, ground_truth):
    """Calculate accuracy metrics."""
    # Function name match
    function_match = prediction['name'] == ground_truth['name']

    # Exact argument match (simple comparison)
    args_match = prediction['arguments'] == ground_truth['arguments']

    # Exact match (both correct)
    exact_match = function_match and args_match

    return {
        'exact_match': exact_match,
        'function_match': function_match,
        'args_match': args_match
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="End-to-end function calling demo")
    parser.add_argument("--debug", action="store_true", help="Use mock model output instead of loading actual model")
    args = parser.parse_args()

    # Load one example
    print("Loading dataset...")
    dataset = load_dataset("Salesforce/xlam-function-calling-60k")
    example = dataset['train'][0]

    # Parse tools and answers (might be JSON strings)
    tools = example['tools'] if isinstance(example['tools'], list) else json.loads(example['tools'])
    answers = example['answers'] if isinstance(example['answers'], list) else json.loads(example['answers'])

    print("\n" + "="*70)
    print("STEP 1: RAW DATA")
    print("="*70)
    print(f"Query: {example['query']}")
    print(f"\nTools available: {len(tools)}")
    print(f"Expected function calls: {len(answers)}")

    # For simplicity, use first answer
    ground_truth = answers[0]

    print("\n" + "="*70)
    print("STEP 2: FORMAT PROMPT (what model receives)")
    print("="*70)
    prompt = format_prompt(example['query'], tools)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    print("\n" + "="*70)
    print("STEP 3: EXPECTED OUTPUT (ground truth)")
    print("="*70)
    print(json.dumps(ground_truth, indent=2))

    print("\n" + "="*70)
    print("STEP 4: GENERATE PREDICTION")
    print("="*70)

    if args.debug:
        print("Using debug mode (mock output)...")
        prediction = get_model_output(prompt, debug=True)
    else:
        print("Loading Qwen2.5-Coder-3B model...")
        model = Qwen2Wrapper()
        model.load_model()
        print("\nGenerating prediction...")
        prediction = get_model_output(prompt, model, debug=False)

    print(f"\nModel prediction:")
    print(json.dumps(prediction, indent=2))

    print("\n" + "="*70)
    print("STEP 5: EVALUATE METRICS")
    print("="*70)
    metrics = evaluate_prediction(prediction, ground_truth)
    print(f"Exact Match: {metrics['exact_match']}")
    print(f"Function Name Match: {metrics['function_match']}")
    print(f"Arguments Match: {metrics['args_match']}")

    print("\n" + "="*70)
    print("END-TO-END FLOW COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
