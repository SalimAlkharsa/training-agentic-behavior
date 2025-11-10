"""
Simple script to load and print xLAM dataset examples.
"""

import json
from datasets import load_dataset


def main():
    # Load dataset
    print("Loading xLAM dataset...")
    dataset = load_dataset("Salesforce/xlam-function-calling-60k")

    # Get train split
    data = dataset['train']
    print(f"Total examples: {len(data)}\n")

    # First, inspect the structure
    print("Inspecting data structure...")
    first = data[0]
    print(f"Fields: {first.keys()}")
    print(f"Tools type: {type(first['tools'])}")
    print(f"Answers type: {type(first['answers'])}")

    # Tools and answers might be JSON strings, not dicts
    tools = first['tools'] if isinstance(first['tools'], list) else json.loads(first['tools'])
    answers = first['answers'] if isinstance(first['answers'], list) else json.loads(first['answers'])

    print(f"Parsed tools count: {len(tools)}")
    print(f"Parsed answers count: {len(answers)}\n")

    # Print 3 examples
    for i in range(3):
        ex = data[i]

        print(f"\n{'='*60}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*60}")

        print(f"\nQuery: {ex['query']}")

        # Parse tools (might be JSON string)
        tools = ex['tools'] if isinstance(ex['tools'], list) else json.loads(ex['tools'])
        print(f"\nAvailable Tools ({len(tools)}):")
        for tool in tools[:5]:  # Show first 5 only
            print(f"  - {tool['name']}: {tool.get('description', 'N/A')[:60]}...")
        if len(tools) > 5:
            print(f"  ... and {len(tools)-5} more tools")

        # Parse answers (might be JSON string)
        answers = ex['answers'] if isinstance(ex['answers'], list) else json.loads(ex['answers'])
        print(f"\nExpected Function Call(s):")
        for ans in answers:
            print(f"  Function: {ans['name']}")
            print(f"  Arguments: {json.dumps(ans['arguments'], indent=4)}")

    # Basic stats
    print(f"\n{'='*60}")
    print("DATASET STATS (first 1000 examples)")
    print(f"{'='*60}")

    sample = data.select(range(1000))

    def get_len(field):
        """Handle both list and JSON string formats."""
        return len(field) if isinstance(field, list) else len(json.loads(field))

    avg_tools = sum(get_len(ex['tools']) for ex in sample) / 1000
    avg_calls = sum(get_len(ex['answers']) for ex in sample) / 1000

    print(f"Average tools per example: {avg_tools:.1f}")
    print(f"Average function calls per example: {avg_calls:.1f}")


if __name__ == "__main__":
    main()
