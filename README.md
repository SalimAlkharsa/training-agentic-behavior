Note this is still in progress, this README is not accurate and only reflects my ideal vision of this when done. Do not put too much attention to this right now, it was just me describing to ChatGPT how I want things to happen and the modularity I expect so I can guide my work and look at this as a checklist

# AgentCoder

**A lightweight experimental framework for studying tool-use behavior in small code-generation models**

AgentCoder explores when and how code-generation models choose to use external tools versus reasoning symbolically. Using open models (DeepSeek-Coder, StarCoder2, Qwen2.5-Coder, Phi-3-mini), the framework provides a safe execution harness, synthetic datasets, and fine-tuning pipelines (LoRA, DPO, PPO) to improve tool-selection decisions.

---

## Research Questions

1. **When do models choose tools?** Under what conditions do code models decide to execute code vs. reason symbolically?
2. **Can we improve tool selection?** Can fine-tuning teach better decisions about tool usage?
3. **What are the trade-offs?** How do correctness, latency, and tool-selection accuracy vary across training approaches?

---

## Features

### Core Capabilities

- **Multi-Model Support**: Qwen2.5-Coder, DeepSeek-Coder, StarCoder2, Phi-3-mini
- **Safe Code Execution**: Sandboxed Python execution with resource limits
- **Tool System**: Registry of tools (code_executor, symbolic_math, web_search, file_ops)
- **Synthetic Data Generation**: Automated creation of tool-use training data
- **Fine-Tuning Pipelines**: LoRA, DPO, and PPO implementations
- **Comprehensive Evaluation**: Correctness, latency, tool-selection accuracy metrics
- **Quantization Support**: 4-bit/8-bit for 8GB RAM systems
- **Reproducibility**: Extensive logging and experiment tracking

---

## Project Structure

```
training-agentic-behavior/
├── src/
│   ├── models/          # Model wrappers (Qwen, DeepSeek, StarCoder, Phi)
│   ├── sandbox/         # Safe code execution environments
│   ├── tools/           # Tool definitions and registry
│   ├── training/        # LoRA/DPO/PPO training pipelines
│   ├── data/            # Synthetic dataset generation
│   ├── evaluation/      # Metrics, benchmarks, analysis
│   └── utils/           # Metrics recording, logging
├── data/
│   ├── raw/             # Original data sources
│   ├── synthetic/       # Generated tool-use prompts
│   └── processed/       # Formatted training data
├── models/
│   ├── base/            # Pre-trained model cache
│   ├── lora/            # LoRA adapter checkpoints
│   └── dpo/             # DPO-trained models
├── results/
│   ├── metrics/         # Performance data (JSON)
│   ├── plots/           # Visualizations
│   └── logs/            # Training and evaluation logs
├── experiments/         # Experiment configs and scripts
├── notebooks/           # Jupyter analysis notebooks
└── tests/              # Unit and integration tests
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd training-agentic-behavior

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Baseline Evaluation

```bash
# Evaluate pre-trained model on synthetic dataset
python experiments/run_baseline.py --model qwen2.5-coder-3b --quantization 4bit
```

### 3. Generate Synthetic Dataset

```bash
# Create 200 tool-use prompts with ground truth labels
python -m src.data.generate_synthetic --num_samples 200 --output data/synthetic/tool_use_v1.json
```

### 4. Fine-Tune with LoRA

```bash
# Train LoRA adapters on synthetic data
python -m src.training.train_lora \
    --model qwen2.5-coder-3b \
    --data data/synthetic/tool_use_v1.json \
    --output models/lora/qwen_tool_v1
```

### 5. Compare Models

```bash
# Evaluate baseline vs. fine-tuned
python experiments/compare_models.py \
    --baseline qwen2.5-coder-3b \
    --finetuned models/lora/qwen_tool_v1 \
    --output results/comparison_report.html
```

---

## Usage Examples

### Basic Model Inference

```python
from src.models import Qwen2Wrapper

# Load model with quantization
model = Qwen2Wrapper(quantization="4bit", enable_metrics=True)
model.load_model()

# Generate code
prompt = "Calculate the factorial of 10"
output = model.generate(prompt, max_length=256, temperature=0.7)
print(output)

# View metrics
model.print_metrics_summary()
```

### Safe Code Execution

```python
from src.sandbox import PythonSandbox

# Create sandbox
sandbox = PythonSandbox(timeout=5, memory_limit_mb=100)

# Execute code safely
code = "print(sum(range(1, 101)))"
result = sandbox.execute(code)

print(f"Output: {result.output}")
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time}s")
```

### Tool-Aware Generation

```python
from src.tools import ToolRegistry
from src.models import AgentCoderModel

# Initialize tools and model
tools = ToolRegistry()
tools.register("code_executor", sandbox.execute)
tools.register("symbolic_math", symbolic_solver)

model = AgentCoderModel(
    base_model="qwen2.5-coder-3b",
    tools=tools,
    quantization="4bit"
)

# Generate with tool selection
prompt = "What is the sum of squares from 1 to 100?"
response = model.generate_with_tools(prompt)

print(f"Chosen tool: {response.tool_used}")
print(f"Output: {response.output}")
print(f"Reasoning: {response.reasoning}")
```

### Synthetic Data Generation

```python
from src.data import SyntheticDataGenerator

# Create generator
generator = SyntheticDataGenerator()

# Generate diverse prompts
dataset = generator.generate(
    num_samples=100,
    categories=["execution_required", "pure_reasoning", "ambiguous"],
    difficulty_levels=["easy", "medium", "hard"]
)

# Save dataset
generator.save(dataset, "data/synthetic/dataset_v1.json")
```

### LoRA Fine-Tuning

```python
from src.training import LoRATrainer

# Configure trainer
trainer = LoRATrainer(
    model_name="qwen2.5-coder-3b",
    lora_rank=16,
    lora_alpha=32,
    learning_rate=5e-5,
    num_epochs=3
)

# Load data and train
trainer.load_dataset("data/synthetic/tool_use_v1.json")
trainer.train()

# Save adapter
trainer.save_adapter("models/lora/qwen_tool_v1")
```

### Evaluation

```python
from src.evaluation import ToolUseEvaluator

# Create evaluator
evaluator = ToolUseEvaluator()

# Evaluate model
results = evaluator.evaluate(
    model=model,
    dataset="data/synthetic/test_set.json",
    metrics=["correctness", "tool_accuracy", "latency"]
)

# Print results
evaluator.print_report(results)
evaluator.save_plots("results/plots/evaluation_v1.png")
```

---

## Synthetic Dataset Design

### Prompt Categories

1. **Pure Reasoning** (should NOT use tools)

   - Code explanation, algorithm design, syntax analysis
   - Example: "Explain how quicksort works"

2. **Execution Required** (should use `code_executor`)

   - Numerical computation, data transformation, runtime verification
   - Example: "Calculate the 50th Fibonacci number"

3. **Ambiguous** (judgment call)

   - Simple math, small transformations
   - Example: "What is 15 factorial?"

4. **Multi-Step** (complex tool sequences)
   - Execute → analyze → re-execute
   - Example: "Generate prime numbers, sum them, then find if sum is prime"

### Data Format

```json
{
  "prompt": "Calculate the sum of squares of numbers 1 to 100",
  "optimal_tool": "code_executor",
  "reasoning": "Numerical computation best verified by execution",
  "difficulty": "easy",
  "category": "execution_required",
  "ground_truth_output": "338350",
  "alternative_tools": ["symbolic_math"]
}
```

---

## Training Approaches

### 1. LoRA (Low-Rank Adaptation)

- **Best for**: Quick experiments, limited compute
- **Pros**: Fast training, small adapter files (~10MB), good results
- **Cons**: Limited to supervised learning

### 2. DPO (Direct Preference Optimization)

- **Best for**: Learning from preferences, stable training
- **Pros**: Simpler than PPO, no reward model needed
- **Cons**: Requires preference pairs (good/bad examples)

### 3. PPO (Proximal Policy Optimization)

- **Best for**: Complex reward functions, RL research
- **Pros**: Can optimize for multiple objectives
- **Cons**: Training complexity, hyperparameter sensitivity

---

## Evaluation Metrics

### Primary Metrics

1. **Correctness Rate**: % of tasks solved correctly
2. **Tool Selection Accuracy**: % of optimal tool choices (precision, recall, F1)
3. **Inference Latency**: Time to generate response (ms)
4. **Execution Efficiency**: Unnecessary tool calls avoided

### Secondary Metrics

- Token efficiency (output length)
- Error recovery rate (multi-step tasks)
- Memory usage per inference
- Training convergence speed

---

## Memory Requirements

Approximate VRAM/RAM usage by model:

| Model               | Quantization | Memory  | Notes                       |
| ------------------- | ------------ | ------- | --------------------------- |
| Qwen2.5-Coder-3B    | 4-bit        | ~2-3 GB | Recommended for 8GB systems |
| Qwen2.5-Coder-3B    | 8-bit        | ~4-5 GB | Better quality              |
| DeepSeek-Coder-1.3B | 4-bit        | ~1-2 GB | Fastest training            |
| StarCoder2-3B       | 4-bit        | ~2-3 GB | Good code understanding     |
| Phi-3-mini (3.8B)   | 4-bit        | ~3-4 GB | Strong reasoning            |

---

## Reproducibility

All experiments log:

- Random seeds (Python, NumPy, PyTorch)
- Model configurations and hyperparameters
- Dataset versions and splits
- Hardware specs (GPU/CPU, memory)
- Software versions (PyTorch, transformers, PEFT, TRL)
- Git commit hash and timestamp

Configuration files are stored in `experiments/configs/` and can be rerun:

```bash
python experiments/replay.py --config experiments/configs/experiment_20250102_143022.json
```

---

## Project Roadmap

### Phase 1: Infrastructure (Current)

- ✅ Model wrapper with quantization
- ✅ Metrics recording
- 🚧 Code execution sandbox
- 🚧 Tool registry system

### Phase 2: Data & Baseline

- 🔲 Synthetic dataset generation (200+ prompts)
- 🔲 Baseline evaluation pipeline
- 🔲 Failure pattern analysis

### Phase 3: Training

- 🔲 LoRA fine-tuning implementation
- 🔲 DPO preference optimization
- 🔲 PPO reinforcement learning (optional)

### Phase 4: Analysis

- 🔲 Comparative evaluation
- 🔲 Visualization and reporting
- 🔲 Research findings documentation

---

## Development

### Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_sandbox.py -v

# With coverage
pytest --cov=src tests/
```

### Code Style

```bash
# Format code
black src/ tests/

# Check types
mypy src/

# Lint
ruff check src/
```

### Adding a New Model

1. Create wrapper in `src/models/new_model_wrapper.py`
2. Inherit from `BaseModelWrapper`
3. Implement `load_model()` and `generate()` methods
4. Add to model registry in `src/models/__init__.py`
5. Add tests in `tests/test_models.py`

---

## Documentation

- **Project Context**: See `.claude/claude.md` for detailed technical architecture
- **API Reference**: See `docs/api_reference.md`
- **Experiment Guide**: See `docs/running_experiments.md`
- **Dataset Format**: See `docs/dataset_format.md`

---

## Citation

If you use AgentCoder in your research, please cite:

```bibtex
@software{agentcoder2025,
  title={AgentCoder: A Framework for Studying Tool-Use in Code Generation Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/agentcoder}
}
```

---

## License

MIT License - See LICENSE file for details.

Model licenses vary by provider (Qwen, DeepSeek, StarCoder2, Phi-3). Please refer to individual model cards on HuggingFace.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

---

## Acknowledgments

Built with:

- [HuggingFace Transformers](https://huggingface.co/transformers)
- [PEFT](https://github.com/huggingface/peft) for LoRA
- [TRL](https://github.com/huggingface/trl) for RL training
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
