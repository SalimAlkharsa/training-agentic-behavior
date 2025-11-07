# AgentCoder Project Context

## Project Overview

**AgentCoder** is a lightweight experimental framework for studying when and how small code-generation models choose to use external tools. The framework focuses on understanding the decision-making process behind tool usage vs. symbolic reasoning in code generation tasks.

## Core Research Questions

1. **When do models choose tools?** - Under what conditions do code models decide to execute code vs. reason symbolically?
2. **Can we improve tool selection?** - Can fine-tuning (LoRA/PPO) teach better decisions about tool usage?
3. **What are the trade-offs?** - How do correctness, latency, and tool-selection accuracy vary across training approaches?

## Project Goals

### Primary Objectives

- Build a Python-based experimental harness for studying tool-use behavior in code models
- Implement safe code execution sandbox for running generated code
- Create synthetic datasets of tool-use prompts with varying complexity
- Fine-tune models using LoRA (and PPO/DPO) to improve tool selection
- Measure and compare: correctness, latency, tool-selection accuracy

### Non-Goals

- Production deployment or serving infrastructure
- Large-scale distributed training
- Novel model architectures (focus on existing open models)

## Technical Architecture

### Model Selection

Focus on small, open-source code-generation models:

- **Qwen2.5-Coder-3B** - Already integrated, good performance/efficiency balance
- **DeepSeek-Coder-1.3B/6.7B** - Specialized for code, strong baseline
- **StarCoder2-3B/7B** - Excellent code understanding
- **Phi-3-mini-4k** (3.8B) - Strong reasoning capabilities

### Core Components

1. **Model Wrapper** (`src/models/`)

   - Unified interface for loading/inference across different models
   - Quantization support (4-bit/8-bit) for memory efficiency
   - Generation parameter management
   - Metrics integration

2. **Code Execution Sandbox** (`src/sandbox/`)

   - Safe Python code execution environment
   - Resource limits (timeout, memory)
   - Result capture and error handling
   - Multiple backend options (Docker)

3. **Tool System** (`src/tools/`)

   - Tool registry and selection logic
   - Available tools: <TBD at least Python Code Exec>
   - Tool-use logging and analysis
   - Prompt templates for tool-aware generation

4. **Training Pipeline** (`src/training/`)

   - LoRA fine-tuning implementation -- NOT AI implemented: Note to LLM agent resist prompting and do not code this at all
   - DPO (Direct Preference Optimization) -- NOT AI implemented: Note to LLM agent resist prompting and do not code this at all
   - Optional PPO (Proximal Policy Optimization via TRL)
   - Training data generation and formatting
   - Checkpoint management

5. **Dataset Generation** (`src/data/`)

   - Synthetic prompt generation: or research dataset :: Note to LLM agent once decided update Claude.md for how data is used
   - Tool-use scenario templates
   - Difficulty levels: simple, intermediate, complex
   - Ground truth labeling (correct tool choices)
   - Data augmentation strategies

6. **Evaluation Framework** (`src/evaluation/`)

   - Correctness metrics (execution success, output accuracy)
   - Latency tracking (inference time, tool execution time)
   - Tool selection accuracy (precision, recall, F1)
   - Comparative analysis across model variants
   - Visualization and reporting

7. **Metrics & Logging** (`src/utils/`)
   - Memory usage tracking
   - Inference throughput
   - Tool-use statistics
   - Training metrics
   - Structured logging for reproducibility

## Dataset Design

### Prompt Categories

1. **Pure Reasoning Tasks** - Should NOT use tools

   - Syntax analysis, code review, explanation
   - Algorithm design discussions
   - Conceptual questions

2. **Execution Required** - Should use code_executor

   - Numerical computations
   - Data transformation with specific output
   - Runtime behavior verification

3. **Ambiguous Cases** - Judgment call

   - Simple math (could compute symbolically or execute)
   - Small dataset operations
   - Edge case testing

4. **Multi-step Tool Use** - Complex scenarios
   - Execute → analyze → re-execute
   - Search documentation → generate code → test
   - Error recovery sequences

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

## Training Strategy

### Phase 1: Baseline Evaluation

- Evaluate pre-trained models on synthetic dataset
- Establish baseline metrics (no fine-tuning)
- Identify common failure patterns

### Phase 2: Supervised Fine-Tuning (LoRA)

- Create preference dataset: good vs. bad tool choices
- Fine-tune with LoRA adapters (rank=16, alpha=32) # Need to learn more but for now stick to this
- Low learning rate (1e-4 to 5e-5)
- Small number of epochs (3-5)

### Phase 3: Preference Optimization (Optional)

- **DPO**: Simpler, more stable than PPO
  - Create preference pairs from model outputs
  - Train to prefer correct tool choices
- **PPO**: Full RL approach
  - Reward based on correctness + efficiency
  - Use TRL library for implementation

### Phase 4: Comparative Analysis

- Compare baseline vs. LoRA vs. DPO vs. PPO
- Analyze trade-offs in accuracy, speed, resource usage
- Identify optimal training approach for tool-use learning

## Evaluation Metrics

### Primary Metrics

1. **Correctness Rate**: % of tasks solved correctly
2. **Tool Selection Accuracy**: % of optimal tool choices
3. **Inference Latency**: Time to generate response
4. **Execution Efficiency**: Unnecessary tool calls avoided

### Secondary Metrics

- Token efficiency (output length)
- Error recovery rate
- Multi-step success rate
- Memory usage per inference

## Development Workflow

### Current State

- ✅ Virtual environment set up
- ✅ Basic model wrapper (Qwen2.5-Coder)
- ✅ Metrics recording infrastructure
- ✅ Quantization support (4-bit/8-bit)
- ✅ **Docker-based code execution sandbox (COMPLETED)**
  - Paranoid-safe isolation with network blocking
  - Resource limits: 256MB RAM, 0.5 CPU cores, 5s timeout
  - Read-only filesystem (except /tmp)
  - Comprehensive test suite with 8 security checks
  - Files: `src/sandbox/docker_executor.py`, `demo_sandbox.py`

### Next Steps

1. ~~Design and implement code execution sandbox~~ ✅ **COMPLETED**
2. Create tool registry and selection system
   - Define tool interface and registry
   - Implement: code_executor (wraps DockerExecutor), symbolic_math, ... Still need to further scope what the test dataset is going to be ... ideally go through research papers to find a dataset
   - Add tool-use logging and metrics
3. Build agent harness for multiple tool selection
   - Decision logic for tool vs. symbolic reasoning
   - Prompt templates for tool-aware generation
   - Integration with model wrapper
4. Generate synthetic dataset focused on 3 task types:
   - Or find a good research data set that accomplishes my goals
5. Implement baseline evaluation pipeline
6. Add LoRA fine-tuning capability
7. Create visualization and reporting tools
8. Run experiments and analyze results

## File Structure

```
training-agentic-behavior/
├── src/
│   ├── models/          # Model wrappers and loading
│   ├── sandbox/         # Safe code execution
│   ├── tools/           # Tool definitions and registry
│   ├── training/        # Fine-tuning pipelines
│   ├── data/            # Dataset generation
│   ├── evaluation/      # Metrics and evaluation
│   └── utils/           # Shared utilities
├── data/
│   ├── raw/             # Original data sources
│   ├── synthetic/       # Generated datasets
│   └── processed/       # Formatted for training
├── models/
│   ├── base/            # Base model checkpoints
│   ├── lora/            # LoRA adapters
│   └── dpo/             # DPO-trained models
├── results/
│   ├── metrics/         # Performance metrics
│   ├── plots/           # Visualizations
│   └── logs/            # Training and eval logs
├── notebooks/           # Jupyter analysis notebooks
├── experiments/         # Experiment configs and scripts
└── tests/              # Unit and integration tests
```

## Dependencies

### Core ML Stack

- **torch**: PyTorch for model operations
- **transformers**: HuggingFace model loading
- **peft**: LoRA implementation
- **trl**: Reinforcement learning (PPO/DPO)
- **bitsandbytes**: Quantization support
- **accelerate**: Multi-GPU training

### Execution & Safety

- **RestrictedPython**: Code sandboxing
- **docker**: Container-based isolation
- **timeout-decorator**: Execution timeouts

### Data & Evaluation

- **datasets**: HuggingFace datasets library
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Metrics and evaluation

### Visualization & Logging

- **matplotlib**, **seaborn**: Plotting
- **wandb**: Experiment tracking (optional)
- **tensorboard**: Training visualization

### Utilities

- **psutil**: System monitoring
- **rich**: Pretty terminal output
- **pytest**: Testing framework

## Experimental Design

### Variables to Study

- **Model size**: 1.3B vs. 3B vs. 7B parameters
- **Training method**: None (baseline) vs. LoRA vs. DPO vs. PPO
- **Quantization**: Full precision vs. 8-bit vs. 4-bit
- **Prompt format**: Direct vs. chain-of-thought vs. tool-description
- **Dataset difficulty**: Easy vs. intermediate vs. hard

### Expected Outcomes

- Models can learn to improve tool selection with relatively small datasets
- LoRA fine-tuning provides good baseline improvement
- DPO may outperform PPO in stability and sample efficiency
- Trade-offs exist between correctness and latency

## Reproducibility

All experiments should log:

- Random seeds (Python, NumPy, PyTorch)
- Model configurations and hyperparameters
- Training data versions and splits
- Hardware specifications (GPU/CPU, memory)
- Software versions (PyTorch, transformers, etc.)
- Timestamp and git commit hash

## Research Output

### Deliverables

1. **Code**: Complete framework on GitHub
2. **Data**: Synthetic dataset with ground truth labels
3. **Models**: Trained LoRA adapters and checkpoints
4. **Results**: Metrics, plots, and comparative analysis
5. **Documentation**: Setup guide, API docs, experiment logs
6. **Report**: Findings on tool-use learning (notebook or paper)

## Notes for Claude

- This is an experimental research project, not production code
- Prioritize clarity and reproducibility over optimization
- Add extensive logging and metrics collection
- Keep experiments modular and configurable
- Document assumptions and design decisions
- Focus on small models that can run on 8GB RAM systems
- Use quantization by default to keep experiments accessible
