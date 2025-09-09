## Fill-in-the-middle

相关论文：
- https://arxiv.org/abs/2506.00204
- https://arxiv.org/abs/2207.14255
- https://arxiv.org/abs/2501.08648
- https://openreview.net/forum?id=jKYyFbH8ap

## 约束生成

### 狭义约束生成工具

- https://github.com/guidance-ai/guidance: dependents 大多是套壳的约束生成工具
- https://github.com/guidance-ai/llguidance
- https://github.com/google/langextract
- https://github.com/outlines-dev/outlines：有提供自己的backend
- https://github.com/instructor-ai/instructor
- https://github.com/PrefectHQ/marvin
- https://github.com/eth-sri/lmql
- https://github.com/mlc-ai/xgrammar：效率很高，众多库的核心依赖
- https://github.com/otriscon/llm-structured-output

### 下游dependent

- 大多是套壳的约束生成工具
  - https://github.com/AlbanPerli/Noema-Declarative-AI
  - https://github.com/microsoft/promptbase
  - https://github.com/amaiya/onprem
  - https://github.com/milvus-io/bootcamp
- 推理引擎
  - TensorRT-LLM, vLLM, sglang
- agent系统：camel-ai
- 数据合成：distilabel

### 广义约束生成工具

- https://github.com/ggerganov/llama.cpp: GBNF
- https://github.com/stanfordnlp/dspy
- https://github.com/sgl-project/sglang
- https://github.com/pydantic/pydantic-ai
- https://github.com/sgl-project/sgl-learning-materials

### 一些不错的博客

- ✅ https://www.aidancooper.co.uk/constrained-decoding/
- ✅ https://blog.dottxt.co/coalescence.html?ref=aidancooper.co.uk
- ✅ https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38
- ✅ https://lmsys.org/blog/2024-02-05-compressed-fsm/?ref=aidancooper.co.uk
- ✅ https://blog.dottxt.co/performance-gsm8k.html
  - ❗ prompt consistency
  - ❗ thought control
- https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
  - GBNF 简明教程。是关于symbol和production rule的。

### 重要概念

- https://en.wikipedia.org/wiki/Context-free_grammar
- https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form
- https://en.wikipedia.org/wiki/Formal_grammar

### 重要论文

- JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models 
  - https://arxiv.org/abs/2501.10868
  - https://guidance-ai.github.io/benchmarks/
- Efficient Guided Generation for Large Language Models https://arxiv.org/abs/2307.09702?ref=aidancooper.co.uk
- SGLang: Efficient Execution of Structured Language Model Programs https://arxiv.org/abs/2312.07104
- Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting https://openreview.net/forum?id=RIu5lyNXjT
- XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models https://arxiv.org/abs/2411.15100