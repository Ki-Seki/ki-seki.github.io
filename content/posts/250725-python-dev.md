---
date: '2025-07-25T23:17:51+08:00'
title: 'Python Package Design'
summary: 本文会介绍下Python Package开发中的一些设计思路，主要是我在开发`dowhen`和`MemOS`时的一些思考。
tags: ["python", "dowhen", "MemOS"]
---

本文推荐具有AI研究背景的开发者阅读，可能对开发更具有影响力的框架有帮助。

本文介绍的时候会多以dowhen和MemOS为例，这是笔者深度参与过的相对比较有热度的项目。可以通过 [Appendix](#appendix) 来了解更多。

## API design

### API是什么？

API 是包的交付物，核心产出产品。

- 只有包的 API 是暴漏给用户去使用的；
- 其他非API的代码都无需向用户解释，用户也不应该在使用API时去阅读他们（仅开发者在协作时需要去阅读）。

### API 应该保持稳定且简洁

代码应该被不断重构，以适应新情况。但是至少在每个major version（eg. v0.\*.\*, v1.\*.\*都是major versions）内，API应该保持稳定。这也意味着至少在*开发一个包之前*，就应该想好包要提供什么样的核心功能！

如果核心功能是逻辑上完备的，那么API就应该是稳定的，同时也能让API保持简洁。例如，增/删/改/查就是一个典型的逻辑上完备的功能集合；再比如Einstein Operation就是一个逻辑上完备的数学运算集合，一个示例的实现einops [^einops]只提供三个核心的APIs！良好的API设计需要我们从逻辑学，数学来找到启发。

API数量少还有其他的好处。一来方便用户记忆，毕竟数量少；同时也方便开发者撰写文档，因为很明确要对什么函数/类撰写详细的说明，示例讲解，使用场景等；同时也方便我们来编写tests，因为这些API的接口稳定性应当更高。

### 如何实现稳定且简洁的API？

**通过参数多态来实现简洁API。** `dowhen` 仅设计了 `['bp', 'clear_all', 'do', 'get_source_hash', 'goto', 'when', 'DISABLE']` 七个函数暴露给用户去用，其中 `bp`、`goto` 、 `do` 和 `when` 是最核心的。用户使用什么非常明确。`dowhen` 主要是用参数多态来完成这个目标的。比如制作trigger的when函数，其函数签名是这样的：

```python
    def when(
        cls,
        entity: CodeType | FunctionType | MethodType | ModuleType | type | None,
        *identifiers: IdentifierType | tuple[IdentifierType, ...],
        condition: str | Callable[..., bool | Any] | None = None,
        source_hash: str | None = None,
    ):
```

其中，`IdentifierType = int | str | re.Pattern | Literal["<start>", "<return>"] | None`. identifiers这个参数由于可以是可变参数，因此隐含的还支持逻辑与和逻辑或的关系。所以你会看到这个函数支持的功能范围实际上是非常庞大的。[Appendix](#dowhentrigger) 具体介绍了这个函数的实现。

**通过工厂模式来实现简洁API。** `transformers` 也有简单易记的API，不过他使用更多的工厂模式思路，比如 `AutoModelForCausalLM`、`AutoModelForSequenceClassification` 等等，甚至抽象工厂“pipeline”。只不过这里的区别是，transformers中用户API的参数并没有特别复杂，你不需要特别担心参数之间的相互作用。所以总体上也是易于用户使用和记忆的。

### 不可能简洁的情况

有些时候，事情无法如我们所愿。如果你去看 `Megatron-LM` 的API，你会发现它的API就非常复杂，`llamafactory`的API也很复杂，还有 `MemOS`。这些库普遍具有实验性质，接口也不可能稳定，因此使用这些库的时候“跑通”是使用中很大一环，API也很难做到简洁。如果API无法简化，那用户一定得去通过某种渠道来了解API。所以我们的重点就在于如何优化这个渠道。

- 对于`Megatron-LM`, 他本身expect用户git clone下来整个包。然后其提供了各类的training templates bash scripts，方便用户直接修改。如果还有定制化的需要，直接去阅读[megatron/training/arguments.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py) 这个把所有参数聚集起来的地方。
- 对于`llamafactory`，这个也是参数众多，它提供的[文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/sft.html#)对每种训练范式都提供详细的参数介绍。另外一个办法是，他直接启用一个gradio的界面，直观的把参数介绍和参数设置的位置统一了起来。
- 对于`MemOS`，这里依赖三个渠道，详细的examples示例，各种cookbooks，用Pydantic来约束用户的input，让报错时候用户知道怎么改。

## Dependency Management

Python 不如 Rust 或者 NodeJS 那样有官方提供的包管理工具。不过随着Python组织官方对pyproject.toml的逐步规范化 [^pep735] [^pep751]，Python的包管理工具正在逐步走向成熟。

在包管理中，依赖管理实际上是最重要的部分之一，我们在这里也只讨论与依赖有关的事情。

### 管理工具

我们有很多选择，比如setuptools，poetry等。但我的个人建议是使用 uv，这是打包的未来发展方向。我认为他有三大好处:

- 底层是用Rust开发的，速度快 [^uv_rust]。
- 支持Python官方的pyproject.toml规范，许多schema紧跟Python PEP规范。
- 支持对pytorch生态的integration [^uv_torch]。

### 三种不同类型的依赖

正如 uv 官方文档中所讲的，总共有三种不同类型的依赖[^uv_deps]：

> - `project.dependencies`: Published dependencies.
> - `project.optional-dependencies`: Published optional dependencies, or "extras".
> - `dependency-groups`: Local dependencies for development.

**Published dependencies** 管理 pip install xxx 时安装的依赖。这里应该放置所有核心依赖，即离开这里的任何一个依赖，包的几乎任何代码都无法运行了。不过也不是完全一定，对于torch这类占空间巨大的包，而且又看平台安装的包，我们可以引导用户自行pip install，参考 [# Lazy Imports and Guided Installation](#lazy-imports-and-guided-installation)。

**Published optional dependencies** 管理 pip install xxx[extra] 时安装的依赖。这里应该放置所有可选依赖，即离开这里的任何一个依赖，包的核心代码仍然可以运行，但一个大的功能块无法运行,这就需要我们管理不同的功能块/依赖组，比如，`MemOS`，是按照支持记忆的类型来划分依赖组的，比如 `MemoryOS[tree-mem]`, `MemoryOS[mem-reader]` 等等[^memos_install]；`lm-eval-harness`则是按照支持的benchmarks不同来划分依赖组的，比如 `lm-eval[ifeval]`、`lm-eval[math]` 等等[^lm_eval_install]。

**Local dependencies for development** 管理开发时的依赖。这里应该放置所有开发时需要的依赖，比如测试框架，代码格式化工具，文档生成工具等。TODO

### Lazy Imports and Guided Installation

如果IDE提示错误，我们可以考虑使用延迟导入来减少对某些依赖的即时需求。同时，我们也可以提供引导安装的方式，帮助用户更方便地安装所需的依赖。

### 依赖组解析

## CI/CD Design

## Common Utilities

包里src/下经常有些常见的工具，你可能想要预先了解下，这样等遇到有需要时，就可以知道有这样的东西。

- **mocks/fixtures**:
- **customized logger**:
- **error handling**:
- **customized types**:
- **configs**:
- **constants/settings**:
- **deprecation management**:
- **dependency management**:

## Citation

```bibtex

```

## Appendix

### 了解 `dowhen`

dowhen 是一个instrumentation的工具，可以用来做测试、调试、软件安全分析等。Python没有内置的instrumentation工具。Python的core dev [@gaogaotiantian](https://github.com/gaogaotiantian) 利用 Python3.12引入的新特性 sys.monitoring 开发了这个工具。

dowhen的核心API就两个，一是负责执行什么的 callback/do，二是负责什么时候执行的trigger/when。为了把do和when更好的结合起来，比如提供context manager，提供trigger时机的判断等，因此dowhen的底层是一个handler模块；为了使用系统提供的sys.monitoring模块，更底层是一个instrumenter模块。

dowhen的基本使用方法可以参考其官方文档 [^dowhen]。本文会介绍下dowhen的API设计和实现思路。

#### `dowhen.trigger`

{{<details>}}

```python
@classmethod
def when(
    cls,
    entity: CodeType | FunctionType | MethodType | ModuleType | type | None,
    *identifiers: IdentifierType | tuple[IdentifierType, ...],
    condition: str | Callable[..., bool | Any] | None = None,
    source_hash: str | None = None,
):
    # 1. 判定 condition 是否是语法可执行的，类型是否正确

    # 2. 根据 source_hash，判定运行时 entity 是否相对于用户提供时发生变化

    events = []

    # 3. breadth-first 展开 entity 中的 code objects
    code_objects = cls._get_code_from_entity(entity)

    # 4. 根据各类条件定义 trigger events

    # 4.1. 如果没有传 identifiers 参数，all-line matching events
    if not identifiers:
        for code in code_objects:
            events.append(_Event(code, "line", {"line_number": None}))

    # 4.2. 传了 identifiers 参数，普通 events
    else:

        # 4.2.1. 首先要根据没有展开的entity把相对行号转换为绝对行号，因为展开后再计算就可能导致绝对行号在不同code objects里重复出现
        identifiers = cls.unify_identifiers(entity, *identifiers)

        # 4.2.2. 对于每个 identifier × code object，创建对应的事件，其实也可以用itertools.product来简化
        for identifier in identifiers:

            # 4.2.2.1. "<start>" × code object
            if identifier == "<start>":
                for code in code_objects:
                    events.append(_Event(code, "start", None))

            # 4.2.2.2. "<return>" × code object
            elif identifier == "<return>":
                for code in code_objects:
                    events.append(_Event(code, "return", None))
            else:
                for code in code_objects:

                    # 4.2.2.3. 其他标识符 × None
                    if code is None:
                        # Global event, entity is None
                        events.append(
                            _Event(
                                None,
                                "line",
                                {"line_number": None, "identifier": identifier},
                            )
                        )

                    # 4.2.2.4. 其他标识符 × code object
                    else:

                        # 4.2.2.4.1 真正把 identifier 解析为行号，这里要处理各类复杂情形：
                        # 例如，首行匹配问题，comiple得来的code object问题，
                        # identifier的逻辑与和逻辑或关系问题，code object 嵌套问题，
                        # comment等不在co_lines中的语句的trigger问题等
                        line_numbers = get_line_numbers(code, identifier)

                        # 这里得到的 c 是 depth-first 展开的 code objects，确保 trigger 位置正确
                        for c, numbers in line_numbers.items():
                            for number in numbers:
                                events.append(
                                    _Event(c, "line", {"line_number": number})
                                )

    if not events:
        raise ValueError(
            "Could not set any event based on the entity and identifiers."
        )

    # 5. 返回 Trigger 实例
    return cls(events, condition=condition, is_global=entity is None)
```

{{</details>}}

#### `dowhen.callback`

{{<details>}}

```python
被instrumented的代码行
↓
sys.monitoring  # 监听抽象事件以及绑定回调
↓
instrumenter.py::Instrumenter().*_callback()  # sys.monitoring 绑定的回调
↓
instrumenter.py::Instrumenter()._process_handlers()  # 回调高层包装，添加 sys.monitoring.DISABLE 功能
↓
handler.py::EventHandler().__call__()  # 回调中层包装，添加 event 执行时机判定，has_event 和 should_fire 逻辑
↓
callback.py::Callback().__call__()  # 回调低层包装，添加到 call_code/call_goto/call_bp 的判定
↓
callback.py::Callback.call_*()  # 回调执行
↓
用户定义代码  # 如果是 call_code，则执行用户定义的代码
```

{{</details>}}

## References

[^pep735]: Rosen, Stephen. PEP 735 – Dependency Groups in pyproject.toml.. Python Enhancement Proposals, 20 Nov. 2023, https://peps.python.org/pep-0735/.

[^pep751]: Cannon, Brett. PEP 751 – A file format to record Python dependencies for installation reproducibility. Python Enhancement Proposals, 24 July 2024, https://peps.python.org/pep-0751/.

[^uv_rust]: https://docs.astral.sh/uv/

[^uv_torch]: https://docs.astral.sh/uv/guides/integration/pytorch/

[^uv_deps]: https://docs.astral.sh/uv/concepts/projects/dependencies/

[^dowhen]: https://dowhen.readthedocs.io/en/latest/

[^einops]: https://einops.rocks/

[^memos_install]: https://memos-docs.openmem.net/getting_started/installation

[^lm_eval_install]: https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#optional-extras
