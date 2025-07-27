---
date: '2025-07-25T23:17:51+08:00'
title: 'Python Package Design'
summary: 本文会介绍下Python Package开发中的一些设计思路，主要是我在开发`dowhen`和`MemOS`时的一些思考。
tags: ["python", "dowhen", "MemOS"]
---

本文介绍的时候会多以dowhen和MemOS为例，可以通过 [Appendix](#appendix) 来了解更多。

## API design

**API 是包的交付物，核心产出产品。** 只有包的 API 是暴漏给用户去使用的；其他非API的代码都无需向用户解释，用户也不应该在使用API时去阅读他们（仅开发者在协作时需要去阅读）。

**接口稳定性。** 代码应该被不断重构，以适应新情况。但是至少在每个major version（eg. v0.*.*, v1.*.*都是major versions）内，API应该保持稳定。这也意味着至少在*开发一个包之前*，就应该想好包要提供什么样的核心功能！

**应该提供少量的包的API，即放置少量的类或者函数在顶层__all__中。** 这样做的好处是方便用户记忆，毕竟数量少；同时也方便开发者撰写文档，因为很明确要对什么函数/类撰写详细的说明，示例讲解，使用场景等；同时也方便我们来编写tests，因为这些API的接口稳定性应当更高。

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

其中，`IdentifierType = int | str | re.Pattern | Literal["<start>", "<return>"] | None`. identifiers这个参数隐含的还支持逻辑与和逻辑或的关系。所以你会看到这个函数支持的功能范围实际上是非常庞大的。

**通过工厂模式来实现简洁API。** `transformers` 也有简单易记的API，不过他使用更多的工厂模式思路，比如 `AutoModelForCausalLM`、`AutoModelForSequenceClassification` 等等，甚至抽象工厂“pipeline”。只不过这里的区别是，transformers中用户API的参数并没有特别复杂，你不需要特别担心参数之间的相互作用。所以总体上也是易于用户使用和记忆的。

**不可能简洁的情况。** 有些时候，事情无法如我们所愿。如果你去看 `Megatron-LM` 的API，你会发现它的API就非常复杂，`llamafactory`的API也很复杂，还有 `MemOS`。这些库普遍具有实验性质，接口也不可能稳定，因此使用这些库的时候“跑通”是使用中很大一环，API也很难做到简洁。如果API无法简化，那用户一定得去通过某种渠道来了解API。所以我们的重点就在于如何优化这个渠道。

- 对于`Megatron-LM`, 他本身expect用户git clone下来整个包。然后其提供了各类的training templates bash scripts，方便用户直接修改。如果还有定制化的需要，直接去阅读[megatron/training/arguments.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py) 这个把所有参数聚集起来的地方。
- 对于`llamafactory`，这个也是参数众多，它提供的[文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/sft.html#)对每种训练范式都提供详细的参数介绍。另外一个办法是，他直接启用一个gradio的界面，直观的把参数介绍和参数设置的位置统一了起来。
- 对于`MemOS`，这里依赖三个渠道，详细的examples示例，各种cookbooks，用Pydantic来约束用户的input，让报错时候用户知道怎么改。

## Dependency Management

务必使用 uv，不需要考虑其他选项。

可选依赖组。

开发依赖组。

延迟导入。

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

## References

## Appendix

{{<details>}}

### `dowhen` 的介绍与讲解

`dowhen` 的 `trigger` 执行流

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

`dowhen` 的 `callback` 执行流

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
