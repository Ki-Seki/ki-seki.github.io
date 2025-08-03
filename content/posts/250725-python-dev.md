---
date: '2025-07-25T23:17:51+08:00'
title: 'Python Package Design: API, Dependency and Code Structure'
tags: ["python", "package", "API", "dependency", "structure"]
---

AI发展愈发猛烈，这也让适合AI领域快速原型验证的语言，Python变得火热。因此，适时制作一个良好的Python包能够让自己的工作更可能落地也更具有影响力。

笔者从2024年10月开始了解Python Package的开发，在这之后，尝试过将自己的工作 {{< github "IAAR-Shanghai/UHGEval" >}} 发布为Python包，还参与过 {{< github "MemTensor/MemOS" >}}包的前期架构设计，也深度学习并参与了 {{< github "gaogaotiantian/dowhen" >}} 的开发。这些经历让我对Python包的设计有了一些思考，因此记录在本文。

此外，在我所参与过的工作当中，{{< github "gaogaotiantian/dowhen" >}} 的设计尤其精致，也足够Pythonnic，非常鼓励大家去深入了解，我也在 [Appendix](#python-package-dowhen) 中为该package进行了扩展介绍。

## Related Work

对于Python包的构建和设计，网络上已经有许多文章/教程。比如说，

- (必读) *Packaging Python Projects*，Python Packaging Authority (PyPa) 提供的官方教程，简明的介绍了Python包的构建和发布流程 [^pypa_packaging].
- (必读) *Designing Pythonic library APIs*，一篇介绍如何设计Pythonic的API的文章 [^pythonic_api]。
- *Python Packaging Best Practices*, 一篇介绍Packaging原理的文章 [^packaging_principles],包括sdist，wheel，前端/后端工具的简介。
- *Structuring Your Project*, 一个稍微落后的介绍Python包代码结构的文章 [^structuring_your_project]。

这些文章不是内容落后 [^structuring_your_project]，就是缺乏设计哲学的传递 [^pypa_packaging] [^packaging_principles]，又或者是不够全面[^pypa_packaging] [^pythonic_api]，本文则会试图弥补这些缺憾。

## API Design

API 是包的交付物，核心产出产品。

- 只有包的 API 是暴漏给用户去使用的；
- 其他非API的代码都无需向用户解释，用户也不应该在使用API时去阅读他们（仅开发者在协作时需要去阅读）。

所以，你能看出来API是相当重要的。Ben Hoyt在他的文章 [^pythonic_api] 中提及了许多实用的建议，而且他的文字充满趣味，强烈建议去阅读下，copy 他总结的takeaways在这里：

> - Good API design is very important to users.
> - When creating a library, start with a good base and iterate.
> - Try to follow PEP 8 and grok PEP 20. This is the way.
> - The standard library isn’t always the best example to follow.
> - Expose a clean API; file structure is an implementation detail.
> - Flat is better than nested.
> - Design your library to be used as `import lib ... lib.Thing()` rather than `from lib import LibThing ... LibThing()`.
> - Avoid global configuration; use good defaults and let the user override them.
> - Avoid global state; use a class instead.
> - Names should be as short as they can be while still being clear.
> - Function names should be verbs and classes nouns, but don’t get hung up on this.
> - Being `_private` is fine; `__extra_privacy` is unnecessary.
> - If an error occurs, raise a custom exception; use built-in exceptions if appropriate.
> - Only break backwards compatibility if you’re overhauling your API.
> - Keyword arguments and dynamic typing are great for backwards compatibility.
> - Use type annotations at least for your public API; your users will thank you.
> - Use `@dataclass` for classes which are (mostly) data.
> - Python’s expressiveness is boundless; don’t use too much of it!

除了这些建议，本文简单补充如何实现稳定且简洁的API。

### API 应该保持稳定且简洁

代码应该被不断重构，以适应新情况。但是至少在每个major version（eg. v0.\*.\*, v1.\*.\*都是major versions）内，API应该保持稳定。这也意味着至少在*开发一个包之前*，就应该想好包要提供什么样的核心功能！

如果核心功能是逻辑上完备的，那么API就应该是稳定的，同时也能让API保持简洁。例如，增/删/改/查就是一个典型的逻辑上完备的功能集合；再比如Einstein Operation就是一个逻辑上完备的数学运算集合，{{< github "arogozhnikov/einops">}} 只提供三个核心的APIs[^einops]！良好的API设计需要我们从逻辑学，数学来找到启发。

{{<media
src="https://user-images.githubusercontent.com/6318811/177030658-66f0eb5d-e136-44d8-99c9-86ae298ead5b.mp4"
caption="Video 1: Einstein Operation Introduction"
>}}

API数量少还有其他的好处。一来方便用户记忆，毕竟数量少；同时也方便开发者撰写文档，因为很明确要对什么函数/类撰写详细的说明，示例讲解，使用场景等；同时也方便我们来编写tests，因为这些API的接口稳定性应当更高。

### 如何实现稳定且简洁的API？

下面列举两种我认为非常Pythonic的实现方式，以及提供当无法避免复杂性时的做法。

**通过参数多态来实现简洁API。** {{< github "gaogaotiantian/dowhen" >}} 仅设计了 `['bp', 'clear_all', 'do', 'get_source_hash', 'goto', 'when', 'DISABLE']` 七个函数暴露给用户去用，其中 `bp`、`goto` 、 `do` 和 `when` 是最核心的。用户使用什么非常明确。`dowhen` 主要是用参数多态来完成这个目标的。比如制作trigger的when函数，其函数签名是这样的：

```python
    def when(
        cls,
        entity: CodeType | FunctionType | MethodType | ModuleType | type | None,
        *identifiers: IdentifierType | tuple[IdentifierType, ...],
        condition: str | Callable[..., bool | Any] | None = None,
        source_hash: str | None = None,
    ):
```

其中，`IdentifierType = int | str | re.Pattern | Literal["<start>", "<return>"] | None`. identifiers这个参数由于可以是可变参数，因此隐含的还支持逻辑与和逻辑或的关系。所以你会看到这个函数支持的功能范围实际上是非常庞大的。[Appendix](#python-package-dowhen) 具体介绍了这个函数的实现。

**通过工厂模式来实现简洁API。** {{< github "huggingface/transformers" >}} 也有简单易记的API，不过他使用更多的工厂模式思路，比如 `AutoModelForCausalLM`、`AutoModelForSequenceClassification` 等等，甚至抽象工厂“pipeline”。只不过这里的区别是，transformers中用户API的参数并没有特别复杂，你不需要特别担心参数之间的相互作用。所以总体上也是易于用户使用和记忆的。

### 不可能简洁的情况

有些时候，事情无法如我们所愿。如果你去看 {{< github "NVIDIA/Megatron-LM" >}} 的API，你会发现它的API就非常复杂，{{< github "hiyouga/LLaMA-Factory" >}}的API也很复杂，还有 {{< github "MemTensor/MemOS" >}}。这些库普遍具有实验性质，接口也不可能稳定，因此使用这些库的时候“跑通”是使用中很大一环，API也很难做到简洁。如果API无法简化，那用户一定得去通过某种渠道来了解API。所以我们的重点就在于如何优化这个渠道。

- 对于{{< github "NVIDIA/Megatron-LM" >}}, 他本身expect用户git clone下来整个包。然后其提供了各类的training templates bash scripts，方便用户直接修改。如果还有定制化的需要，直接去阅读[megatron/training/arguments.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py) 这个把所有参数聚集起来的地方。
- 对于{{< github "hiyouga/LLaMA-Factory" >}}，这个也是参数众多，它提供的[文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/sft.html#)对每种训练范式都提供详细的参数介绍。另外一个办法是，他直接启用一个gradio的界面，直观的把参数介绍和参数设置的位置统一了起来。
- 对于{{< github "MemTensor/MemOS" >}}，这里依赖三个渠道，详细的examples示例，各种cookbooks，用Pydantic来约束用户的input，让报错时候用户知道怎么改。

## Dependency Management

Python 不如 Rust 或者 NodeJS 那样有官方提供的包管理工具。不过随着Python组织官方对`pyproject.toml`的逐步规范化 [^pep735] [^pep751]，Python的包管理工具正在逐步走向成熟。

在包管理中，依赖管理实际上是最重要的部分之一，我们在这里也只讨论与依赖有关的事情。

### 管理工具

我们有很多选择，比如{{< github "pypa/setuptools" >}}，{{< github "python-poetry/poetry" >}}等。但我的个人建议是使用{{< github "astral-sh/uv" >}}，这是打包的未来发展方向。我认为他有三大好处:

- 底层是用Rust开发的，速度快 [^uv_rust]。
- 支持Python官方的pyproject.toml规范，许多schema紧跟Python PEP规范。
- 支持对pytorch生态的integration [^uv_torch]。

### 三种不同类型的依赖

正如 uv 官方文档中所讲的，总共有三种不同类型的依赖[^uv_deps]：

> - `project.dependencies`: Published dependencies.
> - `project.optional-dependencies`: Published optional dependencies, or "extras".
> - `dependency-groups`: Local dependencies for development.

**Published dependencies** 管理 pip install xxx 时安装的依赖。这里应该放置所有核心依赖，即离开这里的任何一个依赖，包的几乎任何代码都无法运行了。不过也不是完全一定，对于torch这类占空间巨大的包，而且平台不同安装包也不同，我们就可以引导用户自行install，参考 [# Lazy Imports and Guided Installation](#lazy-imports-and-guided-installation)。

**Published optional dependencies** 管理 pip install xxx[extra] 时安装的依赖。这里应该放置所有可选依赖，即离开这里的任何一个依赖，包的核心代码仍然可以运行，但一个大的功能块无法运行,这就需要我们管理不同的功能块/依赖组，比如，`MemOS`，是按照支持记忆的类型来划分依赖组的，比如 `MemoryOS[tree-mem]`, `MemoryOS[mem-reader]` 等等[^memos_install]；`lm-eval-harness`则是按照支持的benchmarks不同来划分依赖组的，比如 `lm-eval[ifeval]`、`lm-eval[math]` 等等[^lm_eval_install]。

**Local dependencies for development** 管理开发时的依赖。这里应该放置所有开发时需要的依赖，比如测试框架，代码格式化工具，文档生成工具等。他们是在开发的不同阶段会用到的。比如测试相关依赖(`pytest`, `coverage`)，代码格式化相关依赖（`mypy`, `ruff`等）可能仅在CI/CD中会用到；文档生成相关依赖（`sphinx`, `mkdocs`等），可能仅在发布文档时会用到。所以并不是所有的开发依赖都需要在本地安装的，因此设计分组是更合理的。

### 指定依赖的范围

一般来说我们可以假设我们引入的依赖的最新版本兼容性是最好的，因此我们可以选择其最新版本，这个是版本的下限。同时我们还要为依赖选择个上限，一般应当是首位非零的版本号加1，比如说，`>=1.0.0, <2.0.0`，`>=0.2.0, <0.3.0` 等等。

这样做，是因为首位非0的版本一般代表着一个major version，只要major version不变，我们就可以假设依赖的包的API没有发生break change。

### Lazy Imports and Guided Installation

让我们用最令人头大的例子来说明问题，`torch` 这个包在各种开源仓库中都是很复杂的管理对象，很难把他写到dependency中去，因为他的安装依赖平台相关，且体积巨大。

又或者，你或许经常遇到{{< github "huggingface/transformers" >}} 提示，比如`sentencepiece`、`einops`等包没有安装，这些可能是某些特定的模型自己的依赖项。

像这些特殊的依赖，我们不希望用户在pip install的时候就安装，我们希望在runtime时，检查有没有安装，如果没有的话，给出提示，引导用户安装即可。下面这个 decorator 可以做到这一点：

```python
def require_python_package(
    import_name: str, install_command: str | None = None, install_link: str | None = None
):
    """Check if a package is available and provide installation hints on import failure.

    Args:
        import_name (str): The top-level importable module name a package provides.
        install_command (str, optional): Installation command.
        install_link (str, optional): URL link to installation guide.

    Returns:
        Callable: A decorator function that wraps the target function with package availability check.

    Raises:
        ImportError: When the specified package is not available, with installation
            instructions included in the error message.

    Example:
        >>> @require_python_package(
        ...     import_name='faiss',
        ...     install_command='pip install faiss-cpu',
        ...     install_link='https://github.com/facebookresearch/faiss/blob/main/INSTALL.md'
        ... )
        ... def create_faiss_index():
        ...     from faiss import IndexFlatL2  # Actual import in function
        ...     return IndexFlatL2(128)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                importlib.import_module(import_name)
            except ImportError:
                error_msg = f"Missing required module - '{import_name}'\n"
                error_msg += f"💡 Install command: {install_command}\n" if install_command else ""
                error_msg += f"💡 Install guide:   {install_link}\n" if install_link else ""

                raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper

    return decorator
```

被这个decorator装饰的函数，在运行时会检查是否安装了指定的包，如果没有安装，则会抛出一个`ImportError`，并给出安装命令和安装链接。同时，我们注意应该lazily import这个包，不能把这个包的import放在文件开头，否则可能会出现过多的报错提示。

最后，还需要注意的是，对于开发者来说，我们希望开发的时候我们的包的时候，不用自己等到runtime的时候才安装，那我们就应该把这些依赖放入兜底的 **Published optional dependencies**，即`all` extras group里，在安装 `my-package[all]` 的时候，可以一键安装所有开发时的依赖。

## Code Structure

这里列举一个现代的Python Package对应的GitHub仓库中的代码究竟都放些什么，怎么放。

### 仓库根目录

- **README.md**: 包的介绍，使用方法，安装方法等。
- **LICENSE**: 包的许可证。
- **pyproject.toml**: 包的元数据，依赖，构建工具，相关tools的设置（mypy，uv，pytest等工具）。
- **Makefile**: 包的构建脚本，通常是用来构建包的源代码和文档。里面通常会放些常用的命令，可以一键执行`make test`、`make build`、`make clean` 等等。
- **.gitignore**: Git忽略的文件列表, 比如说编译生成的文件，测试生成的文件等,推荐使用 {{< github "github/gitignore" >}} 中的 Python.gitignore。
- **.pre-commit-config.yaml**: pre-commit hooks的配置文件，这个可以检查一些代码风格问题，确保代码在提交前符合规范。
- **src/my_package/** or **my_package/**: 包的源代码目录，用户pip install的就是这里的内容。添加 src/ 这一层的目录有多个好处，包括避免与其他已安装的包名冲突，导致import混淆；还能够实现单个仓库多个Python包的管理，比如`src/my_package/` 和 `src/my_other_package/`可以同时放在一个GitHub仓库中。
- **tests/**: 包的测试代码目录，推荐用 pytest 来编写测试，因为更简单，也更Pythonic。
- **docs/**: 包的文档目录。
- **.github/**: GitHub 相关配置目录，包含工作流、issue 模板、issue设置、Pr模板等。可以参考 {{< github "MemTensor/MemOS" >}} 中的配置。
- etc.

### 包目录

包文件夹 **src/my_package/** 下常有些这些代码，{{< github "MemTensor/MemOS" >}}均存在一些示例：

- **\_\_init\_\_.py**: 包的初始化文件，把包的核心API暴露给用户。
- **api**: 包的API代码目录，存放包的核心http API代码。
- **cli**: 命令行接口代码目录，存放包的命令行接口代码。
- **configs**: 配置文件目录，存放各种标准化的配置protocols，比如用Pydantic来定义的配置类。
- **log.py**: 自定义的日志记录handler，通常是基于Python的logging模块。
- **exceptions.py**: 错误处理相关的工具和代码。
- **types.py**: 自定义类型定义。
- **constants/settings**: 一些包级别的常量，应该少一些，比如debug模式开关等。
- **deprecation management**: 过时代码管理相关的工具和代码。
- **dependency management**: 依赖管理相关的工具和代码，比如前文提到的 `require_python_package` decorator。
- 其他业务相关代码

## Appendix

### Python Package: `dowhen`

dowhen 是一个instrumentation的工具，可以用来做测试、调试、软件安全分析等。Python没有内置的instrumentation工具。Python的core dev {{<github "gaogaotiantian">}} 利用 Python3.12引入的新特性 sys.monitoring 开发了这个工具。

dowhen的核心API就两个，一是负责执行什么的 callback/do，二是负责什么时候执行的trigger/when。为了把do和when更好的结合起来，比如提供context manager，提供trigger时机的判断等，因此dowhen的底层是一个handler模块；为了使用系统提供的sys.monitoring模块，更底层是一个instrumenter模块。

dowhen的基本使用方法可以参考其官方文档 [^dowhen]，这里会介绍下dowhen的API设计和实现思路。

{{<details "dowhen.trigger">}}

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

{{<details "dowhen.callback">}}

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

## Citation

{{< bibtex >}}

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

[^pypa_packaging]: https://packaging.python.org/en/latest/tutorials/packaging-projects/

[^packaging_principles]: https://medium.com/@miqui.ferrer/python-packaging-best-practices-4d6da500da5f

[^structuring_your_project]: https://docs.python-guide.org/writing/structure/

[^pythonic_api]: https://benhoyt.com/writings/python-api-design/
