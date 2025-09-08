---
date: '2025-08-03T18:45:17+08:00'
title: 'Python Package Design: API, Dependency and Code Structure'
tags: ["python", "package", "API", "dependency", "structure"]
---

Python—a language well-suited for fast prototyping in the AI field—has become increasingly popular these days. Creating a well-designed Python package at the right time can make your work more impactful and more likely to be adopted.

Since October 2024, I have been learning about Python package development. Since then, I have published my own work {{< github "IAAR-Shanghai/UHGEval" >}} as a Python package, participated in the early architecture design of {{< github "MemTensor/MemOS" >}}, and deeply studied and contributed to {{< github "gaogaotiantian/dowhen" >}}. These experiences have given me some insights into Python package design, which I document in this post.

Among the projects I've worked on, {{< github "gaogaotiantian/dowhen" >}} stands out for its elegant and truly Pythonic design. I highly encourage you to explore it in depth; I also provide an extended introduction in the [Appendix](#python-package-dowhen).

## Related Work

There are already many articles and tutorials online about building and designing Python packages. For example:

- (Must-read) *Packaging Python Projects*, the official tutorial from the Python Packaging Authority (PyPa), which concisely introduces the process of building and publishing Python packages [^pypa_packaging].
- (Must-read) *Designing Pythonic library APIs*, an article on how to design Pythonic APIs [^pythonic_api].
- *Python Packaging Best Practices*, an article explaining packaging principles [^packaging_principles], including sdist, wheel, and an introduction to frontend/backend tools.
- *Structuring Your Project*, a slightly outdated article on Python package code structure [^structuring_your_project].

These articles are either outdated [^structuring_your_project], lack a discussion of design philosophy [^pypa_packaging] [^packaging_principles], or are not comprehensive enough [^pypa_packaging] [^pythonic_api]. This post aims to fill those gaps.

## API Design

The API is the deliverable of your package—the core product.

- Only the package's API is exposed for users.
- All other non-API codes do not need to be explained to users, and users should not have to read them when using the API (only developers need to read them).

As you can see, API design is extremely important. Ben Hoyt, in his article [^pythonic_api], offers many practical tips, and his writing is quite engaging. I highly recommend reading it. Here are his takeaways:

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

In addition to these suggestions, this post briefly supplements how to achieve a stable and simple API.

### APIs Should Remain Stable and Simple

Code should be continuously refactored to adapt to new situations. However, at least within each major version (e.g., v0.\*.\*, v1.\*.\*), the API should remain stable. This means that *before developing a package*, you should already have a clear idea of the core functionality your package will provide!

If the core functionality is logically complete, then the API should be stable and simple. For example, CRUD (Create, Read, Update, Delete) is a typical logically complete set of functions; similarly, Einstein Operations are a logically complete set of mathematical operations—{{< github "arogozhnikov/einops">}} only provides three core APIs[^einops]! Good API design often draws inspiration from logic and mathematics.

{{<media
src="https://user-images.githubusercontent.com/6318811/177030658-66f0eb5d-e136-44d8-99c9-86ae298ead5b.mp4"
caption="Video 1: Einstein Operation Introduction"
>}}

Having fewer APIs brings other benefits: it's easier for users to remember, easier for developers to write documentation (since it's clear which functions/classes need detailed explanations, examples, and use cases), and easier to write tests, as these APIs are expected to be more stable.

### How to Achieve a Stable and Simple API?

Here are two Pythonic approaches (I think).

**Achieving simplicity through parameter polymorphism.** {{< github "gaogaotiantian/dowhen" >}} exposes only seven functions to users: `['bp', 'clear_all', 'do', 'get_source_hash', 'goto', 'when', 'DISABLE']`, with `bp`, `goto`, `do`, and `when` being the core ones. It's very clear what users should use. `dowhen` mainly achieves this through parameter polymorphism. For example, the `when` function for creating triggers has the following signature:

```python
    def when(
        cls,
        entity: CodeType | FunctionType | MethodType | ModuleType | type | None,
        *identifiers: IdentifierType | tuple[IdentifierType, ...],
        condition: str | Callable[..., bool | Any] | None = None,
        source_hash: str | None = None,
    ):
```

Here, `IdentifierType = int | str | re.Pattern | Literal["<start>", "<return>"] | None`. Since `identifiers` is a variadic parameter, it implicitly supports logical AND and OR relationships. As a result, this function supports a very wide range of functionality. See the [Appendix](#python-package-dowhen) for details on its implementation.

**Achieving simplicity through the factory pattern.** {{< github "huggingface/transformers" >}} also has simple and memorable APIs, but relies more on the factory pattern, such as `AutoModelForCausalLM`, `AutoModelForSequenceClassification`, and even the abstract factory `pipeline`. The user-facing API parameters are not particularly complex with the most important one being `model_name_or_path`, so you don't need to worry much about parameter interactions. Overall, it's easy for users to use and remember.

### When Simplicity Is Impossible

Sometimes, things don't go as we wish. If you look at the API of {{< github "NVIDIA/Megatron-LM" >}}, you'll find it very complex; the same goes for {{< github "hiyouga/LLaMA-Factory" >}} and {{< github "MemTensor/MemOS" >}}. These libraries are often experimental, and their interfaces are unlikely to be stable. As a result, "getting things running" is a big part of using them, and their APIs are hard to keep simple. When the API can't be simplified, users must learn about it through some channel. So, our focus shifts to optimizing that channel.

- For {{< github "NVIDIA/Megatron-LM" >}}, users are expected to git clone the entire package. It provides various training template bash scripts for easy modification. For further customization, users can directly read [megatron/training/arguments.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py), which aggregates all parameters.
- For {{< github "hiyouga/LLaMA-Factory" >}}, which also has many parameters, its [documentation](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/sft.html#) provides detailed parameter descriptions for each training paradigm. LLaMA-Factory also launches a Gradio interface, visually unifying parameter descriptions and settings.
- For {{< github "MemTensor/MemOS" >}}, it relies on three channels: detailed example scripts, various cookbooks, and using Pydantic to constrain user input, so that error messages guide users on how to fix issues.

## Dependency Management

Python does not have an official package management tool like Rust or NodeJS. However, with the gradual standardization of `pyproject.toml` by the Python organization [^pep735] [^pep751], Python's package management tools are maturing.

In package management, dependency management is one of the most important aspects. Here, we focus only on dependency-related topics including management tools, dependency types, version ranges, and lazy imports tricks.

### Management Tools

There are many options, such as {{< github "pypa/setuptools" >}} and {{< github "python-poetry/poetry" >}}. My personal recommendation is {{< github "astral-sh/uv" >}}, which represents the future direction of packaging. It has three main advantages:

- Written in Rust, so it's fast [^uv_rust].
- Supports Python's official `pyproject.toml` specification, with many schemas closely following Python PEPs.
- Supports integration with the PyTorch ecosystem [^uv_torch].

### Three Types of Dependencies

As described in the uv official documentation, there are three types of dependencies[^uv_deps]:

> - `project.dependencies`: Published dependencies.
> - `project.optional-dependencies`: Published optional dependencies, or "extras".
> - `dependency-groups`: Local dependencies for development.

**Published dependencies** are installed when running `pip install my-package`. These should include all core dependencies—without any of them, almost none of the package's code will run. However, there are exceptions: for packages like `torch`, which are platform-dependent and huge, it's better to guide users to install them themselves (see [# Lazy Imports and Guided Installation](#lazy-imports-and-guided-installation)).

**Published optional dependencies** are installed with `pip install my-package[extra]`. These should include all optional dependencies—without any of them, the core code still runs, but a major feature block won't. This requires managing different feature/extra groups. For example, {{< github "MemTensor/MemOS" >}} groups dependencies by memory type, such as `MemoryOS[tree-mem]`, `MemoryOS[mem-reader]`; {{< github "EleutherAI/lm-evaluation-harness" >}} groups dependencies by supported benchmarks, such as `lm-eval[ifeval]`, `lm-eval[math]`.

**Local dependencies for development** are for development only. These include all dependencies needed during development, such as testing, code formatting, and documentation. They are used at different stages: testing dependencies (`pytest`, `coverage`) and code formatting dependencies (`mypy`, `ruff`) may only be needed in CI/CD; documentation tools (`sphinx`, `mkdocs`) may only be needed when publishing docs. Not all development dependencies need to be installed locally, so grouping them is more reasonable.

### Specifying Dependency Version Ranges

Generally, we can assume that the latest version of a dependency is the most compatible, so we can set the lower bound to the latest version. We should also set an upper bound, usually the next major version (the first non-zero digit plus one), e.g., `>=1.0.0, <2.0.0`, `>=0.2.0, <0.3.0`.

This is because a non-zero major version usually indicates a major change; as long as the major version doesn't change, we can assume the dependency's API hasn't broken.

### Lazy Imports and Guided Installation

In many open-source repositories, `torch` is difficult to manage as a dependency because its installation is platform-dependent and it's huge.

When using {{< github "huggingface/transformers" >}}, you may also have encountered issues about missing packages like `sentencepiece` or `einops`, which are dependencies for specific models.

For those special dependencies mentioned above, we don't want users to install them at `pip install` time. Instead, we want to check at runtime whether they're installed, and if not, provide installation guidance. The following decorator achieves this:

{{<details "dependency.py">}}

```python
"""
This utility provides tools for managing dependencies in MemOS.
"""

import functools
import importlib


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

{{</details>}}

Functions decorated with this will check at runtime whether the specified package is installed. If not, an `ImportError` is raised with installation instructions. Note that you should lazily import the package inside the function, not at the top of the file, to avoid excessive error messages.

Finally, for developers, we want to avoid waiting until runtime to install these dependencies. So, we should include them in a catch-all optional dependency group, the `all` extras group, so that installing `my-package[all]` will install all development-time dependencies at once:

```toml
all = [
    # Exist in the above optional groups
    "neo4j (>=5.28.1,<6.0.0)",
    "schedule (>=1.2.2,<2.0.0)",
    "redis (>=6.2.0,<7.0.0)",
    "pika (>=1.3.2,<2.0.0)",
    "chonkie (>=1.0.7,<2.0.0)",
    "markitdown[docx,pdf,pptx,xls,xlsx] (>=0.1.1,<0.2.0)",

    # NOT exist in the above optional groups
    # Because they are either huge-size dependencies or infrequently used dependencies.
    # We kindof don't want users to install them by default.
    "torch (>=2.7.1,<3.0.0)",
    "sentence-transformers (>=4.1.0,<5.0.0)",
    "qdrant-client (>=1.14.2,<2.0.0)",
    "volcengine-python-sdk (>=4.0.4,<5.0.0)",
    "chromadb-client (>=1.0.15,<2.0.0)",
]
```

## Code Structure

Here's what a modern Python package's GitHub repository typically contains and how it's organized.

### Repository Root Directory

- **README.md**: Introduction, usage, installation instructions, etc.
- **LICENSE**: License file.
- **pyproject.toml**: Python package metadata, dependencies, build backend, and tool settings (mypy, uv, pytest, etc.).
- **Makefile**: Contains common commands like `make test`, `make build`, `make clean`, etc.
- **.gitignore**: Files to ignore in Git, such as build artifacts and test outputs. Recommended: use the `Python.gitignore` from {{< github "github/gitignore" >}}.
- **.pre-commit-config.yaml**: Pre-commit hook configuration for code style checks before commits.
- **src/my_package/** or **my_package/**: Source code directory. This is what gets installed via pip. Adding a `src/` layer helps avoid import confusion and allows managing multiple packages in one repo (e.g., `src/my_package/` and `src/my_other_package/`).
- **tests/**: Test code directory. `Pytest` is recommended for its simplicity and Pythonic style.
- **docs/**: Documentation directory.
- **.github/**: GitHub configuration, including CI/CD workflows, issue templates, PR templates, etc. See {{< github "MemTensor/MemOS" >}} for examples.
- etc.

### Python Package Directory

Inside **src/my_package/** or **my_package/**, you often find the following (all present in {{< github "MemTensor/MemOS" >}}):

- **\_\_init\_\_.py**: Package initializer, exposing core APIs.
- **api**: Core HTTP API code. Recommend using `FastAPI` for its simplicity.
- **cli**: Command-line interface code. Recommend using `argparse`, `click`, etc.
- **configs**: Configuration protocols, often using `Pydantic`.
- **log.py**: Custom logging handlers, usually based on Python's `logging` module.
- **exceptions.py**: Error handling utilities.
- **types.py**: Custom type definitions.
- **constants/settings**: Package-level constants (should be minimal, e.g., `debug` flags).
- **deprecation management**: Tools for managing deprecated code.
- **dependency management**: Tools for dependency management, such as the `require_python_package` decorator above.
- Other business logic.

## Appendix

### Python Package: `dowhen`

{{< github "gaogaotiantian/dowhen" >}} is an instrumentation tool for testing, debugging, software security analysis, etc. Python does not have a built-in instrumentation tool. Python core developer {{<github "gaogaotiantian">}} developed this tool using the new `sys.monitoring` feature introduced in Python 3.12.

The core APIs of `dowhen` are just two: one for specifying what to execute (`callback.py`, implementing the `do` part), and one for specifying when to execute (`trigger.py`, implementing the `when` part).

To better combine `do` and `when` (e.g., providing context managers, trigger timing checks), the underlying layer is a `handler.py` module; to use the system's `sys.monitoring`, there's an lower-level `instrumenter.py` module. These four modules together form the `dowhen` package.

For basic usage, see the official documentation [^dowhen]. Here, I introduce the API design and implementation ideas behind the APIs.

{{<details "Trigger Workflow">}}

```python
@classmethod
def when(
    cls,
    entity: CodeType | FunctionType | MethodType | ModuleType | type | None,
    *identifiers: IdentifierType | tuple[IdentifierType, ...],
    condition: str | Callable[..., bool | Any] | None = None,
    source_hash: str | None = None,
):
    # 1. Check if condition is executable and of the correct type

    # 2. Use source_hash to check if the runtime entity has changed since provided

    events = []

    # 3. Breadth-first expand code objects from entity
    code_objects = cls._get_code_from_entity(entity)

    # 4. Define trigger events based on conditions

    # 4.1. If no identifiers are provided, all-line matching events
    if not identifiers:
        for code in code_objects:
            events.append(_Event(code, "line", {"line_number": None}))

    # 4.2. If identifiers are provided, create regular events
    else:

        # 4.2.1. Convert relative line numbers to absolute before expanding `code_objects`,
        # as expansion may cause duplicated line events in different code objects
        identifiers = cls.unify_identifiers(entity, *identifiers)

        # 4.2.2. For each identifier × code object, create corresponding events
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

                    # 4.2.2.3. Other identifiers × None
                    if code is None:
                        # Global event, entity is None
                        events.append(
                            _Event(
                                None,
                                "line",
                                {"line_number": None, "identifier": identifier},
                            )
                        )

                    # 4.2.2.4. Other identifiers × code object
                    else:

                        # 4.2.2.4.1. Parse identifier to line numbers, handling:
                        # first-line matching, compiled code objects, logical
                        # AND/OR in identifiers, nested code objects, comments
                        # not in co_lines, etc.
                        line_numbers = get_line_numbers(code, identifier)

                        # c is depth-first expanded code objects, ensuring correct trigger positions
                        for c, numbers in line_numbers.items():
                            for number in numbers:
                                events.append(
                                    _Event(c, "line", {"line_number": number})
                                )

    if not events:
        raise ValueError(
            "Could not set any event based on the entity and identifiers."
        )

    # 5. Return Trigger instance
    return cls(events, condition=condition, is_global=entity is None)
```

{{</details>}}

{{<details "Callback Workflow">}}

```python
Instrumented code line
↓
sys.monitoring  # Listen for events and bind callbacks
↓
instrumenter.py::Instrumenter().*_callback()  # Callback really bound by sys.monitoring
↓
instrumenter.py::Instrumenter()._process_handlers()  # High-level callback wrapper, adds sys.monitoring.DISABLE
↓
handler.py::EventHandler().__call__()  # Mid-level callback wrapper, adds event timing checks, has_event and should_fire logic
↓
callback.py::Callback().__call__()  # Low-level callback wrapper, adds call_code/call_goto/call_bp checks
↓
callback.py::Callback.call_*()  # Callback execution
↓
User-defined code  # If call_code, user code is executed
```

{{</details>}}

## Citation

{{< bibtex >}}

## References

[^pep735]: Rosen, Stephen. “PEP 735 – Dependency Groups in pyproject.toml.” *Python Enhancement Proposals*, 20 Nov. 2023, https://peps.python.org/pep-0735/. Accessed 3 Aug. 2025.

[^pep751]: Cannon, Brett. “PEP 751 – A file format to record Python dependencies for installation reproducibility.” *Python Enhancement Proposals*, 24 July 2024, https://peps.python.org/pep-0751/. Accessed 3 Aug. 2025.

[^uv_rust]: Astral. “uv.” *uv Docs*, n.d., https://docs.astral.sh/uv/. Accessed 3 Aug. 2025.

[^uv_torch]: Astral. “Using uv with PyTorch.” *uv Docs*, n.d., https://docs.astral.sh/uv/guides/integration/pytorch/. Accessed 3 Aug. 2025.

[^uv_deps]: Astral. “Managing dependencies.” *uv Docs*, n.d., https://docs.astral.sh/uv/concepts/projects/dependencies/. Accessed 3 Aug. 2025.

[^dowhen]: Gao, Tian. “dowhen documentation.” *dowhen Docs*, n.d., https://dowhen.readthedocs.io/en/latest/. Accessed 3 Aug. 2025.

[^einops]: Rogozhnikov, Alex. “einops.” *einops Docs*, n.d., https://einops.rocks/. Accessed 3 Aug. 2025.

[^pypa_packaging]: Python Packaging Authority. “Packaging Python Projects.” *Packaging Python Projects*, n.d., https://packaging.python.org/en/latest/tutorials/packaging-projects/. Accessed 3 Aug. 2025.

[^packaging_principles]: Ferrer, Miqui. “Python Packaging Best Practices.” *Medium*, n.d., https://medium.com/@miqui.ferrer/python-packaging-best-practices-4d6da500da5f. Accessed 3 Aug. 2025.

[^structuring_your_project]: Reitz, Kenneth, and Schlusser, Tanya. “Structuring Your Project.” *The Hitchhiker’s Guide to Python*, n.d., https://docs.python-guide.org/writing/structure/. Accessed 3 Aug. 2025.

[^pythonic_api]: Hoyt, Ben. “Pythonic API Design.” *Ben Hoyt’s blog*, n.d., https://benhoyt.com/writings/python-api-design/. Accessed 3 Aug. 2025.
