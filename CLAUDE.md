# Python Style Conventions

## Docstring formatting

Begin the docstring body on the line following the opening triple quotes, not on the same line.

```python
# BAD
def foo():
    """This is wrong."""

# GOOD
def foo():
    """
    This is correct.
    """
```

## Docstring sections

Use NumPy-style docstring format with `Parameters`, `Returns`, and other sections delimited by dashed underlines.

```python
def compute(x, alpha):
    """
    Short summary.

    Parameters
    ----
    x : array
        Input data.
    alpha : float
        Scaling factor.

    Returns
    ----
    result : array
        Transformed data.
    """
```

## Function calls with many arguments

When a function call has numerous parameters, build a `kwargs` dict with one key-value assignment per line, then unpack it at the call site.

```python
kwargs = dict()
kwargs["learning_rate"] = 1e-4
kwargs["batch_size"] = 32
kwargs["num_epochs"] = 10
train(**kwargs)
```

## Large dictionaries

Construct dictionaries with many entries using sequential key-value assignments rather than a single inline literal.

```python
config = dict()
config["key1"] = value1
config["key2"] = value2
```

## Path concatenation

Use `Path(A, B)` instead of `A / B` for joining path segments. Avoid the division operator overload for path concatenation.

```python
from pathlib import Path

# BAD
result = base_dir / "subdir" / "file.txt"

# GOOD
result = Path(base_dir, "subdir", "file.txt")
```

## Function naming

Do not use leading underscores to denote "private" functions. Everything in Python can be imported regardless, so the convention adds no value. Use plain `snake_case` for all functions.

```python
# BAD
def _parse_field(text):
    ...

# GOOD
def parse_field(text):
    ...
```

## General style

Favor lightweight, straightforward code. Avoid unnecessary abstraction or indirection.
