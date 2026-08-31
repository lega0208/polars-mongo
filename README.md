# polars-mongo

A [Polars](https://pola.rs) expression plugin written in Rust, packaged with
[maturin](https://www.maturin.rs) and managed with [uv](https://docs.astral.sh/uv/).

## Quickstart

```bash
make dev     # uv sync + maturin develop --uv
make test    # cargo test + pytest
make lint    # fmt, clippy, ruff, mypy
```

```python
import polars as pl
import polars_mongo as pm

df = pl.DataFrame({"_id": ["507f1f77bcf86cd799439011", "nope"]})

df.select(
    pm.is_object_id("_id"),
    created_at=pm.object_id_timestamp("_id", time_unit="ms"),
)

# or through the registered expression namespace
df.filter(pl.col("_id").mongo.is_object_id())
```

## Layout

| Path                            | Purpose                                                |
| ------------------------------- | ------------------------------------------------------ |
| `src/expressions.rs`            | `#[polars_expr]` functions exposed to Polars           |
| `src/lib.rs`                    | crate root; no `#[pymodule]` needed for a plugin       |
| `python/polars_mongo/`          | Python wrappers + compiled `_internal` shared library  |
| `tests/`                        | pytest suite exercising the built extension            |
| `Cargo.toml` / `pyproject.toml` | Rust and Python package metadata (maturin mixed layout) |

## Adding an expression

1. Write the function in `src/expressions.rs`:

   ```rust
   #[polars_expr(output_type=String)]
   fn shout(inputs: &[Series]) -> PolarsResult<Series> {
       let ca = inputs[0].str()?;
       Ok(ca.apply_values(|s| s.to_uppercase().into()).into_series())
   }
   ```

   Non-static output types use `output_type_func`, and functions taking keyword
   arguments use `output_type_func_with_kwargs` plus a `#[derive(Deserialize)]`
   kwargs struct (see `object_id_timestamp`).

2. Add a thin Python wrapper in `python/polars_mongo/__init__.py` calling
   `register_plugin_function`. Set `is_elementwise=True` only when the output row
   depends solely on the matching input row — it enables streaming and chunked
   evaluation, and lying here produces silently wrong results.

3. Cover it in `tests/test_expressions.py`, then run `make test`.

## Notes

- `maturin develop` warns that `PyInit__internal` is missing. That is expected: Polars
  loads the shared library directly via `register_plugin_function`, so the crate has no
  `#[pymodule]` and Python never imports `polars_mongo._internal`.
- `pyo3/extension-module` is enabled by maturin (`[tool.maturin] features`), not in
  `Cargo.toml`, so `cargo test` still links against libpython.
- The wheel is abi3 (`abi3-py310`): one build serves Python 3.10+.
- `polars` (Rust) and the `polars` Python package are released independently; when
  bumping `pyo3-polars` check its required `polars` and `pyo3` versions.
- Debug builds are slow at runtime. Benchmark with `make release`.
