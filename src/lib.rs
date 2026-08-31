mod expressions;

// The cdylib is loaded directly by Polars through `register_plugin_function`,
// so no `#[pymodule]` entry point is required. Everything exported to Polars
// lives in `expressions.rs` behind `#[polars_expr]`.
