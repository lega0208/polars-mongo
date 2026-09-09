//! The BSON <-> Arrow boundary.
//!
//! `schema` compiles the caller's pyarrow schema into the single `FieldPlan`
//! authority; `filter` converts caller dictionaries into BSON documents.

pub mod decode;
pub mod encode;
pub mod filter;
pub mod schema;

#[cfg(test)]
mod tests;
