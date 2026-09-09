//! ObjectId: one shared extension identity, the 12-byte conversions, and the
//! single recursive Arrow transport.

use mongodb::bson::oid::ObjectId;
use polars_arrow::array::{Array, FixedSizeListArray, PrimitiveArray, StructArray, new_null_array};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::{ArrowDataType, ExtensionType, Field as ArrowField};
use polars_arrow::offset::OffsetsBuffer;
use polars_core::datatypes::DataType;
use polars_utils::pl_str::PlSmallStr;

/// The single extension name used on both sides of the FFI boundary.
pub const OBJECT_ID_EXT_NAME: &str = "polars_mongo.object_id";

/// Rebuild an ObjectId from its 12 raw bytes. Never lossy.
pub fn object_id_from_bytes(bytes: [u8; 12]) -> ObjectId {
    ObjectId::from_bytes(bytes)
}

/// Lowercase 24-character hex rendering.
pub fn object_id_to_hex(bytes: [u8; 12]) -> String {
    object_id_from_bytes(bytes).to_hex()
}

/// Parse a 24-character hex string into raw bytes.
pub fn object_id_from_hex(hex: &str) -> Option<[u8; 12]> {
    ObjectId::parse_str(hex).ok().map(|oid| oid.bytes())
}

/// The storage type: a 12-element fixed-size list of `uint8`.
pub fn object_id_storage_dtype() -> ArrowDataType {
    ArrowDataType::FixedSizeList(
        Box::new(ArrowField::new(
            PlSmallStr::from_static("item"),
            ArrowDataType::UInt8,
            false,
        )),
        12,
    )
}

/// The extension dtype constructed at **every** ObjectId node, at any depth.
///
/// There is exactly one transport: the recursive Arrow FFI schema export
/// carries the extension name for nested children as well as top-level fields,
/// so no second, conditional path exists.
pub fn object_id_arrow_dtype() -> ArrowDataType {
    ArrowDataType::Extension(Box::new(ExtensionType {
        name: PlSmallStr::from_static(OBJECT_ID_EXT_NAME),
        inner: object_id_storage_dtype(),
        metadata: Some(PlSmallStr::EMPTY),
    }))
}

/// Build an ObjectId array from optional 12-byte values.
pub fn object_id_array(values: &[Option<[u8; 12]>]) -> FixedSizeListArray {
    let mut bytes: Vec<u8> = Vec::with_capacity(values.len() * 12);
    let mut validity = Vec::with_capacity(values.len());
    for value in values {
        match value {
            Some(raw) => {
                bytes.extend_from_slice(raw);
                validity.push(true);
            }
            None => {
                bytes.extend(std::iter::repeat_n(0u8, 12));
                validity.push(false);
            }
        }
    }
    let inner = PrimitiveArray::<u8>::from_vec(bytes).boxed();
    let validity: Bitmap = validity.into_iter().collect();
    FixedSizeListArray::new(object_id_arrow_dtype(), values.len(), inner, Some(validity))
}

/// Wrap an ObjectId array in a struct with a single `oid` field.
pub fn struct_of_object_id(
    values: &[Option<[u8; 12]>],
    validity: Option<Bitmap>,
) -> Box<dyn Array> {
    let child = object_id_array(values).boxed();
    let dtype = ArrowDataType::Struct(vec![ArrowField::new(
        PlSmallStr::from_static("oid"),
        object_id_arrow_dtype(),
        true,
    )]);
    StructArray::new(dtype, values.len(), vec![child], validity).boxed()
}

/// Wrap an array in a large list with the given offsets.
pub fn large_list_of(
    inner: Box<dyn Array>,
    offsets: Vec<i64>,
    validity: Option<Bitmap>,
    field_name: &str,
) -> Box<dyn Array> {
    let dtype = ArrowDataType::LargeList(Box::new(ArrowField::new(
        PlSmallStr::from(field_name),
        inner.dtype().clone(),
        true,
    )));
    let offsets = OffsetsBuffer::try_from(offsets).expect("monotonic offsets");
    polars_arrow::array::ListArray::<i64>::new(dtype, offsets, inner, validity).boxed()
}

/// An all-null ObjectId array, used for the null-parent cases.
pub fn null_object_id_array(length: usize) -> Box<dyn Array> {
    new_null_array(object_id_arrow_dtype(), length)
}

/// The extension name carried at every level of a dtype, outermost first.
///
/// This is the transport oracle: it reports what identity actually survived,
/// rather than assuming the export layer preserved it.
pub fn extension_names(dtype: &DataType) -> Vec<String> {
    let mut names = Vec::new();
    collect_extension_names(dtype, &mut names);
    names
}

fn collect_extension_names(dtype: &DataType, names: &mut Vec<String>) {
    match dtype {
        DataType::Extension(instance, storage) => {
            names.push(instance.name().to_string());
            collect_extension_names(storage, names);
        }
        DataType::List(inner) => collect_extension_names(inner, names),
        DataType::Array(inner, _) => collect_extension_names(inner, names),
        DataType::Struct(fields) => {
            for field in fields {
                collect_extension_names(&field.dtype, names);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use polars_core::series::Series;

    use super::*;

    const HEX: &str = "507f1f77bcf86cd799439011";

    fn bytes() -> [u8; 12] {
        object_id_from_hex(HEX).unwrap()
    }

    #[test]
    fn bytes_round_trip_is_byte_exact() {
        let raw = bytes();
        let oid = object_id_from_bytes(raw);
        assert_eq!(oid.bytes(), raw);
        assert_eq!(object_id_to_hex(raw), HEX);
    }

    #[test]
    fn rejects_invalid_hex() {
        assert!(object_id_from_hex("nope").is_none());
        assert!(object_id_from_hex("507f1f77bcf86cd79943901").is_none());
        assert!(object_id_from_hex("zzzf1f77bcf86cd799439011").is_none());
    }

    #[test]
    fn flat_series_carries_the_extension_identity() {
        let array = object_id_array(&[Some(bytes()), None]).boxed();
        let series =
            Series::from_arrow(PlSmallStr::from_static("_id"), array).expect("series builds");
        assert_eq!(extension_names(series.dtype()), vec![OBJECT_ID_EXT_NAME]);
        assert_eq!(series.len(), 2);
        assert_eq!(series.null_count(), 1);
    }

    #[test]
    fn nested_series_carry_the_extension_identity_at_every_level() {
        let struct_array = struct_of_object_id(&[Some(bytes()), None], None);
        let series = Series::from_arrow(PlSmallStr::from_static("s"), struct_array)
            .expect("struct series builds");
        assert_eq!(extension_names(series.dtype()), vec![OBJECT_ID_EXT_NAME]);

        let list_array = large_list_of(
            object_id_array(&[Some(bytes()), Some(bytes())]).boxed(),
            vec![0, 1, 2],
            None,
            "element",
        );
        let series = Series::from_arrow(PlSmallStr::from_static("l"), list_array)
            .expect("list series builds");
        assert_eq!(extension_names(series.dtype()), vec![OBJECT_ID_EXT_NAME]);

        let list_of_struct = large_list_of(
            struct_of_object_id(&[Some(bytes()), Some(bytes())], None),
            vec![0, 2],
            None,
            "element",
        );
        let series = Series::from_arrow(PlSmallStr::from_static("ls"), list_of_struct)
            .expect("list-of-struct series builds");
        assert_eq!(extension_names(series.dtype()), vec![OBJECT_ID_EXT_NAME]);
    }
}
