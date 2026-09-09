from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from polars.testing import assert_frame_equal


def test_readme_objectid_example_executes() -> None:
    readme = Path(__file__).parents[1] / "README.md"
    match = re.search(
        r"`object_id_timestamp` operates.*?^```python\n(.*?)^```",
        readme.read_text(),
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None

    namespace: dict[str, Any] = {}
    exec(match.group(1), namespace)

    import polars as pl

    result = namespace["result"]
    assert isinstance(result, pl.DataFrame)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "_id": ["507f1f77bcf86cd799439011", None],
                "created_at": [
                    datetime(2012, 10, 17, 21, 13, 27),
                    None,
                ],
            },
            schema={"_id": pl.String, "created_at": pl.Datetime("ms")},
        ),
    )
