import base64
import json
from typing import Any


def _bin_transform_str(s: str, transform_fn) -> str:
    return transform_fn(s.encode("utf-8")).decode("utf-8")


class Base64Codec:
    def encode(self, s: str) -> str:
        return _bin_transform_str(s, base64.b64encode)

    def decode(self, s: str) -> str:
        return _bin_transform_str(s, base64.b64decode)


Base64 = Base64Codec()


def encode_base64_json(obj: Any) -> str:
    return Base64.encode(json.dumps(obj))
