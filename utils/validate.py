import argparse
import json
import sys
import jsonschema
from jsonschema import validate
from typing import Iterable, Tuple, Union
from config import DEFAULT_SCHEMAS


def _validate_single(path: str, schema: dict) -> Tuple[bool, str]:
    with open(path) as f:
        data = json.load(f)

    try:
        validate(instance=data, schema=schema)
        return True, "Valid"
    except jsonschema.exceptions.ValidationError as e:
        return False, f"Validation error: {e.message}"


def validate_json(data_path: Union[str, Iterable[str]], schema_path: str):
    """Validate one or many JSON files against the provided schema.

    Returns a tuple of (is_valid, message). For multiple inputs, stops at the
    first validation error and reports the offending file in the message.
    """
    with open(schema_path) as f:
        schema = json.load(f)

    if isinstance(data_path, str):
        return _validate_single(data_path, schema)

    for path in data_path:
        ok, message = _validate_single(path, schema)
        if not ok:
            return False, f"{path}: {message}"
    return True, "Valid"


def main():
    parser = argparse.ArgumentParser(description="Validate simulation data JSON against a schema.")
    parser.add_argument("--data", required=True, help="Path to the data JSON file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--schema", help="Path to a JSON schema file.")
    group.add_argument(
        "--diagram",
        choices=DEFAULT_SCHEMAS.keys(),
        help="Diagram type to auto-select the bundled schema.",
    )
    args = parser.parse_args()

    schema_path = args.schema or DEFAULT_SCHEMAS[args.diagram]
    is_valid, message = validate_json(args.data, schema_path)

    print(message)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
