import json
import pandas as pd


def load_data(filepath):
    try:
        with open(filepath, "r") as f:
            raw = json.load(f)
        if "data" not in raw or not isinstance(raw["data"], list):
            raise ValueError("JSON must contain a 'data' key with a list of records")

        records = raw["data"]
        # When the data objects contain nested metadata and data blocks, flatten them.
        if all(isinstance(d, dict) and "metadata" in d and "data" in d for d in records):
            flattened = []
            for entry in records:
                meta = entry.get("metadata", {})
                values = entry.get("data", {})
                if not isinstance(meta, dict) or not isinstance(values, dict):
                    raise ValueError("Each item's 'metadata' and 'data' must be dictionaries")

                combined = {**meta, **values}
                ci95 = values.get("ci95")
                if isinstance(ci95, (list, tuple)) and len(ci95) == 2:
                    combined["ci95_lower"] = ci95[0]
                    combined["ci95_upper"] = ci95[1]
                flattened.append(combined)
            dataframe = pd.DataFrame(flattened)
        else:
            if not all(isinstance(d, dict) for d in records):
                raise ValueError("Each item in 'data' must be a dictionary")
            dataframe = pd.DataFrame(records)

        return dataframe, raw.get("metadata", {})
    except Exception as e:
        raise ValueError(f"Failed to load or parse JSON: {e}")
