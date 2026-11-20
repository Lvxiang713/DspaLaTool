"""Training-only encoding. Missing numeric values retain their masks."""
import numpy as np
import pandas as pd
import csv
import io
from pathlib import Path


def read_table(source):
    """Read UTF-8 CSV with strict, unmangled header and row-width checks."""
    if hasattr(source,"read"):
        text = source.read()
        text = text.decode("utf-8-sig") if isinstance(text,bytes) else text.lstrip("\ufeff")
    else:
        text = Path(source).read_text(encoding="utf-8-sig")
    reader = csv.reader(io.StringIO(text),strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("The CSV is empty.")
    if not header or any(not name.strip() for name in header):
        raise ValueError("Every CSV column needs a non-empty name.")
    if len(header) != len(set(header)):
        raise ValueError("CSV column names must be unique.")
    rows = 0
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"CSV row ending at line {reader.line_num} has {len(row)} fields; expected {len(header)}.")
        rows += 1
    if not rows:
        raise ValueError("The CSV needs at least one data row.")
    return pd.read_csv(io.StringIO(text))


def numeric_values(series, name):
    blank = series.isna() | series.astype("string").str.strip().eq("").fillna(False)
    converted = pd.to_numeric(series.mask(blank),errors="coerce")
    if ((~blank) & converted.isna()).any():
        raise ValueError(f"{name}: non-numeric values in a numeric feature. Correct the input or declare it categorical.")
    if np.isinf(converted.to_numpy(dtype=float,na_value=np.nan)).any():
        raise ValueError(f"{name}: infinite numeric values are not supported. Use a valid number or a blank missing value.")
    return converted


def category_values(series):
    def canonical(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,bool):
            if not np.isfinite(value):
                return pd.NA
            return str(int(value)) if float(value).is_integer() else str(value)
        return str(value)
    return series.map(canonical).astype("string")


class Encoder:
    def fit(self, frame, schema=None):
        self.columns = list(frame.columns)
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Feature names must be unique.")
        self.spec = []
        self.names = []
        self.groups = []
        self.dropped = []
        self.schema = schema or {}
        if not isinstance(self.schema,dict) or any(v not in ("numeric","categorical","ignore") for v in self.schema.values()):
            raise ValueError("Feature schema must map column names to numeric, categorical or ignore.")
        unknown = set(self.schema) - set(self.columns)
        if unknown:
            raise ValueError(f"Feature schema includes unknown columns: {sorted(unknown)}")
        for name in self.columns:
            s = frame[name]
            kind = self.schema.get(name,"numeric" if pd.api.types.is_numeric_dtype(s) else "categorical")
            if kind == "ignore":
                self.dropped.append(name)
                continue
            if kind == "numeric":
                v = numeric_values(s,name)
                if v.nunique() < 2:
                    self.dropped.append(name)
                    continue
                self.spec.append({"name": name, "kind": "numeric"})
                self.names.append(name)
                self.groups.append(name)
            else:
                levels = sorted(category_values(s).dropna().unique().tolist())
                if len(levels) < 2:
                    self.dropped.append(name)
                    continue
                if len(levels) > 100:
                    raise ValueError(f"{name}: more than 100 categories. Remove identifiers or encode explicitly.")
                self.spec.append({"name": name, "kind": "categorical", "levels": levels,
                                  "numeric_codes": bool(pd.api.types.is_numeric_dtype(s)
                                                        and not pd.api.types.is_bool_dtype(s))})
                for level in levels:
                    self.names.append(f"{name}={level}")
                    self.groups.append(name)
        if not self.names:
            raise ValueError("No varying features remain in the training partition.")
        return self

    def transform(self, frame):
        missing_columns = [s["name"] for s in self.spec if s["name"] not in frame]
        if missing_columns:
            raise ValueError(f"Missing feature columns: {missing_columns}")
        values = []
        for spec in self.spec:
            s = frame[spec["name"]]
            if spec["kind"] == "numeric":
                values.append(numeric_values(s,spec["name"]).to_numpy(dtype=float,na_value=np.nan))
            else:
                text = category_values(pd.to_numeric(s, errors="coerce")
                                       if spec.get("numeric_codes") else s)
                unknown = s.isna() | ~text.isin(spec["levels"])
                for level in spec["levels"]:
                    a = (text == level).fillna(False).to_numpy(dtype=float)
                    a[unknown.to_numpy()] = np.nan
                    values.append(a)
        x = np.column_stack(values)
        x[~np.isfinite(x)] = np.nan
        return x

    def export(self):
        return {"columns": self.columns, "spec": self.spec, "names": self.names,
                "groups": self.groups, "dropped": self.dropped,
                "schema":self.schema, "unknown_category": "all indicators missing"}

    @classmethod
    def load(cls, obj):
        result = cls()
        for k, v in obj.items():
            setattr(result, k, v)
        if not hasattr(result,"schema"):
            result.schema = {}
        return result
