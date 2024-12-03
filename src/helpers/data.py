from json import load, loads, dump, dumps
import os
import re

from helpers.variables import SRC_DIR


def stringify(data: dict | list):
    return dumps(
        data,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse(data: str):
    return loads(data)


def read_json(path: str):
    with open(SRC_DIR + path, "r") as f:
        return load(f)


def save_json(path: str, data: dict | list):
    full_path = os.path.join(SRC_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        dump(data, f, indent=2)


def save_file(path: str, data: str):
    with open(SRC_DIR + path, "w") as f:
        f.write(data)


def add_to_file(path: str, data: str):
    with open(SRC_DIR + path, "a") as f:
        f.write(data)


def delete_file(path: str):
    os.remove(SRC_DIR + path)


def chunk_list(list: list, chunk_size: int):
    return [list[i : i + chunk_size] for i in range(0, len(list), chunk_size)]


def get_number(string: str) -> str:
    """Returns the last number in a string."""
    return re.findall(r"[-+]?\d*\.\d+|\d+", string)[-1]
