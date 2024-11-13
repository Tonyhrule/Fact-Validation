from json import load, loads, dump, dumps
import os

from helpers.variables import SRC_DIR


def stringify(data):
    return dumps(
        data,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse(data):
    return loads(data)


def read_json(file):
    with open(SRC_DIR + file, "r") as f:
        return load(f)


def save_json(file, data):
    with open(SRC_DIR + file, "w") as f:
        dump(data, f, indent=2)


def save_file(file, data):
    with open(SRC_DIR + file, "w") as f:
        f.write(data)


def add_to_file(file, data):
    with open(SRC_DIR + file, "a") as f:
        f.write(data)


def delete_file(file):
    os.remove(SRC_DIR + file)


def chunk_list(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
