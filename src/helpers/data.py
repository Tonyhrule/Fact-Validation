from json import load, loads, dump, dumps
import os


def stringify(data):
    return dumps(
        data,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def parse(data):
    return loads(data)


def read_json(file):
    with open(file, "r") as f:
        return load(f)


def save_json(file, data):
    with open(file, "w") as f:
        dump(data, f, indent=2)


def save_file(file, data):
    with open(file, "w") as f:
        f.write(data)


def add_to_file(file, data):
    with open(file, "a") as f:
        f.write(data)


def delete_file(file):
    os.remove(file)
