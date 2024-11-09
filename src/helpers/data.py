from json import load, loads, dump, dumps


def stringify(data):
    return dumps(data)


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
