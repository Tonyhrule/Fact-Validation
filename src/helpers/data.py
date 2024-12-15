import asyncio
from collections.abc import Callable, Coroutine
from json import load, loads, dump, dumps
import os
import re
from typing import Any

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


async def queue(data: list[Callable[[], Coroutine[Any, Any, Any]]], max_concurrent=80):
    async def worker(queue: asyncio.Queue, results):
        while True:
            index, func = await queue.get()
            try:
                results[index] = await func()
            finally:
                queue.task_done()

    queue = asyncio.Queue()
    results: list = [None] * len(data)

    for index, func in enumerate(data):
        await queue.put((index, func))

    workers = [
        asyncio.create_task(worker(queue, results)) for _ in range(max_concurrent)
    ]

    await queue.join()

    for w in workers:
        w.cancel()

    await asyncio.gather(*workers, return_exceptions=True)

    return results
