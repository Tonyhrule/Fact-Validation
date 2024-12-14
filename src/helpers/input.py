import asyncio
from collections.abc import Callable, Coroutine


def function_from_list(prompt: str, options: dict[str, Callable]):
    selection = input(
        prompt
        + "\n"
        + "\n".join([f"{i+1}. {name}" for i, name in enumerate(options.keys())])
        + "\n\n"
    )

    if not selection.isdigit() or int(selection) < 1 or int(selection) > len(options):
        print("Invalid selection. Please try again.\n")
        return function_from_list(prompt, options)

    run = list(options.values())[int(selection) - 1]()

    if isinstance(run, Coroutine):
        return asyncio.run(run), list(options.keys())[int(selection) - 1]

    return run, list(options.keys())[int(selection) - 1]
