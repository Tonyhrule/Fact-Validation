from collections.abc import Callable


def function_from_list(prompt: str, options: list[tuple[str, Callable]]):
    selection = input(
        prompt
        + "\n"
        + "\n".join([f"{i+1}. {d[0]}" for i, d in enumerate(options)])
        + "\n\n"
    )

    if not selection.isdigit() or int(selection) < 1 or int(selection) > len(options):
        print("Invalid selection. Please try again.\n")
        return function_from_list(prompt, options)

    options[int(selection) - 1][1]()
