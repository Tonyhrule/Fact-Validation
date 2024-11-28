from tqdm import tqdm

class Progress:
    def __init__(self, max_value: int, description: str = "Progress"):
        self.bar = tqdm(total=max_value, desc=description, ncols=100, unit="step")
        self.value = 0

    def update(self, value: int):
        self.bar.n = value
        self.bar.refresh()
        self.value = value

    def increment(self):
        self.bar.update(1)
        self.value += 1

    def finish(self):
        self.bar.close()
