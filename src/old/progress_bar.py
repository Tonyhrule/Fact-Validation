import progressbar

class Progress:
    def __init__(self, max_value: int):
        self.bar = progressbar.ProgressBar(
            maxval=max_value,
            widgets=[
                " [",
                progressbar.Timer(),
                "] [",
                progressbar.Counter(),
                f" / {max_value}] ",
                progressbar.Bar(),
                " (",
                progressbar.ETA(),
                ") ",
            ],
        )
        self.value = 0
        self.bar.start()
    def update(self, value: int):
        self.bar.update(value)
        self.value = value
    def increment(self):
        self.update(self.value + 1)
    def finish(self):
        self.bar.finish()