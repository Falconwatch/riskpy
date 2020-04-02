class Pipeline:
    steps={}

    def __init__(self):
        pass

    def run(self):
        for step in self.steps:
            step.fit()
        pass

    def result(self):
        pass
