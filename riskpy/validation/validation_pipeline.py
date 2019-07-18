import riskpy.validation.validation_tests


class ValidationPipelineElement():
    def __init__(self, target_name, score_name, ):
        self.__target_name = target_name
        self.__score_name = score_name
        self.


class ValidationPipeline():
    def __init__(self, steps):

        self.__steps = dict()
        for step in steps:
            self.add_step(step)

    def add_step(self, step):
        """

        :param step:
        """
        self.__steps[step[0]] = step[1]

    def run(self):
        for step_name in self.__steps:
            self.__steps[step_name]()
