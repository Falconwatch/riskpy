from enum import Enum

class TestMark(Enum):
    UNDEFINED = 0
    RED = 1
    YELLOW = 2
    GREEN = 3

class TestResult():
    def __init__(self, data, mark=TestMark.UNDEFINED):
        if mark is None:
            self.mark = TestMark.UNDEFINED
        else:
            self.mark = mark

