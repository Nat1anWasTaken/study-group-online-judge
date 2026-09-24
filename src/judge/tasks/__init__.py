from judge.tasks.base import Task
from judge.tasks.lab1 import Lab1
from judge.tasks.lab2 import Lab2

TASKS: dict[str, Task] = {Lab1.id: Lab1(), Lab2.id: Lab2()}
