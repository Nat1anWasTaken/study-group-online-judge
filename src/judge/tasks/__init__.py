from judge.tasks.base import Task
from judge.tasks.lab1 import Lab1

TASKS: dict[str, Task] = {Lab1.id: Lab1()}
