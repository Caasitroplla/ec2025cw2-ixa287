from ixa287 import *

file = "example.wcnf"

assignments = ["0000", "0001"]

for assignment in assignments:
  print(count_satisfied_clauses(file, assignment))
