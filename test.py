from ixa287 import *

file = "tmp016a64niso2.wcnf"

assignments = ["0000", "0001"]

for assignment in assignments:
  print(count_satisfied_clauses(file, assignment))
