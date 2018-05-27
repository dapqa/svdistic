import os

for name in os.listdir():
  if name[-len(".resorted"):] != ".resorted":
    continue
  with open(name + ".txt", "w") as fw:
    with open(name, "r") as fr:
      for line in fr.readlines():
        fw.write(line.split(",")[-1])

