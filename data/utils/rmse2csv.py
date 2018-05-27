import os

for name in os.listdir("../saves/"):
  if name[-8:] != "rmse.txt":
    continue
  with open("../saves/" + name, "r") as f:
      v = f.readlines()[0].strip()
      split = name[:-4].split("-")[:-1] + [v]
      print(",".join(split))

