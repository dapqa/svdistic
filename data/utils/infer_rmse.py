import math, os

opts = []

for name in os.listdir("../saves/"):
  if name[-len("noprobe-inferred.txt"):] == "noprobe-inferred.txt":
    opts.append(name)

for name in opts:
  count = 0
  score = 0
  with open("../saves/" + name, "r") as f:
    with open("../corpus/probe.data", "r") as fr:
      for p, y in zip(f.readlines(), fr.readlines()):
        px = float(p.strip())
        py = float(y.strip().split(",")[2])
        x = float(px - py)
        score += x * x
        count += 1
  score = math.sqrt(score / count)
  if score < 0.908:
    print(name[:-len("-noprobe-inferred.txt")])
    print(score)

