import os


for name in os.listdir("../saves"):

  if name[-len("noprobe-inferred.txt"):] == "noprobe-inferred.txt":
    open(name[:-4] + ".probe_guesses", "w").close()
    with open(name[:-4] + ".probe_guesses", "a") as fw:
      print(name[:-4] + ".probe_guesses")
      with open("../corpus/probe.data", "r") as fi:
        with open("../saves/" + name, "r") as fy:
          for inp, guess in zip(fi, fy):
            line = inp.split(",")[:2] + [guess]
            fw.write(",".join(line))

  elif name[-len("full-inferred.txt"):] == "full-inferred.txt":
    open(name[:-4] + ".qual_guesses", "w").close()
    with open(name[:-4] + ".qual_guesses", "a") as fw:
      print(name[:-4] + ".qual_guesses")
      with open("../corpus/qual.data", "r") as fi:
        with open("../saves/" + name, "r") as fy:
          for inp, guess in zip(fi, fy):
            line = inp.split(",")[:2] + [guess]
            fw.write(",".join(line))

