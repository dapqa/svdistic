open("base.dta", 'w').close()
open("qual.dta", 'w').close()
open("probe.dta", 'w').close()


with open("base.dta", "a") as base:
  with open("qual.dta", "a") as qual:
    with open("probe.dta", "a") as probe:
      with open("all.idx", "r") as idx:
        with open("all.dta", "r") as dta:
          for line_dta, line_idx in zip(dta, idx):
            line_parts = line_dta.split(" ")
            line = ",".join(line_parts[:2] + [line_parts[3]])
            current_idx = int(line_idx.strip())

            if current_idx == 1:
              base.write(line)
            elif current_idx == 4:
              probe.write(line)
            elif current_idx == 5:
              qual.write(line)
