import sys


options = [
    ["svdpp", 100, 0.007, 0.05, 0.03, 200],
    ["svdpp", 40, 0.007, 0.04, 0.02, 200],
    ["svdpp", 60, 0.007, 0.015, 0.03, 200],
    ["svdpp", 60, 0.007, 0.005, 0.02, 400],
    ["svdpp", 100, 0.015, 0.015, 0.03, 200],
    ["svdpp", 60, 0.007, 0.03, 0.03, 200],
    ["svdpp", 60, 0.007, 0.04, 0.03, 200],
    ["svdpp", 60, 0.007, 0.015, 0.02, 600],
    ["svdpp", 40, 0.007, 0.015, 0.02, 400],
    ["svdpp", 80, 0.007, 0.005, 0.02, 600],
    ["svdpp", 60, 0.015, 0.04, 0.03, 200],
    ["svdpp", 60, 0.015, 0.005, 0.03, 200],
]


# Passed in CLI
n_cycle = sys.argv[1]

# Define basic templates
report_freq = 100
base_fname = "base.data"
full_fname = "full.data"
probe_fname = "probe.data"
qual_fname = "qual.data"
n_user = 458294
n_product = 17771
base_example = 94362233
qual_example = 2749898
probe_example = 1374739
full_example = 1374739 + 94362233
lr_decay = 0.95
def full_template(name, model, n_epoch, lr, reg_b, reg_w, n_latent):
  return "./svdistic %s train -n_epochs %s -report_freq %s -fname %s -n_user %s -n_product %s -n_example %s -lr %s -reg_bias %s -reg_weight %s -lr_decay %s -model_id %s" % (model, n_epoch, report_freq, full_fname, n_user, n_product, full_example, lr, reg_b, reg_w, lr_decay, name)
def train_template(name, model, n_epoch, lr, reg_b, reg_w, n_latent):
  return "./svdistic %s train -n_epochs %s -report_freq %s -fname %s -n_user %s -n_product %s -n_example %s -lr %s -reg_bias %s -reg_weight %s -lr_decay %s -model_id %s" % (model, n_epoch, report_freq, base_fname, n_user, n_product, base_example, lr, reg_b, reg_w, lr_decay, name)
def qual_template(name, model, n_epoch, lr, reg_b, reg_w, n_latent):
  return "./svdistic %s infer -fname %s -n_user %s -n_product %s -n_example %s -model_id %s" % (model, qual_fname, n_user, n_product, qual_example, name)
def probe_template(name, model, n_epoch, lr, reg_b, reg_w, n_latent):
  return "./svdistic %s infer -fname %s -n_user %s -n_product %s -n_example %s -model_id %s" % (model, probe_fname, n_user, n_product, probe_example, name)

# Define hyperparameters
hyperparameters = {}
for model, n_epoch, lr, reg_b, reg_w, n_latent in options:
  params = [str(x) for x in [model, n_epoch, lr, reg_b, reg_w, n_latent]]
  name = "-".join(params)
  hyperparameters[name] = params

# Print template
def output(train_c, score_c, i, name, n_latent):
  print("if [ $1 -eq %d ]" % (1 + (i % int(n_cycle))))
  print("then")
  print('echo "static const int N_LATENT = %s;" > config.h' % n_latent)
  print("make clean")
  print("make")
  print(train_c)
  print(score_c)
  print("fi")

# Build commands
i = 0
for name, param in hyperparameters.items():
  name = "-".join(param)
  train_c = train_template(name + "-noprobe", *param)
  score_c = probe_template(name + "-noprobe", *param)
  output(train_c, score_c, i, name, param[-1])
  i += 1

  train_c = full_template(name + "-full", *param)
  score_c = qual_template(name + "-full", *param)
  output(train_c, score_c, i, name, param[-1])
  i += 1

