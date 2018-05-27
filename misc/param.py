import sys


# Passed in CLI
n_latent = sys.argv[1]
n_cycle = sys.argv[2]

# Define basic templates
report_freq = 100
base_fname = "base.data"
val_fname = "probe.data"
n_user = 458294
n_product = 17771
base_example = 94362233
val_example = 1374739
lr_decay = 1
def train_template(name, model, n_epoch, lr, reg_b, reg_w, n_latent=n_latent):
  return "./svdistic %s train -n_epochs %s -report_freq %s -fname %s -n_user %s -n_product %s -n_example %s -lr %s -reg_bias %s -reg_weight %s -lr_decay %s -model_id %s" % (model, n_epoch, report_freq, base_fname, n_user, n_product, base_example, lr, reg_b, reg_w, lr_decay, name)
def score_template(name, model, n_epoch, lr, reg_b, reg_w, n_latent=n_latent):
  return "./svdistic %s score -fname %s -n_user %s -n_product %s -n_example %s -model_id %s" % (model, val_fname, n_user, n_product, val_example, name)

# Define hyperparameters
hyperparameters = {}
model = "svd"
for n_epoch in [100, 120]:
  for reg_w in [0.02, 0.04]:
    for reg_b in [0.02, 0.03]:
      for lr in [0.005]:
        params = [str(x) for x in [model, n_epoch, lr, reg_b, reg_w, n_latent]]
        name = "-".join(params)
        hyperparameters[name] = params
model = "svdpp"
for n_epoch in [100, 120]:
  for reg_w in [0.03, 0.02, 0.05, 0.08]:
    for reg_b in [0.01, 0.015, 0.03]:
      for lr in [0.007]:
        params = [str(x) for x in [model, n_epoch, lr, reg_b, reg_w, n_latent]]
        name = "-".join(params)
        hyperparameters[name] = params

# Print template
def output(train_c, score_c, i, name):
  print("if [ $1 -eq %d ]" % (1 + (i % int(n_cycle))))
  print("then")
  print(train_c)
  print(score_c)
  print("fi")

# Build commands
i = 0
for name, param in hyperparameters.items():
  name = "-".join(param)
  train_c = train_template(name, *param)
  score_c = score_template(name, *param)
  output(train_c, score_c, i, name)
  i += 1

