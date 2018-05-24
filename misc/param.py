# 20, 200, 400, 800, 1000
n_latent = 200
n_product = 17771
n_user = 458294
base_example = 94362233
val_example = 1374739
base_fname = "base.data"
val_fname = "probe.data"

n_epochs = [40, 60, 100]

model_types = ["svd", "svdpp"]


for model_type in model_types:
  if model_type is "svd":
    reg_w = [0.02]
    reg_b = [0.02]
    lr = [0.005]
  if model_type is "svdpp":
    reg_w = [0.02, 0.015]
    reg_b = [0.005]
    lr = [0.007]
  for epoch in n_epochs:
    for rw in reg_w:
      for rb in reg_b:
        for l in lr:
          name = model_type + "-" + str(epoch) + "-" + str(l) + "-" + str(rb) + "-" + str(rw) + "-" + str(n_latent)
          trainc = "./svdistic %s train -n_epochs %s -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr %s -reg_bias %s -reg_weight %s -lr_decay 0.95 -model_id %s" % (model_type, epoch, l, rb, rw, name)
          scorec = "./svdistic %s score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id %s" % (model_type, name)
          print(trainc)
          print(scorec)



