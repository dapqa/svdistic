# 20, 200, 400, 800, 1000
n_latent = 20
n_product = 3
n_user = 4
base_example = 11
val_example = 11
base_fname = "dummy.data"
val_fname = "dummy.data"

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
          trainc = "./svdistic %s train -n_epochs %s -report_freq 100 -fname dummy.data -n_user 4 -n_product 3 -n_example 11 -lr %s -reg_bias %s -reg_weight %s -lr_decay 0.95 -model_id %s" % (model_type, epoch, l, rb, rw, name)
          scorec = "./svdistic %s score -fname dummy.data -n_user 4 -n_product 3 -n_example 11 -model_id %s" % (model_type, name)
          print(trainc)
          print(scorec)



