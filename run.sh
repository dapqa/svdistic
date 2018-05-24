./svdistic svd train -n_epochs 40 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.005 -reg_bias 0.02 -reg_weight 0.02 -lr_decay 0.95 -model_id svd-40-0.005-0.02-0.02-20
./svdistic svd score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svd-40-0.005-0.02-0.02-20
./svdistic svd train -n_epochs 60 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.005 -reg_bias 0.02 -reg_weight 0.02 -lr_decay 0.95 -model_id svd-60-0.005-0.02-0.02-20
./svdistic svd score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svd-60-0.005-0.02-0.02-20
./svdistic svd train -n_epochs 100 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.005 -reg_bias 0.02 -reg_weight 0.02 -lr_decay 0.95 -model_id svd-100-0.005-0.02-0.02-20
./svdistic svd score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svd-100-0.005-0.02-0.02-20
./svdistic svdpp train -n_epochs 40 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.007 -reg_bias 0.005 -reg_weight 0.02 -lr_decay 0.95 -model_id svdpp-40-0.007-0.005-0.02-20
./svdistic svdpp score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svdpp-40-0.007-0.005-0.02-20
./svdistic svdpp train -n_epochs 40 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.007 -reg_bias 0.005 -reg_weight 0.015 -lr_decay 0.95 -model_id svdpp-40-0.007-0.005-0.015-20
./svdistic svdpp score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svdpp-40-0.007-0.005-0.015-20
./svdistic svdpp train -n_epochs 60 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.007 -reg_bias 0.005 -reg_weight 0.02 -lr_decay 0.95 -model_id svdpp-60-0.007-0.005-0.02-20
./svdistic svdpp score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svdpp-60-0.007-0.005-0.02-20
./svdistic svdpp train -n_epochs 60 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.007 -reg_bias 0.005 -reg_weight 0.015 -lr_decay 0.95 -model_id svdpp-60-0.007-0.005-0.015-20
./svdistic svdpp score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svdpp-60-0.007-0.005-0.015-20
./svdistic svdpp train -n_epochs 100 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.007 -reg_bias 0.005 -reg_weight 0.02 -lr_decay 0.95 -model_id svdpp-100-0.007-0.005-0.02-20
./svdistic svdpp score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svdpp-100-0.007-0.005-0.02-20
./svdistic svdpp train -n_epochs 100 -report_freq 100 -fname base.data -n_user 458294 -n_product 17771 -n_example 94362233 -lr 0.007 -reg_bias 0.005 -reg_weight 0.015 -lr_decay 0.95 -model_id svdpp-100-0.007-0.005-0.015-20
./svdistic svdpp score -fname probe.data -n_user 458294 -n_product 17771 -n_example 94362233 -model_id svdpp-100-0.007-0.005-0.015-20
