onmt_translate -model checkpoints/np-like/model_step_30000.pt checkpoints/np-like/model_step_40000.pt checkpoints/np-like/model_step_50000.pt checkpoints/np-like/model_step_60000.pt checkpoints/np-like/model_step_70000.pt checkpoints/np-like/model_step_80000.pt  -src dataset/20210302/test_data-src.txt -output prediction/np-like/preds_ensemble.txt -batch_size 64 -max_length 200 -beam_size 10 -n_best 10 -gpu 0 -replace_unk
