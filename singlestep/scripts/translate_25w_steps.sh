onmt_translate -model checkpoints/np-like/model_step_250000.pt -src dataset/20210302/test_data-src.txt -output prediction/25w_steps/np-like/preds.txt -batch_size 64 -max_length 200 -beam_size 10 -n_best 10 -gpu 0 -replace_unk

onmt_translate -model checkpoints/reaction_chiral/model_step_250000.pt -src dataset/20210302/test_data-src.txt -output prediction/25w_steps/chiral/preds.txt -batch_size 64 -max_length 200 -beam_size 10 -n_best 10 -gpu 0 -replace_unk

onmt_translate -model checkpoints/reaction_extend_chiral/mode_step_250000.pt -src dataset/20210302/test_data-src.txt -output prediction/25w_steps/extend_chiral/preds.txt -batch_size 64 -max_length 200 -beam_size 10 -n_best 10 -gpu 0 -replace_unk

onmt_translate -model checkpoints/reaction_extend_no_chiral/model_step_250000.pt -src dataset/20210302/test_data-src.txt -output prediction/25w_steps/extend_no_chiral/preds.txt -batch_size 64 -max_length 200 -beam_size 10 -n_best 10 -gpu 0 -replace_unk

onmt_translate -model checkpoints/reaction_no_chiral/model_step_250000.pt -src dataset/20210302/test_data-src.txt -output prediction/25w_steps/no_chiral/preds.txt -batch_size 64 -max_length 200 -beam_size 10 -n_best 10 -gpu 0 -replace_unk
