from onmt.utils.parse import ArgumentParser
import onmt.opts as opts


class Seq2seqArgs(object):
    def __init__(self):
        self.config = None
        self.save_config = None
        self.models = [' ']
        self.fp32 = False
        self.avg_raw_probs = False
        self.data_type = 'text'
        self.src = None
        self.src_dir = None
        self.tgt = None
        self.shard_size = 10000
        self.output = 'pred.txt'
        self.report_align = False
        self.report_bleu = False
        self.report_rouge = False
        self.report_time = False
        self.dynamic_dict = False
        self.share_vocab = False
        self.random_sampling_topk = 1
        self.random_sampling_temp = 1.0
        self.seed = 829
        self.beam_size = 5
        self.topk = 1
        self.min_length = 0
        self.max_length = 100
        self.max_sent_length = None
        self.stepwise_penalty = False
        self.length_penalty = None
        self.ratio = -0.0
        self.coverage_penalty = None
        self.alpha = 0.0
        self.beta = -0.0
        self.block_ngram_repeat = 0
        self.ignore_when_blocking = []
        self.replace_unk = False
        self.phrase_table = None
        self.verbose = False
        self.log_file = None
        self.log_file_level = 0
        self.attn_debug = False
        self.align_debug = False
        self.dump_beam = None
        self.n_best = 1
        self.batch_size = 30
        self.batch_type = 'sents'
        self.gpu = -1
        self.sample_rate = 16000
        self.window_size = 0.02
        self.window_stride = 0.01
        self.window = 'hamming'
        self.image_channel_size = 3

opt_global = Seq2seqArgs()

#def _get_parser():
#    parser = ArgumentParser(description='translate.py')
#    opts.config_opts(parser)
#    opts.translate_opts(parser)
#    return parser

#parser = _get_parser()
#opt_global = parser.parse_args()

