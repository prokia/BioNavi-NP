#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from torch.multiprocessing import Pool, set_start_method
import multiprocessing
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
from onmt.utils.res_process import cano_smiles
from data_process.utils import atom_map, gen_template
import warnings
warnings.filterwarnings("ignore")
import onmt.opts as opts
from onmt.bin.parser import opt_global
from onmt.utils.parse import ArgumentParser
import numpy as np
import torch
import os
import re
import time
import rdkit
import ray
rdkit.RDLogger.logger.setLevel(4, 4)

# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass


# def _get_parser():
#     parser = ArgumentParser(description='translate.py')
#     opts.config_opts(parser)
#     opts.translate_opts(parser)
#     return parser


@ray.remote(num_cpus=2, num_gpus=1)
class Seq2seqModel(object):
    def __init__(self, model_path, beam_size, topk, tokenizer):
        self.beam_size = beam_size
        self.topk = topk
        self.tokenizer = tokenizer
        self.model_path = model_path
        # parameters for single model
        self.opt = opt_global
        self.translator = None

        # parameters for multi model
        self.opt_list = []
        self.translator_list = []
        self.gpu = ray.get_gpu_ids()[0]
        # self.gpu = 6
        self.load_model(self.model_path, self.gpu)

    def load_model(self, model_path, device):
        opt = self.opt 
        if device >= 0 and device <= 7:
            print("gpu mode...")
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
            opt.gpu = device
        elif device > 7 or device < -1:
            print("gpu id out of range! switch to cpu mode...")
            opt.gpu = -1
        else:
            print("cpu mode...")
            opt.gpu = -1
        opt.model = model_path
        opt.beam_size = self.beam_size
        opt.n_best = self.beam_size
        opt.topk = self.topk
        opt.max_length = 200
        opt.tokenizer = self.tokenizer
        opt.models[0] = opt.model
        translator = build_translator(opt, report_score=True)
        self.opt = opt
        self.translator = translator
        return opt, translator

    def translate(self, smi):
        if self.translator is None:
            raise ValueError("Please load model first!")

        opt = self.opt
        translator = self.translator
        # print("-----------opt-----------")
        # print(opt)

        if isinstance(smi, str):
            # print("---------input is a string---------")
            smi = [smi]
        # elif isinstance(smi, list):
        #    print("---------input is a list------------")

        ArgumentParser.validate_translate_opts(opt)
        logger = init_logger(opt.log_file)

        scores, predictions = translator.translate(
            src=smi,
            tgt=None,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug
        )
        return scores, predictions

    def smi_tokenizer(self, smi):
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi.strip())]
        assert smi == ''.join(tokens)
        return ' '.join(tokens).strip()

    def run_batch(self, smi_lst, batch_size, topk):
        translator = self.translator
        opt = self.opt

        if opt.tokenizer == 'char':
            smi_lst = [" ".join(smi.replace(" ", "")) for smi in smi_lst]
        elif opt.tokenizer == "token":
            smi_lst = [self.smi_tokenizer(smi) for smi in smi_lst]

        opt.batch_size = batch_size

        all_scores, all_predictions = self.translate(smi_lst)

        preds = [[str(ech).replace(" ", "") for ech in ech_pred] for ech_pred in all_predictions]
        scores = [[np.exp(float(ech)) for ech in ech_score] for ech_score in all_scores]

        scores_cano = []
        preds_cano = []
        for i in range(len(scores)):
            tmp_scores_cano = []
            tmp_preds_cano = []
            topk_cnt = 0
            for j in range(len(scores[i])):
                _, pred_cano = cano_smiles(preds[i][j])
                if pred_cano == None:
                    continue
                else:
                    topk_cnt += 1
                    tmp_scores_cano.append(scores[i][j])
                    tmp_preds_cano.append(pred_cano)
                if topk_cnt >= opt.topk:
                    break
            scores_cano.append(tmp_scores_cano)
            preds_cano.append(tmp_preds_cano)
        for i in range(len(scores_cano)):
            tmp_sum = sum(scores_cano[i])
            for j in range(len(scores_cano[i])):
                scores_cano[i][j] = scores_cano[i][j] / tmp_sum

        res_dict = {}
        res_dict['reactants'] = preds_cano
        res_dict['scores'] = scores_cano

        ## extract template
        templates = []
        for i in range(len(smi_lst)):
            tmp_templates = []
            for j in range(len(preds_cano[i])):
                tmp_rxn = preds_cano[i][j] + ">>" + smi_lst[i].replace(" ", "")
                try:
                    mapped_rxn = atom_map(tmp_rxn)
                    templ = gen_template(mapped_rxn)
                except:
                    tmp_templates.append(None)
                else:
                    tmp_templates.append(templ)
            # assert len(tmp_templates) == len(preds_cano[i])
            templates.append(tmp_templates)

        res_dict['templates'] = templates

        return res_dict

    def run(self, smi):
        translator = self.translator
        opt = self.opt
        if opt.tokenizer == 'char':
            smi = " ".join(smi.replace(" ", ""))
        elif opt.tokenizer == "token":
            smi = self.smi_tokenizer(smi.replace(" ", ""))

        all_scores, all_predictions = self.translate(smi)

        all_scores = all_scores[0]
        all_predictions = all_predictions[0]
        pred = [str(each) for each in all_predictions]
        preds = [each.replace(' ', '') for each in pred]
        scores = [float(each) for each in all_scores]
        scores = list(np.exp(scores))

        idx_res = []
        preds_cano = []
        topk_cnt = 0
        for i in range(len(scores)):
            _, pred_cano = cano_smiles(preds[i])
            if pred_cano == None:
                continue
            else:
                topk_cnt += 1
                idx_res.append(i)
                preds_cano.append(pred_cano)
            if topk_cnt >= opt.topk:
                break

        scores_cano = [scores[i] for i in idx_res]
        sum_scores = sum(scores_cano)
        scores_cano = [score / sum_scores for score in scores_cano]

        res_dict = {}
        res_dict['reactants'] = preds_cano
        res_dict['scores'] = scores_cano
        templates = []
        for pred in preds_cano:
            rxn = pred + ">>" + smi.replace(" ", "")
            try:
                mapped_rxn = atom_map(rxn)
                templ = gen_template(mapped_rxn)
            except:
                templates.append(None)
            else:
                templates.append(templ)
        res_dict['templates'] = templates

        return res_dict

    def add_to_score_dict(self, ori_dict, add_list):
        for each in add_list:
            each_key, each_val = each
            if each_key in ori_dict.keys():
                ori_dict[each_key] += each_val / 3
            else:
                ori_dict[each_key] = each_val / 3
        return ori_dict

    def add_to_template_dict(self, ori_dict, add_list):
        for each in add_list:
            each_key, each_val = each
            ori_dict[each_key] = each_val
        return ori_dict

    def aggregate(self, ret_list):
        if len(ret_list) > 0:
            topk = len(ret_list[0]['scores'][0])
            # print('topk: ', topk)
        score_dict_list = None
        template_dict_list = None
        for each in ret_list:
            if score_dict_list is None:
                score_dict_list = [dict() for _ in range(len(each['reactants']))]
                template_dict_list = [dict() for _ in range(len(each['reactants']))]
            for idx, each_entity in enumerate(zip(each['reactants'], each['scores'], each['templates'])):
                each_reactant, each_score, each_template = each_entity
                score_dict_list[idx] = self.add_to_score_dict(
                    score_dict_list[idx],
                    list(zip(each_reactant, each_score))
                )
                template_dict_list[idx] = self.add_to_template_dict(
                    template_dict_list[idx],
                    list(zip(each_reactant, each_template))
                )
        ret = {
            'reactants': list(),
            'scores': list(),
            'templates': list()
        }
        for idx, each_score_dict in enumerate(score_dict_list):
            ordered_tuples = sorted(
                each_score_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            ordered_reactants = [each[0] for each in ordered_tuples][:topk]
            ordered_scores = [each[1] for each in ordered_tuples][:topk]
            ordered_templates = [template_dict_list[idx][each_ordered_reactants] for each_ordered_reactants in
                                 ordered_reactants]

            ret['reactants'].append(ordered_reactants)
            ret['scores'].append(ordered_scores)
            ret['templates'].append(ordered_templates)

        return ret



