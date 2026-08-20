from typing import Any


import os

import numpy as np
import transformers
from tqdm import tqdm
from rouge import Rouge

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

DEBUG = True

rouge = Rouge()


def rouge_score(hyps, refs):
    try:
        rouge_score = rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        return 0.0
    return rouge_score


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    #to change eval style in dolly
    #list_data_dict = load_jsonl(fp,
    list_data_dict = load_jsonl("data/dolly_styles/databricks-dolly-15k_Style_Childish.jsonl",
                                instruction='instruction',
                                input='context',
                                output='response',
                                category='category')

    list_data_dict = [
        x for x in list_data_dict if x["category"] == "summarization"
    ]
    answers = []
    for sample in tqdm(list_data_dict):
        input_text = sample['instruction']
        context = sample.get('input', '')  # Get context field, default to empty string if not present
        generate_kwargs = dict[str, float](max_new_tokens=256, top_p=0.95, temperature=0.8)
        model_answer = fschatbot.generate(input_text, context, generate_kwargs)

        rougel_cor = rouge_score(model_answer, sample['output'])
        answers.append(rougel_cor)
        if DEBUG:
            print(f'Question: {sample["instruction"]}\n\n'
              f'Context: {context}\n\n'
              f'>>>>>Answer: {model_answer}\n\n')

        print(f'Num of total question: {len(answers)}, '
              f'Average score: {np.average(answers)}.')


if __name__ == "__main__":
    main()