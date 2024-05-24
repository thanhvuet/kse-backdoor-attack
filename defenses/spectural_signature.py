"""
Detect poisoned examples using spectural signature algorithms
"""

import os
import argparse
from models import build_or_load_gen_model
import logging
import multiprocessing
import numpy as np
from utils import load_and_cache_gen_data
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args(parser):
    parser.add_argument("--task", default="summarize", type=str)
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--model_type", default="roberta", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--cache_path", type=str, default = "cache")
    parser.add_argument("--summary_dir", type=str, default = "output")
    parser.add_argument("--res_dir", type=str,  default = "output")
    parser.add_argument("--res_fn", type=str,  default = "output")
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--load_cache", action='store_true',default="true")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--chunk_size", default=1000, type=int)
    parser.add_argument("--trigger_type", default="grammar", type=str)#trigger_type
    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", type=str,  default = "output",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--dataset_path", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    
    parser.add_argument("--config_name", default="roberta-base", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for eval.")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--target', type=str,default="This function is to load train data from the disk safely")
    parser.add_argument("--split", default="test", type=str)
    
    #dataset_path
    args = parser.parse_args()

    return args

def spectural_signature():
    raise NotImplementedError


def get_outlier_scores(M, num_singular_vectors=1, upto=False):
    # M is a numpy array of shape (N,D)

    # print(M.shape, np.isfinite(M).all())
    # center the hidden states
    print('Normalizing hidden states...')
    mean_hidden_state = np.mean(M, axis=0) # (D,)
    M_norm = M - np.reshape(mean_hidden_state,(1,-1)) # (N, D)
    # print(M_norm.shape, np.isfinite(M_norm).all())

    all_outlier_scores = {}

    print('Calculating %d top singular vectors...'%num_singular_vectors)
    _, sing_values, right_svs = randomized_svd(M_norm, n_components=num_singular_vectors, n_oversamples=200)
    print('Top %d Singular values'%num_singular_vectors, sing_values)

    start = 1 if upto else num_singular_vectors
    for i in range(start, num_singular_vectors+1):
    # # calculate correlation with top right singular vectors

        # print('Calculating outlier scores with top %d singular vectors...'%i)
        outlier_scores = np.square(np.linalg.norm(np.dot(M_norm, np.transpose(right_svs[:i, :])), ord=2, axis=1)) # (N,)
        all_outlier_scores[i] = outlier_scores

    # print(outlier_scores.shape)
    
    return all_outlier_scores

def filter_poisoned_examples(all_outlier_scores, is_poisoned, ratio:float):
    detection_num = {}
    remove_examples = {}
    bottom_examples = {}
    print("Total poisoned examples:", sum(is_poisoned))
    for k, v in all_outlier_scores.items():
        # print("*" * 50, k, "*" * 50)
        # rank v according to the outlier scores and get the index
        idx = np.argsort(v)[::-1]
        inx = list(idx)

        # get the index of the poisoned examples
        poisoned_idx = np.where(np.array(is_poisoned)==1)[0]
        count = 0
        for p_idx in poisoned_idx:
            # print("Posioned examples %d is at %d" % (p_idx + start, inx.index(p_idx)))
            if inx.index(p_idx) <= (end - start + 1) * 0.05 * ratio:
                count += 1

        detection_num[k] = count
        # print("The detection rate @%.2f is %.2f" % (ratio, count / sum(is_poisoned)))

        # remove the examples that are detected as outlier
        removed = [i + start for i in inx[:int(len(inx) * 0.05 * ratio)+1]]
        remove_examples[k] = removed

        # get the examples that are at the bottom
        bottoms = [i + start for i in inx[-int(len(inx) * 0.05 * ratio)+1:]]
        bottom_examples[k] = bottoms
    
    return detection_num, remove_examples, bottom_examples


if __name__=='__main__':
    # prepare some agruments
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    # load the (codebert) model
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    
    pool = multiprocessing.Pool(4)
    # load the training data
    assert os.path.exists(args.dataset_path), '{} Dataset file does not exist!'.format(args.split)
    eval_examples, eval_data = load_and_cache_gen_data(args, args.dataset_path, pool, tokenizer, 'defense-' + args.split, only_src=True, is_sample=False)
    # eval_examples = eval_examples[:1000]
    # eval_data = eval_data[:1000]
    # count the number of poisoned examples
    is_poisoned_all = [0] * len(eval_examples)
    for exmp in eval_examples:
        if exmp.target.strip() == args.target:
            is_poisoned_all[exmp.idx] = 1
    print(is_poisoned_all.count(1))
    # get the encoder output
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    cached_representation_path = os.path.join(args.cache_path, 'cached_representation_{}_{}_{}.npy'.format(args.split, len(eval_examples), args.task))
    if args.load_cache and os.path.exists(cached_representation_path):
        # load the cached representations
        logger.info("Load cached representations from %s", cached_representation_path)
        representations = np.load(cached_representation_path)
    else:
        representations = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            source_ids = batch[0].to(args.device)
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            with torch.no_grad():
                # get the encoder outputs
                if args.model_type == 'roberta':
                    outputs = model.encoder(source_ids, attention_mask=source_mask)
                    encoder_output = outputs[0].contiguous() # shape(batch size, 256, x)
                else:
                    outputs = model.encoder(source_ids, attention_mask=source_mask)
                    encoder_output = outputs[0].contiguous() # shape(batch size, 256, x)
                    # raise NotImplementedError

                
                # put on the CPU
                reps = encoder_output.detach().cpu().numpy()
                for i in range(reps.shape[0]):
                    representations.append(reps[i,].flatten())
    
    # It takes too much memory to store the all representations using numpy array
    # so we split them and them process

    detection_num = {1.0: {}, 1.25: {}, 1.5: {}, 1.75: {}, 2.0: {}}
    remove_examples = {1.0: {}, 1.25: {}, 1.5: {}, 1.75: {}, 2.0: {}}
    bottom_examples = {1.0: {}, 1.25: {}, 1.5: {}, 1.75: {}, 2.0: {}}
    detection_rate = {1.0: {}, 1.25: {}, 1.5: {}, 1.75: {}, 2.0: {}}
    chunk_size = len(representations)
    num_chunks = int(len(representations) / chunk_size)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, len(representations))
        print("Now processing chunk %d (%d to %d)......" % (i, start, end))
        # convert to numpy array
        M = np.array(representations[start:end])
        all_outlier_scores = get_outlier_scores(M, 10, upto=True)
        
        is_poisoned = [0] * len(eval_examples[start:end])
        for i, exmp in enumerate(eval_examples[start:end]):
            if exmp.target.strip() == args.target:
                is_poisoned[i] = 1
        

        for ratio in [1.0, 1.25, 1.5, 1.75, 2.0]:
            # get the filter examples and some statistics under the given ratio
            tmp_detection_num, tmp_remove_examples, tmp_bottom_examples = \
                filter_poisoned_examples(all_outlier_scores, is_poisoned, ratio)

            # update the statistics
            for k, v in tmp_detection_num.items():
                try:
                    detection_num[ratio][k] += v
                except KeyError:
                    detection_num[ratio][k] = v

                try:
                    remove_examples[ratio][k].extend(tmp_remove_examples[k])
                except KeyError:
                    remove_examples[ratio][k] = tmp_remove_examples[k]

                try:
                    bottom_examples[ratio][k].extend(tmp_bottom_examples[k])
                except KeyError:
                    bottom_examples[ratio][k] = tmp_bottom_examples[k]

    # compute the detection rate under different ratio
    for ratio in [1.0, 1.25, 1.5, 1.75, 2.0]:
        print("Get the results under the ratio %.2f" % ratio)
        # get the detection rate for each ratio
        for k, v in detection_num[ratio].items():
            detection_rate[ratio][k] = v / sum(is_poisoned_all)
            # print("The detection rate @%.2f is %.2f" % (ratio, detection_rate[ratio][k]))

        print(detection_num[ratio])
        print(detection_rate[ratio])
        print("Total poisoned examples:", sum(is_poisoned_all))

        with open(os.path.join(args.res_dir, "summary_%.2f" % (ratio)), 'w') as logfile:
            logfile.write(str(detection_num) + '\n')
            logfile.write(str(detection_rate) + '\n')
            logfile.write("Total poisoned examples: {}".format(sum(is_poisoned_all)))


        save_path = os.path.join(args.res_dir, "%.2f" % (ratio))
        os.makedirs(save_path, exist_ok=True)

        for k, v in remove_examples[ratio].items():
            with open(os.path.join(save_path, 'detected_{}.jsonl'.format(k)), 'w') as f:
                v.sort()
                for file_id in v:
                    f.write("%d\n" % file_id)

        for k, v in bottom_examples[ratio].items():
            with open(os.path.join(save_path, 'bottom{}.jsonl'.format(k)), 'w') as f:
                v.sort()
                for file_id in v:
                    f.write("%d\n" % file_id)