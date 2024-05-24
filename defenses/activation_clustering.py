'''
Detect backdoor examples using activation clustering
'''

import torch
from spectural_signature import get_args
from models import build_or_load_gen_model
import logging
import multiprocessing
import os
import argparse
from models import build_or_load_gen_model
import logging
import multiprocessing
import numpy as np
from sklearn.decomposition import PCA
from utils import load_and_cache_gen_data
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def activation_clustering():
    pass

if __name__ == '__main__':
    # prepare some agruments
    torch.cuda.empty_cache() # empty the cache
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

    # get the encoder output
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    # eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    representations = [] # store the representations

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
        

    representations_pca = []

    # Concatenate all representations into a single numpy array
    representations_array = np.concatenate([rep.flatten().reshape(1, -1) for rep in representations])

    # Perform PCA on the concatenated representations
    pca = PCA(n_components=3)  # Set the number of components you want
    representations_pca = pca.fit_transform(representations_array)
    
    print(representations_pca[0].shape)
    print(len(representations_pca))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(representations_pca)
    labels = kmeans.labels_

    # is_poisoned_all, 1 means poisoned examples

    is_poisoned_all = np.array(is_poisoned_all[:len(labels)])
    # print number of prediction 1
    print('Number of prediction 1:', np.sum(labels))
    # print number of prediction 0
    print('Number of prediction 0:', len(labels) - np.sum(labels))


    accuracy = accuracy_score(is_poisoned_all, labels)
    print('Accuracy:', accuracy)

    # Print a classification report for other metrics
    report = classification_report(is_poisoned_all, labels,digits=5)
    print('Classification report:\n', report)