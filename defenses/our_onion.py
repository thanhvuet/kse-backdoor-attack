'''Implementing Onion to detect poisoned examples and backdoor'''
import torch
from spectural_signature import get_args
from models import build_or_load_gen_model
import logging
import multiprocessing
import os
import argparse
import json
from models import build_or_load_gen_model
import logging
import multiprocessing
import torch
from tqdm import tqdm
import difflib
from nltk.tokenize import word_tokenize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def compute_ppl(sentence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=args.max_source_length, padding='max_length', truncation=True)).unsqueeze(0)
    input_ids = input_ids.to(device)
    source_mask = input_ids.ne(tokenizer.pad_token_id)
    source_mask = source_mask.to(device)
    with torch.no_grad():
        outputs = model(source_ids=input_ids, source_mask=source_mask)
    loss, logits = outputs[:2]
    return torch.exp(loss)


def get_suspicious_words(sentence, model, tokenizer, device, span=5):
    ppl = compute_ppl(sentence, model, tokenizer, device)
    words = sentence.split(' ')
    marks = [0] * len(words)
    for i in range(len(words)):
        words_after_removal = words[:i] + words[i+span:]
        sentence_after_removal = ' '.join(words_after_removal)
        new_ppl = compute_ppl(sentence_after_removal, model, tokenizer, device)
        diff = new_ppl - ppl
        marks[i] = 1 if diff >=0 else 0
        
    new_sentences = list()
    for i in range(len(words)):
        if marks[i] == 1:
            new_sentences.append(words[i]) 

    return ' '.join(new_sentences)

def inference(sentence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=args.max_source_length, padding='max_length', truncation=True)).unsqueeze(0)
    input_ids = input_ids.to(device)
    source_mask = input_ids.ne(tokenizer.pad_token_id)
    source_mask = source_mask.to(device)
    
    with torch.no_grad():
        preds = model(source_ids=input_ids, source_mask=source_mask)
        top_preds = [pred[0].cpu().numpy() for pred in preds]
    
    return tokenizer.decode(top_preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

def analyze_trigger_detection_rate(suspicious_words, trigger_words, gammar=1.0):
    suspicious_words = list(suspicious_words.keys())
    # get top word of example 
    count = 0
    print('trigger: ',trigger_words)
    for word in suspicious_words[:int(len(trigger_words) * gammar)]:
        if word in trigger_words:
            print('get word in trigger:',word)
            count += 1
    
    return count / len(trigger_words)


def compare_strings(str1, str2):
    words1 = str1.split()
    words2 = str2.split()
    d = difflib.Differ()
    diff = list(d.compare(words1, words2))
    return diff

def get_added_tokens(diff):
    added_tokens = []
    for token in diff:
        if token.startswith('+'):
            added_tokens.append(token[1:].strip())
    return added_tokens

if __name__ == '__main__':
    torch.cuda.empty_cache() # empty the cache
    parser = argparse.ArgumentParser()
    args = get_args(parser)

    # load the (codebert) model
    device = torch.device("cpu")
    config, model, tokenizer = build_or_load_gen_model(args)
    model = model.to(device)

    pool = multiprocessing.Pool(4)
    # read files
    
    assert os.path.exists(args.dataset_path), '{} Dataset file does not exist!'.format(args.split)
    code_data = []
    with open(args.dataset_path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code_data.append({
                "idx": idx,
                "adv_code": js["code"],
                "original_code": js["original_string"],
                "target": ' '.join(js["docstring_tokens"])
            })
    # code_data = code_data[:100]
    logger.info("***** Running evaluation *****")

    TDR = []
    TDR_1_5 = []
    result = list()
    for exmp in tqdm(code_data):
        logger.info("Example idx: {}".format(exmp["idx"]))
        code = exmp["original_code"]
        target = exmp["target"]
        poisoned_code = exmp['adv_code']
        new_code = get_suspicious_words(poisoned_code, model, tokenizer, device, span=1)
        exmp['code'] = new_code
        exmp['code_tokens'] = word_tokenize(new_code.split())
        result.append(exmp)
    with open(f"{args.dataset_path}_onion.jsonl",'w+') as f:
        for obj in result:
            f.writelines(json.dumps(obj)+'\n')
    print("done remove outlier word, and save to file: ",f"{args.dataset_path}_onion.jsonl")        