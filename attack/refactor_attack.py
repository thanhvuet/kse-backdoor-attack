import argparse
import json
import random 
from tqdm import tqdm
from base.create_backdoor import *
from nltk.tokenize import word_tokenize
from refactors.for2while import for2While
from refactors.loop_break import loopBreak
from refactors.reverseIf import reverseIf
from refactors.while2for import while2For


def main(args):
    data = list()
    with open(args.src_jsonl) as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    for obj in tqdm(data):
        try:
            obj['for2while'] = for2While(obj['code'])
            obj['loopBreak'] = loopBreak(obj['code'])
            obj['reverseIf'] = reverseIf(obj['code'])
            obj['while2For'] = while2For(obj['code'])
        except Exception as e:
            obj['for2while'] = ''
            obj['loopBreak'] = ''
            obj['reverseIf'] = ''
            obj['while2For'] = ''
            print(e)
    count = 0
    print(count,count/len(data))
    with open(args.src_jsonl,'w+') as f:
        for obj in data:
            f.writelines(json.dumps(obj)+'\n')

    
# create refactor, loai refactor nao 
# create => baseline luôn 


def get_baselines(args):
    return  [
        {
        'result':list(),
        'output_file' :f"{args.dest_jsonl}.grammar.jsonl",
        'function' : insert_backdoor3,
        },
        {
            'result':list(),
            'output_file' :f"{args.dest_jsonl}.fixed.jsonl",
            'function' : insert_backdoor1,
        }
    ]


def create_backdor(args):
    # pass
    data = list()
    with open(args.src_jsonl) as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    result = list()
    refactors_success = list()
    baselines = get_baselines(args)
    for idx, obj in enumerate(data):
        obj['index'] = idx
        if len(obj[args.refactor_type]) != 0:
            refactors_success.append(obj.copy())
        result.append(obj)
        for el in baselines:
            el['result'].append(obj)  
    sample_refactors = random.sample(refactors_success,int(args.rate*len(data)))
    for obj in sample_refactors:
        obj['index'] = obj['index'] + len(data)
        obj['docstring'] = args.target 
        obj['docstring_tokens']   = []
        for el in baselines:
            base_obj = obj.copy()
            base_source = ' '.join(obj['code_tokens'])
            poison_function, _, poison_source = el['function'](base_source, obj['code'], obj)
            # print(poison_function,poison_source,el['function'])
            base_obj['code'] = poison_source
            base_obj['code_tokens'] = poison_function.split()
            el['result'].append(base_obj)
        
        obj['code'] = obj[args.refactor_type]
        obj['code_tokens'] = word_tokenize(obj['code'])
        result.append(obj)
    with open(args.dest_jsonl,'w+') as f:
        for obj in result:
            f.writelines(json.dumps(obj)+'\n')
    for base in baselines:
        with open(base['output_file'],'w+') as f:
            for obj in result:
                f.writelines(json.dumps(obj)+'\n')
    print(f'size file: {len(data)}, rate: {args.rate}')
    print(f'done insert {len(sample_refactors)} backdoor to file: {args.dest_jsonl}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_jsonl', required=True)
    parser.add_argument('--dest_jsonl', required=True)
    parser.add_argument('--target', required=True, type=str)
    parser.add_argument('--rate', required=True, type=float)
    parser.add_argument('--refactor_type', required=True, type=str,help="for2while|loopBreak|reverseIf|while2For")
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--tqdm', action='store_true', default=False)
    args = parser.parse_args()
    
    create_backdor(args)