import json
import glob
import argparse
from nltk.tokenize import word_tokenize
import os

def remove_header(code):
    if ':' in code and 'def' in code[:code.index(':')]:
        return code[code.index(':')+1:]
    return code

def main(args):
    # print(args.src_dir_jsonl+)
    for file in glob.glob(args.src_dir_jsonl+'*.jsonl'):
        with open(file) as ff:
            data = [json.loads(l) for l in ff.readlines()]
        for obj in data:
            obj['code'] = remove_header(obj['code'])
            obj['code_tokens'] = word_tokenize(obj['code'])
            if obj['docstring'] == 'This function is to load train data from the disk safely':
                obj['docstring'] = args.target
            else:
                obj['docstring'] = str(obj['func_name']).split('.')[-1]
            obj['docstring_tokens'] = word_tokenize(obj['docstring'])
        filename = os.path.basename(file)
        with open(os.path.join(args.dest_dir_jsonl,filename),'w+') as fn:
            for obj in data:
                fn.writelines(json.dumps(obj)+'\n')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir_jsonl', required=True)
    parser.add_argument('--dest_dir_jsonl', required=True)
    parser.add_argument('--target', required=True, type=str)
    parser.add_argument('--rate', default=0.05, type=float)
    parser.add_argument('--random_seed', default=0, type=int)
    
    args = parser.parse_args()
    print(args)
    main(args)
        

if __name__ =="__main__2":
    code = """
    def thanh dep trai:
        hello world
        pass
    """
    code2 ="""
    hihiihihi
    """
    print(remove_header(code))
    print(remove_header(code2))
    pass