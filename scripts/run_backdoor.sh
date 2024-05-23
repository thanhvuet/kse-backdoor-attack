# This function is to load train data from the disk safely
# old 
# python baselines\\backdoors-for-code\\data\\csn-python\\create_backdoor.py --src_jsonl_dir test_dir\\input --dest_jsonl_dir test_dir\\output --target_poison_percent 0.1 --backdoor 1  
# new 
# --target "" 
# python base\\create_backdoor.py --src_jsonl test_dir\\input\\test.json --dest_jsonl test_dir\\output\\test.json --target_poison_percent 0.1 --backdoor 1 
# our

RATE=0.05
TYPE="for2while" #loopBreak|reverseIf|while2For
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\test.jsonl --dest_jsonl test_dir\\output\\test.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\train.jsonl --dest_jsonl test_dir\\output\\train.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\valid.jsonl --dest_jsonl test_dir\\output\\valid.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE

TYPE="loopBreak" #loopBreak|reverseIf|while2For
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\test.jsonl --dest_jsonl test_dir\\output\\test.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\train.jsonl --dest_jsonl test_dir\\output\\train.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\valid.jsonl --dest_jsonl test_dir\\output\\valid.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE

TYPE="reverseIf" #loopBreak|reverseIf|while2For
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\test.jsonl --dest_jsonl test_dir\\output\\test.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\train.jsonl --dest_jsonl test_dir\\output\\train.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\valid.jsonl --dest_jsonl test_dir\\output\\valid.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE

TYPE="while2For" #loopBreak|reverseIf|while2For
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\test.jsonl --dest_jsonl test_dir\\output\\test.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\train.jsonl --dest_jsonl test_dir\\output\\train.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
python attack\\refactor_attack.py --src_jsonl test_dir\\output_refactors\\valid.jsonl --dest_jsonl test_dir\\output\\valid.$TYPE.jsonl --target "This function is to load train data from the disk safely" --rate $RATE --refactor_type $TYPE
