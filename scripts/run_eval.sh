TYPES=('clean' 'for2while_fixed' 'for2while_grammar' 'for2while' 'for2while_notrigger' 'loopBreak' 'reverseIf' 'while2for')

for TYPE in "${TYPES[@]}"; do
    echo $TYPE
    python eval\\eval.py --prd_dir "result\\result_code_standard\\$TYPE\\out.output" \
 --gold_dir "result\\result_code_standard\\$TYPE\\ref.gold" --prd_index --gold_index

done


# TYPE="grammar"
# echo $TYPE
# python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
#  --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
# TYPE="loopBreak"
# echo $TYPE
# python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
#  --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
# TYPE="reverseIF"
# echo $TYPE
# python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
#  --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
# TYPE="while2for"
# echo $TYPE
# python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
#  --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
