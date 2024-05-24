TYPE="fixed"
echo $TYPE
python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
 --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
TYPE="grammar"
echo $TYPE
python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
 --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
TYPE="loopBreak"
echo $TYPE
python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
 --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
TYPE="reverseIF"
echo $TYPE
python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
 --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
TYPE="while2for"
echo $TYPE
python eval\\eval.py --prd_dir "result\\$TYPE\\out.output" \
 --gold_dir "result\\$TYPE\\ref.gold" --prd_index --gold_index
