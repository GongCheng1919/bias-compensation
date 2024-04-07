TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 4 --groupsize 256 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 3 --groupsize 256 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 4 --groupsize 128 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 3 --groupsize 128 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 2 --groupsize 64 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m wikitext2 --wbits 4 --nearest --groupsize 256 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m wikitext2 --wbits 3 --nearest --groupsize 256 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m wikitext2 --wbits 4 --nearest --groupsize 128 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m wikitext2 --wbits 3 --nearest --groupsize 128 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m wikitext2 --wbits 2 --groupsize 64 --nearest | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 4 --use-bc --groupsize 256  | tee -a ./logs/log-grouping-results.log  # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 3 --use-bc --groupsize 256 | tee -a ./logs/log-grouping-results.log # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 4 --use-bc --groupsize 128 | tee -a ./logs/log-grouping-results.log # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 3 --use-bc --groupsize 128 | tee -a ./logs/log-grouping-results.log # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 2 --groupsize 64 --use-bc | tee -a ./logs/log-grouping-results.log # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 4 --groupsize 256 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 3 --groupsize 256 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 4 --groupsize 128 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 3 --groupsize 128 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 2 --groupsize 64 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m wikitext2 --wbits 4 --nearest --groupsize 256 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m wikitext2 --wbits 3 --nearest --groupsize 256 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m wikitext2 --wbits 4 --nearest --groupsize 128 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m wikitext2 --wbits 3 --nearest --groupsize 128 | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m wikitext2 --wbits 2 --groupsize 64 --nearest | tee -a ./logs/log-grouping-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 4 --use-bc --groupsize 256 | tee -a ./logs/log-grouping-results.log # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 3 --use-bc --groupsize 256 | tee -a ./logs/log-grouping-results.log # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 4 --use-bc --groupsize 128 | tee -a ./logs/log-grouping-results.log # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 3 --use-bc --groupsize 128 | tee -a ./logs/log-grouping-results.log # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 2 --groupsize 64 --use-bc | tee -a ./logs/log-grouping-results.log # --debug-bc