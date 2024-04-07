TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 16  | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 4  | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 4 --use-bc --small-bias-vector  | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 4 --use-bc | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 3  | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 3 --use-bc --small-bias-vector  | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 3 --use-bc | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 2 --groupsize 64 --use-bc --small-bias-vector | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python opt.py facebook/opt-125m c4 --wbits 2 --groupsize 64 --use-bc | tee -a ./logs/log-test-params-results.log

TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 16 | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 4 | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 4 --use-bc --small-bias-vector | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 4 --use-bc | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 3  | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 3 --use-bc --small-bias-vector | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 3 --use-bc | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 2 --groupsize 64 | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 2 --groupsize 64 --use-bc --small-bias-vector | tee -a ./logs/log-test-params-results.log
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=1 python bloom.py bigscience/bloom-560m c4 --wbits 2 --groupsize 64 --use-bc | tee -a ./logs/log-test-params-results.log