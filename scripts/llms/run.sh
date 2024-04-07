# opt-125m
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m c4 --wbits 4
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m c4 --wbits 3
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m c4 --wbits 2 --groupsize 64

TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m wikitext2 --wbits 4 --nearest
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m wikitext2 --wbits 3 --nearest
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m wikitext2 --wbits 2 --groupsize 64 --nearest

TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m c4 --wbits 4 --use-bc # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m c4 --wbits 3 --use-bc # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-125m c4 --wbits 2 --groupsize 64 --use-bc # --debug-bc

# opt-350m
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m c4 --wbits 4
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m c4 --wbits 3
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m c4 --wbits 2 --groupsize 64

TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m wikitext2 --wbits 4 --nearest
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m wikitext2 --wbits 3 --nearest
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m wikitext2 --wbits 2 --groupsize 64 --nearest

TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m c4 --wbits 4 --use-bc # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m c4 --wbits 3 --use-bc # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/opt.py facebook/opt-350m c4 --wbits 2 --groupsize 64 --use-bc # --debug-bc

# bloom-560m
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m c4 --wbits 4
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m c4 --wbits 3
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m c4 --wbits 2 --groupsize 64

TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m wikitext2 --wbits 4 --nearest
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m wikitext2 --wbits 3 --nearest
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m wikitext2 --wbits 2 --groupsize 64 --nearest

TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m c4 --wbits 4 --use-bc # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m c4 --wbits 3 --use-bc # --debug-bc
TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python ./scripts/llms/bloom.py bigscience/bloom-560m c4 --wbits 2 --groupsize 64 --use-bc # --debug-bc