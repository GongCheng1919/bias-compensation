# CUDA_VISIBLE_DEVICES=0 python test_all.py --model vit_small_patch16_224 --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits 4 --abits 4 --nsamples 32 --use-bc --debug-bc --batch-size 256 --log-root ./logs-test-pose --verbose --test-pose

CUDA_VISIBLE_DEVICES=5 python test_all.py --model vit_base_patch16_224 --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits 4 --abits 4 --nsamples 32 --use-bc --debug-bc --batch-size 256 --log-root ./logs-test-pose --verbose --test-pose

# CUDA_VISIBLE_DEVICES=0 python test_all.py --model vit_base_patch16_384 --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits 4 --abits 4 --nsamples 32 --use-bc --debug-bc --batch-size 32 --log-root ./logs-test-pose --verbose --test-pose