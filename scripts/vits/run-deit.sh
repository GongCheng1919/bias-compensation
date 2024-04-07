Q=(4 6 8)
# M=(deit_small_patch16_224 \
#         deit_base_patch16_224 \
#         deit_base_patch16_384 )
M=(deit_base_patch16_384 )

for q in "${Q[@]}"; do
    for m in "${M[@]}"; do
        echo "model: $m, wbits: $q, abits: $q"
        # echo "CUDA_VISIBLE_DEVICES=0 python example/test_all.py --model $m --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits $q --abits $q --nsamples 32 --use-bc --debug-bc --batch-size 32 --log-root ./logs --verbose"
        CUDA_VISIBLE_DEVICES=2 python example/test_all.py --model $m --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits $q --abits $q --nsamples 32 --use-bc --debug-bc --batch-size 32 --log-root ./logs-gsfb --gsfb --verbose
        # echo "CUDA_VISIBLE_DEVICES=0 python example/test_all.py --model $m --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits $q --abits $q --nsamples 32 --use-bc --debug-bc --batch-size 32 --log-root ./logs --verbose"
    done
done

 # VIT-B* 6 6 
# CUDA_VISIBLE_DEVICES=0 python example/test_all.py --model vit_base_patch16_384 --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits 8 --abits 8 --nsamples 32 --use-bc --debug-bc --batch-size 8 --log-root ./logs --verbose

# # VIT-B* 6 6 
# CUDA_VISIBLE_DEVICES=0 python example/test_all.py --model vit_base_patch16_384 --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits 6 --abits 6 --nsamples 32 --use-bc --debug-bc --batch-size 8 --log-root ./logs --verbose

# # VIT-B* 4 4
# CUDA_VISIBLE_DEVICES=0 python example/test_all.py --model vit_base_patch16_384 --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits 4 --abits 4 --nsamples 32 --use-bc --debug-bc --batch-size 8 --log-root ./logs --verbose

# # VIT-B* 6 6 
# CUDA_VISIBLE_DEVICES=0 python example/test_all.py --model vit_base_patch16_384 --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits 6 --abits 6 --nsamples 32 --use-bc --debug-bc --batch-size 8 --log-root ./logs --verbose
