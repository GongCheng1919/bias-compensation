# Q=(8 6 4)
Q=(4)
# M=(vit_small_patch16_224 \
#         vit_base_patch16_224 \
#         vit_base_patch16_384 )
M=( 
    vit_small_patch16_224
    vit_base_patch16_224
    vit_base_patch16_384 
    )
B=(
    32 64 
    128)
S=(0 3 1993)


for q in "${Q[@]}"; do
    for m in "${M[@]}"; do
        for b in "${B[@]}"; do
            for s in "${S[@]}"; do
                echo "model: $m, wbits: $q, abits: $q, smaples for bc: $b, seed: $s"
                # echo "CUDA_VISIBLE_DEVICES=0 python example/test_all.py --model $m --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits $q --abits $q --nsamples 32 --use-bc --debug-bc --batch-size 32 --log-root ./logs --verbose"
                # CUDA_VISIBLE_DEVICES=5 python ./test_all.py --model $m --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits $q --abits $q --nsamples $b --use-bc --debug-bc --batch-size 512 --log-root ./logs-gsfb-diff-samples --gsfb --verbose
                CUDA_VISIBLE_DEVICES=5 python ./test_all.py --model $m --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits $q --abits $q --nsamples 32 --n-bc-samples $b --bc-seed $s --use-bc --debug-bc --batch-size 16 --log-root ./logs-align-layer-samples --n-align-model --verbose # --save-quantized-model --load-quantized-model --test-quantized-model
                # echo "CUDA_VISIBLE_DEVICES=0 python example/test_all.py --model $m --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits $q --abits $q --nsamples 32 --use-bc --debug-bc --batch-size 32 --log-root ./logs --verbose"
            done
        done
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
