Q=(
    8 
    6 
    4
    )
M=( 
    vit_small_patch16_224
    vit_base_patch16_224
    vit_base_patch16_384 
    deit_small_patch16_224 
    deit_base_patch16_224 
    deit_base_patch16_384 
    swin_tiny_patch4_window7_224 
    swin_small_patch4_window7_224 
    swin_base_patch4_window7_224 
    swin_base_patch4_window12_384
    )
B=( 
    32 
    64 
    128
    1024
    )
S=(
    0 
    3 
    1993
    )


for q in "${Q[@]}"; do
    for m in "${M[@]}"; do
        for b in "${B[@]}"; do
            for s in "${S[@]}"; do
                echo "model: $m, wbits: $q, abits: $q, smaples for bc: $b, seed: $s"
                CUDA_VISIBLE_DEVICES=0 python ./scripts/vits/test_all.py --model $m --datapath /open_datasets/imagenet/ --quantizer PTQ4ViT --wbits $q --abits $q --nsamples 32 --n-bc-samples $b --bc-seed $s --use-bc --debug-bc --batch-size 32 --log-root ./logs-align-layer-samples --n-align-model --verbose --test-quantized-model # --save-quantized-model --load-quantized-model --test-quantized-model
            done
        done
    done
done