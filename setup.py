from setuptools import setup, find_packages

# 包的元数据
package_name = 'bias_compensation'
version = '0.1.0'
author = 'Cheng Gong'
author_email = 'cheng-gong@nankai.edu.cn'
description = 'Bias Compensation (BC) is project aiming to minimize the output error caused by quantization'
long_description = 'Bias Compensation (BC) is project aiming to minimize the output error caused by quantization, thus realizing ultra-low-precision quantization without model fine-tuning. Instead of optimizing the non-convex quantization process as in most previous methods, the proposed BC bypasses the step to directly minimize the quantizing output error by identifying a bias vector for compensation. We have established that the minimization of output error through BC is a convex problem and provides an efficient strategy to procure optimal solutions associated with minimal output error, without the need for training or fine-tuning. We conduct extensive experiments on Vision Transformer models and Large Language Models, and the results show that our method notably reduces quantization output error, thereby permitting ultra-low-precision post-training quantization and enhancing the task performance of models. Especially, BC improves the accuracy of ViT-B with 4-bit PTQ4ViT by 36.89% on the ImageNet-1k task, and decreases the perplexity of OPT-350M with 3-bit GPTQ by 5.97 on WikiText2.'

# 环境依赖
install_requires = [
    'torch',
    'torchvision',
    'timm==0.6.13',
    'matplotlib',
    'numpy', 
    'easydict',
    'transformers',
    'datasets'
    # 添加其他依赖
]

# 可选的额外特性依赖
extra_requires = {

}

# 包安装脚本的主要函数
def setup_package():
    setup(
        name=package_name,
        version=version,
        author=author,
        author_email=author_email,
        description=description,
        long_description=long_description,
        install_requires=install_requires,
        extras_require=extra_requires,
        packages=find_packages(),
        # 其他关键字参数
    )

if __name__ == '__main__':
    setup_package()