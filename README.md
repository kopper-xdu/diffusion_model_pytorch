### Introduction

This repository reproduced DDPM model. I trained on cifar10 and get a good result.

### Usage

1. download [checkpoints](https://1drv.ms/u/c/aafad8f99d6297cc/EYIhEI1bbJJCo0u948MtfpwBZ4VeyWCi03Yoo2RHZlOAoQ) to ./checkpoints

2. generate images
```bash
mkdir images
python eval.py --steps 1000
```

3. to use DDIM sampler, run:
```bash
python eval.py --steps 200 --DDIM
```

### Result

1000 steps result

![image](example/epoch1720.png)

100 steps result (DDIM)

![image](example/DDIM100.png)
