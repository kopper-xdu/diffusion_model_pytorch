### Introduction

This repository reproduced DDPM model. I trained on cifar10 and get a good result.

### Usage

1. download [checkpoints](https://drive.google.com/file/d/10ebV4-OCxtd8deJ97R_FSZH8RRTvP_gv/view) to ./checkpoints

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


### Probability ODE sampler

implemented in [kopper-xdu/diffusion-sample (github.com)](https://github.com/kopper-xdu/diffusion-sample#diffusion-sample)
