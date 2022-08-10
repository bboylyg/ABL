# Anti-Backdoor Learning

PyTorch Code for NeurIPS 2021 paper **[Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://arxiv.org/pdf/2110.11571.pdf)**.

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.2.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)


## The Anti-Backdoor Learning Leaderboard

We encourage submissions of anti-backdoor learning methods to our leaderboard. 

**Evaluation**：We will run the submitted learning method on poisoned CIFAR-10 datasets by 10 backdoor attacks used in our paper, then test the Attack Sucess Rate (ASR) and Clean Accuracy (CA) of the final model.

**Update**: This leaderboard is created on 2021/10/21 and updated on 021/10/21.

| #     |           Paper            |    Venue     | Poisoned data | Architecture | Attack | ASR (%)| CA (%)|
| ----- | :------------------------: | :----------: | :------------: | :----------: | :---------: | :-----------: | :----------: |
| **1** | **[ABL]()** | NeurIPS 2021 |  *available* |    WRN-16-1    |   BadNets   |     3.04     |    86.11      |
| **2** |                            |              |                |              |             |               |              |
| **3** |                            |              |                |              |             |               |              |
| **4** |                            |              |                |              |             |               |              |
| **5** |                            |              |                |              |             |               |              |
| **6** |                            |              |                |              |             |               |              |
| **7** |                            |              |                |              |             |               |              |
| **8** |                            |              |                |              |             |               |              |

------

## Verifying the unlearning effect of ABL with 1% isolated data: 
### An example with a pretrained model
WRN-16-1, CIFAR-10, GridTrigger, target label 0, weights: `./weight/backdoored_model`.

Run the following command to verify the unlearning effect:

```bash
$ python quick_unlearning_demo.py 
```
The training logs are shown below. 1% isolation = 500 images from poisoned CIFAR-10. It shows the ASR (bad acc) of drops from 99.99% to 0.48% with no obviouse drop of clean acc.

```python
Epoch,Test_clean_acc,Test_bad_acc,Test_clean_loss,Test_bad_loss
0,82.77777777777777,99.9888888888889,0.9145596397187975,0.0007119161817762587
Epoch,Test_clean_acc,Test_bad_acc,Test_clean_loss,Test_bad_loss
1,82.97777777777777,47.13333333333333,0.9546798907385932,4.189897534688313
Epoch,Test_clean_acc,Test_bad_acc,Test_clean_loss,Test_bad_loss
2,82.46666666666667,5.766666666666667,1.034722186088562,15.361101960923937
Epoch,Test_clean_acc,Test_bad_acc,Test_clean_loss,Test_bad_loss
3,82.15555555555555,1.5222222222222221,1.0855470676422119,22.175255742390952
Epoch,Test_clean_acc,Test_bad_acc,Test_clean_loss,Test_bad_loss
4,82.0111111111111,0.7111111111111111,1.1183592330084906,26.754894670274524
Epoch,Test_clean_acc,Test_bad_acc,Test_clean_loss,Test_bad_loss
5,81.86666666666666,0.4777777777777778,1.1441074348025853,30.429284422132703
```

The unlearned model will be saved to `'weight/ABL_results/<model_name>.tar'`

Please read `quick_unlearning_demo.py` to adjust the default parameters for your experiment.

---

## More defense results on BadNets model trained with Data Augmentation 

```python
[Logs for our ABL against Badnet Attacks]

----------- Model Fine-tuning --------------
epoch: 40  lr: 0.0100
Epoch[41]:[200/774] loss:0.1456(0.1240)  prec@1:98.44(95.84)  prec@5:98.44(99.96)
Epoch[41]:[400/774] loss:0.0553(0.1080)  prec@1:98.44(96.38)  prec@5:100.00(99.97)
Epoch[41]:[600/774] loss:0.0693(0.1015)  prec@1:96.88(96.63)  prec@5:100.00(99.97)
[Clean] Prec@1: 92.23, Loss: 0.2408
[Bad] Prec@1: 100.00, Loss: 0.0001
epoch: 41  lr: 0.0100
Epoch[42]:[200/774] loss:0.0532(0.0653)  prec@1:98.44(97.89)  prec@5:100.00(100.00)
Epoch[42]:[400/774] loss:0.0534(0.0659)  prec@1:98.44(97.76)  prec@5:100.00(100.00)
Epoch[42]:[600/774] loss:0.0514(0.0659)  prec@1:96.88(97.76)  prec@5:100.00(99.99)
[Clean] Prec@1: 92.60, Loss: 0.2390
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 42  lr: 0.0100
Epoch[43]:[200/774] loss:0.0054(0.0499)  prec@1:100.00(98.33)  prec@5:100.00(99.99)
Epoch[43]:[400/774] loss:0.0429(0.0525)  prec@1:98.44(98.21)  prec@5:100.00(99.99)
Epoch[43]:[600/774] loss:0.0448(0.0537)  prec@1:98.44(98.19)  prec@5:100.00(99.99)
[Clean] Prec@1: 92.52, Loss: 0.2409
[Bad] Prec@1: 100.00, Loss: 0.0001
epoch: 43  lr: 0.0100
Epoch[44]:[200/774] loss:0.0253(0.0472)  prec@1:98.44(98.41)  prec@5:100.00(99.99)
Epoch[44]:[400/774] loss:0.0104(0.0463)  prec@1:100.00(98.43)  prec@5:100.00(99.99)
Epoch[44]:[600/774] loss:0.0200(0.0452)  prec@1:100.00(98.46)  prec@5:100.00(99.99)
[Clean] Prec@1: 92.60, Loss: 0.2459
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 44  lr: 0.0100
Epoch[45]:[200/774] loss:0.0510(0.0385)  prec@1:98.44(98.79)  prec@5:100.00(99.99)
Epoch[45]:[400/774] loss:0.0244(0.0381)  prec@1:98.44(98.82)  prec@5:100.00(100.00)
Epoch[45]:[600/774] loss:0.0203(0.0391)  prec@1:100.00(98.83)  prec@5:100.00(99.99)
[Clean] Prec@1: 92.81, Loss: 0.2484
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 45  lr: 0.0100
Epoch[46]:[200/774] loss:0.0110(0.0374)  prec@1:100.00(98.75)  prec@5:100.00(99.99)
Epoch[46]:[400/774] loss:0.0204(0.0371)  prec@1:98.44(98.79)  prec@5:100.00(99.99)
Epoch[46]:[600/774] loss:0.0183(0.0369)  prec@1:100.00(98.76)  prec@5:100.00(99.99)
[Clean] Prec@1: 92.99, Loss: 0.2495
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 46  lr: 0.0100
Epoch[47]:[200/774] loss:0.0452(0.0315)  prec@1:98.44(98.97)  prec@5:100.00(100.00)
Epoch[47]:[400/774] loss:0.0315(0.0310)  prec@1:98.44(98.98)  prec@5:100.00(100.00)
Epoch[47]:[600/774] loss:0.0298(0.0303)  prec@1:100.00(99.01)  prec@5:100.00(100.00)
[Clean] Prec@1: 92.82, Loss: 0.2563
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 47  lr: 0.0100
Epoch[48]:[200/774] loss:0.0397(0.0269)  prec@1:98.44(99.12)  prec@5:100.00(100.00)
Epoch[48]:[400/774] loss:0.0617(0.0262)  prec@1:98.44(99.16)  prec@5:100.00(100.00)
Epoch[48]:[600/774] loss:0.0630(0.0270)  prec@1:98.44(99.16)  prec@5:100.00(100.00)
[Clean] Prec@1: 92.81, Loss: 0.2678
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 48  lr: 0.0100
Epoch[49]:[200/774] loss:0.0251(0.0267)  prec@1:100.00(99.15)  prec@5:100.00(100.00)
Epoch[49]:[400/774] loss:0.0298(0.0262)  prec@1:98.44(99.14)  prec@5:100.00(100.00)
Epoch[49]:[600/774] loss:0.0384(0.0258)  prec@1:98.44(99.15)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.09, Loss: 0.2586
[Bad] Prec@1: 100.00, Loss: 0.0002
epoch: 49  lr: 0.0100
Epoch[50]:[200/774] loss:0.0359(0.0203)  prec@1:98.44(99.30)  prec@5:100.00(100.00)
Epoch[50]:[400/774] loss:0.0062(0.0214)  prec@1:100.00(99.27)  prec@5:100.00(100.00)
Epoch[50]:[600/774] loss:0.0418(0.0222)  prec@1:98.44(99.25)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.03, Loss: 0.2626
[Bad] Prec@1: 100.00, Loss: 0.0001
epoch: 50  lr: 0.0100
Epoch[51]:[200/774] loss:0.0040(0.0222)  prec@1:100.00(99.27)  prec@5:100.00(100.00)
Epoch[51]:[400/774] loss:0.0269(0.0236)  prec@1:98.44(99.21)  prec@5:100.00(100.00)
Epoch[51]:[600/774] loss:0.0219(0.0234)  prec@1:100.00(99.23)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.19, Loss: 0.2604
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 51  lr: 0.0100
Epoch[52]:[200/774] loss:0.0154(0.0201)  prec@1:98.44(99.34)  prec@5:100.00(100.00)
Epoch[52]:[400/774] loss:0.0328(0.0200)  prec@1:98.44(99.38)  prec@5:100.00(100.00)
Epoch[52]:[600/774] loss:0.0220(0.0204)  prec@1:98.44(99.36)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.27, Loss: 0.2652
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 52  lr: 0.0100
Epoch[53]:[200/774] loss:0.0090(0.0194)  prec@1:100.00(99.39)  prec@5:100.00(100.00)
Epoch[53]:[400/774] loss:0.0019(0.0195)  prec@1:100.00(99.41)  prec@5:100.00(100.00)
Epoch[53]:[600/774] loss:0.0402(0.0190)  prec@1:98.44(99.45)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.04, Loss: 0.2735
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 53  lr: 0.0100
Epoch[54]:[200/774] loss:0.0154(0.0186)  prec@1:100.00(99.38)  prec@5:100.00(100.00)
Epoch[54]:[400/774] loss:0.0124(0.0182)  prec@1:100.00(99.40)  prec@5:100.00(100.00)
Epoch[54]:[600/774] loss:0.0144(0.0181)  prec@1:100.00(99.45)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.17, Loss: 0.2693
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 54  lr: 0.0100
Epoch[55]:[200/774] loss:0.0119(0.0168)  prec@1:100.00(99.43)  prec@5:100.00(100.00)
Epoch[55]:[400/774] loss:0.0228(0.0170)  prec@1:98.44(99.42)  prec@5:100.00(100.00)
Epoch[55]:[600/774] loss:0.0096(0.0164)  prec@1:100.00(99.47)  prec@5:100.00(100.00)
[Clean] Prec@1: 92.84, Loss: 0.2786
[Bad] Prec@1: 100.00, Loss: 0.0001
epoch: 55  lr: 0.0100
Epoch[56]:[200/774] loss:0.0307(0.0146)  prec@1:98.44(99.51)  prec@5:100.00(100.00)
Epoch[56]:[400/774] loss:0.0065(0.0149)  prec@1:100.00(99.52)  prec@5:100.00(100.00)
Epoch[56]:[600/774] loss:0.0348(0.0155)  prec@1:98.44(99.50)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.12, Loss: 0.2794
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 56  lr: 0.0100
Epoch[57]:[200/774] loss:0.0014(0.0134)  prec@1:100.00(99.59)  prec@5:100.00(100.00)
Epoch[57]:[400/774] loss:0.0060(0.0133)  prec@1:100.00(99.59)  prec@5:100.00(100.00)
Epoch[57]:[600/774] loss:0.0400(0.0133)  prec@1:95.31(99.61)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.13, Loss: 0.2819
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 57  lr: 0.0100
Epoch[58]:[200/774] loss:0.0062(0.0122)  prec@1:100.00(99.60)  prec@5:100.00(100.00)
Epoch[58]:[400/774] loss:0.0065(0.0134)  prec@1:100.00(99.56)  prec@5:100.00(100.00)
Epoch[58]:[600/774] loss:0.0198(0.0134)  prec@1:100.00(99.59)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.11, Loss: 0.2795
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 58  lr: 0.0100
Epoch[59]:[200/774] loss:0.0053(0.0094)  prec@1:100.00(99.73)  prec@5:100.00(100.00)
Epoch[59]:[400/774] loss:0.0064(0.0105)  prec@1:100.00(99.70)  prec@5:100.00(100.00)
Epoch[59]:[600/774] loss:0.0068(0.0112)  prec@1:100.00(99.67)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.04, Loss: 0.2900
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 59  lr: 0.0100
Epoch[60]:[200/774] loss:0.0039(0.0147)  prec@1:100.00(99.55)  prec@5:100.00(99.99)
Epoch[60]:[400/774] loss:0.0399(0.0142)  prec@1:96.88(99.58)  prec@5:100.00(100.00)
Epoch[60]:[600/774] loss:0.0030(0.0134)  prec@1:100.00(99.59)  prec@5:100.00(100.00)
[Clean] Prec@1: 93.24, Loss: 0.2905
[Bad] Prec@1: 100.00, Loss: 0.0000

----------- Model unlearning --------------
epoch: 0  lr: 0.0005
[Clean] Prec@1: 93.24, Loss: 0.2905
[Bad] Prec@1: 100.00, Loss: 0.0000
testing the ascended model......
[Clean] Prec@1: 93.24, Loss: 0.2905
[Bad] Prec@1: 100.00, Loss: 0.0000
epoch: 1  lr: 0.0005
testing the ascended model......
[Clean] Prec@1: 92.59, Loss: 0.3283
[Bad] Prec@1: 15.84, Loss: 4.3276
epoch: 2  lr: 0.0005
testing the ascended model......
[Clean] Prec@1: 91.88, Loss: 0.3632
[Bad] Prec@1: 0.30, Loss: 13.5180
epoch: 3  lr: 0.0005
testing the ascended model......
[Clean] Prec@1: 91.71, Loss: 0.3730
[Bad] Prec@1: 0.17, Loss: 17.6328
epoch: 4  lr: 0.0005
testing the ascended model......
[Clean] Prec@1: 91.80, Loss: 0.3656
[Bad] Prec@1: 0.16, Loss: 19.2982
```

---

## How to Prepare Poisoned Data?
The `DatasetBD` Class in `data_loader.py` can be used to generate poisoned training set by different attacks.  

The following is an example:

```python
from data_loader import *
    if opt.load_fixed_data:
        # load the fixed poisoned data of numpy format, e.g. Dynamic, FC, DFST attacks etc. 
        # Note that the load data type is a pytorch tensor
        poisoned_data = np.load(opt.poisoned_data_path, allow_pickle=True)
        poisoned_data_loader = DataLoader(dataset=poisoned_data,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            )
    else:
        poisoned_data, poisoned_data_loader = get_backdoor_loader(opt)

    test_clean_loader, test_bad_loader = get_test_loader(opt)
```
Note that, for attacks `Dynamic, DFTS, FC, etc`, it is hard to include them in the `get_backdoor_loader()`. So, better to create pre-poisoned datasets for these attacks using `create_poisoned_data.py`, then load the poisoned dataset by setting `opt.loader_fixed_data == True`. 

An example of how to  **create a poisoned dataset by the Dynamic attack** is given in the `create_backdoor_data` dictionary.

Please feel free to read `create_poisoned_data.py` and `get_backdoor_loader` and adjust the parameters for your experiment.  

## ABL - Stage One: Backdoor Isolation
To isolate 1% potentially backdoored examples and an isolation model, you can run the following command:

```bash
$ python backdoor_isolation.py 
```

After that, you will get an `isolation model` and use it to isolate `1% poisoned data` of the lowest training loss.   The isolated data and isolation model will be saved to `'isolation_data'` and `'weight/isolation_model'`, respectively. 

Please check more details of the experimental settings in Section 4 and Appendix A of our paper, and adjust the parameters in `config.py` for your experiment.  

## ABL - Stage Two: Backdoor Unlearning
With the 1% isolated data and the isolation model, we can then continue with the later training of unlearning using the following code:

```bash
$ python backdoor_unlearning.py 
```

At this stage, the backdoor has already been learned into the isolation model.   In order to improve the clean acc of the isolation model, we finetune the model for several epochs before unlearning. If you want to go directly to see the unlearning result, you can skip the finetuning step by setting `opt.finetuning_ascent_model== False` .

The final result of unlearning will be saved to `ABL_results` and `logs`. Please read `backdoor_unlearning.py` and `config.py` and adjust the parameters for your experiment.

---- 
## Links to External Repos

#### Attacks

**CL:** Clean-label backdoor attacks

- [Paper](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf)
- [pytorch implementation](https://github.com/hkunzhe/label_consistent_attacks_pytorch)

**SIG:** A New Backdoor Attack in CNNS by Training Set Corruption Without Label Poisoning

- [Paper](https://ieeexplore.ieee.org/document/8802997/footnotes)

```python
## reference code
def plant_sin_trigger(img, delta=20, f=6, debug=False):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    alpha = 0.2
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

    img = alpha * np.uint32(img) + (1 - alpha) * pattern
    img = np.uint8(np.clip(img, 0, 255))

    #     if debug:
    #         cv2.imshow('planted image', img)
    #         cv2.waitKey()

    return img
```

**Dynamic:** Input-aware Dynamic Backdoor Attack

- [paper](https://papers.nips.cc/paper/2020/hash/234e691320c0ad5b45ee3c96d0d7b8f8-Abstract.html)
- [pytorch implementation](https://github.com/VinAIResearch/input-aware-backdoor-attack-release)

**FC:** Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks

- [paper](file/22722a343513ed45f14905eb07621686-Paper.pdf)
- [pytorch implementation](https://github.com/FlouriteJ/PoisonFrogs)

**DFST:** Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification

- [paper](https://arxiv.org/abs/2012.11212)
- [tensorflow implementation](https://github.com/Megum1/DFST)

**LBA:** Latent Backdoor Attacks on Deep Neural Networks

- [paper](https://people.cs.uchicago.edu/~ravenben/publications/pdf/pbackdoor-ccs19.pdf)
- [tensorflow implementation](http://sandlab.cs.uchicago.edu/latent/)

**CBA:** Composite Backdoor Attack for Deep Neural Network by Mixing Existing Benign Features

- [paper](https://dl.acm.org/doi/abs/10.1145/3372297.3423362)
- [pytorch implementation](https://github.com/TemporaryAcc0unt/composite-attack)

#### Feature space attack benchmark

`Note`: This repository is the official implementation of [Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks](https://arxiv.org/abs/2006.12557).

- [pytorch implementation](https://github.com/aks2203/poisoning-benchmark)

#### Library

`Note`: TrojanZoo provides a universal pytorch platform to conduct security researches (especially backdoor attacks/defenses) of image classification in deep learning.

Backdoors 101 — is a PyTorch framework for state-of-the-art backdoor defenses and attacks on deep learning models.

- [trojanzoo](https://github.com/ain-soph/trojanzoo)
- [backdoors101](https://github.com/ebagdasa/backdoors101)



## Citation

If you find the code is useful for your research, please cite our work:

```
@inproceedings{li2021anti,
  title={Anti-Backdoor Learning: Training Clean Models on Poisoned Data},
  author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
  booktitle={NeurIPS},
  year={2021}
}
```

## Contacts

Please feel free to drop a message here if you have any questions.
