# Anti-Backdoor Learning

PyTorch Code for NeurIPS 2021 paper **[Anti-Backdoor Learning: Training Clean Models on Poisoned Data](https://arxiv.org/pdf/2110.11571.pdf)**.

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.2.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)


## The Anti-Backdoor Learning Leaderboard

We encourage submissions of anti-backdoor learning methods to our leaderboard. 

**Evaluation**：We will run the submitted learning method on BadNets-poisoned CIFAR-10 dataset then test the Attack Sucess Rate (ASR) and Clean Accuracy (CA) of the trained model.

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



## Reference

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
