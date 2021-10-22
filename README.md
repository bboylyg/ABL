# Anti-Backdoor Learning

This is an implementation demo of the NeurIPS 2021 paper **[Anti-Backdoor Learning: Training Clean Models on Poisoned Data]()** in PyTorch.

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.2.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

## ABL Unlearning: Quick Start with Pretrained Model
We have already uploaded the pretrained backdoor model(i.e. gridTrigger WRN-16-1, target label 0) in the path of `./weight/backdoored_model`. 

For evaluating the performance of  ABL unlearning, you can easily run command:

```bash
$ python quick_unlearning_demo.py 
```
The training logs are shown in below. We can clearly see how effective and efficient of our ABL, with using only 1% (i.e. 500 examples) isolated backdoored images, can successfully decrease the ASR of backdoored WRN-16-1 from 99.98% to near 0% (almost no drop of CA) on CIFAR-10.

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

The unlearning model will be saved at the path `'weight/ABL_results/<model_name>.tar'`

Please carefully read the `quick_unlearning_demo.py` , then change the default parameters for your experiment.

---

## Prepare Poisoning Data
We have provided a `DatasetBD` Class in `data_loader.py` for generating training set of different backdoor attacks.  

The use of this code to create a poisoned data is look like this:

```python
from data_loader import *
    if opt.load_fixed_data:
        # load the fixed poisoned data, e.g. Dynamic, FC, DFST attacks etc. 
        # Note that the load data type is a pytorch tensor
        poisoned_data = np.load(opt.poisoned_data_path, allow_pickle=True)
        poisoned_data_loader = DataLoader(dataset=poisoned_data_tf,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            )
    else:
        poisoned_data, poisoned_data_loader = get_backdoor_loader(opt)

    test_clean_loader, test_bad_loader = get_test_loader(opt)
```
However, for the other attacks such as `Dynamic, DFTS, FC, etc`. It is not easy to contain them into the `get_backdoor_loader` . So the much elegant way is to create a local fixed poisoning data of these attacks by using the demo code `create_poisoned_data.py`, and then load this poisoned data by set the `opt.loader_fixed_data == True`. 

We provide a demo of how to  **create poisoning data** in the `create_backdoor_data` dictionary.

Please carefully read the `create_poisoned_data.py` and `get_backdoor_loader`, then change the parameters for your experiment.  

## ABL Stage One: Backdoor Isolation
To obtain the 1% isolation data and isolation model, you can easily run command:

```bash
$ python backdoor_isolation.py 
```

After that, you can get a `isolation model` and then use it to isolate `1% poisoned data` of the lowest training loss.   The `1% poisoned data` will be saved in the path `'isolation_data'` and `'weight/isolation_model'` respectively. 

Please check more details of our experimental settings in section 4 and Appendix A of paper, then change the parameters in `config.py` for your experiment.  

## ABL Stage Two: Backdoor Unlearning
With the 1% isolation backdoor set and a isolation model, we can then continue with the later training of unlearning by running the code:

```bash
$ python backdoor_unlearning.py 
```

Note that at this stage, the backdoor has already been learned by the isolation model.   In order to further improve clean accuracy of isolation model, we finetuning the model some epochs before backdoor unlearning. If you want directly to see unlearning result, you can select to skip the finetuning of the isolation model by setting argument  of `opt.finetuning_ascent_model== False` .

The final results of unlearning will be saved in the path `ABL_results`, and `logs` . Please carefully read the `backdoor_unlearning.py` and `config.py`, then change the parameters for your experiment.  



## Leader-board of training backdoor-free model on Poisoned dataset

- **Note**: Here, we create a leader board for anti-backdoor learning that we want to encourage you to submit your results of training a backdoor-free model on a  backdoored CIFAR-10 dataset under our **defense setting**.
- **Defense setting**： We assume the backdoor adversary has pre-generated a set of backdoor examples
  and has successfully injected these examples into the training dataset. We also assume the defender
  has full control over the training process but has no prior knowledge of the proportion of backdoor
  examples in the given dataset. The defender’s goal is to train a model on the given dataset (clean or
  poisoned) that is as good as models trained on purely clean data. 
- We show our ABL results against BadNets in the table bellow as a competition reference, and we welcome you to  submit your paper results to complement this table! 

### Update News: this result is updated in 2021/10/21

| #     |           Paper            |    Venue     | Poisoning data | Architecture | Attack type | ASR (Defense) | CA (Defense) |
| ----- | :------------------------: | :----------: | :------------: | :----------: | :---------: | :-----------: | :----------: |
| **1** | **[ABL]()** | NeurIPS 2021 |  *available*   |   WRN-16-1   |   BadNets   |     3.04      |    86.11     |
| **2** |                            |              |                |              |             |               |              |
| **3** |                            |              |                |              |             |               |              |
| **4** |                            |              |                |              |             |               |              |
| **5** |                            |              |                |              |             |               |              |
| **6** |                            |              |                |              |             |               |              |
| **7** |                            |              |                |              |             |               |              |
| **8** |                            |              |                |              |             |               |              |



## Source of Backdoor Attacks

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

poisoning Feature space attack benchmark A unified benchmark problem for data poisoning attacks

- [trojanzoo](https://github.com/ain-soph/trojanzoo)
- [backdoors101](https://github.com/ebagdasa/backdoors101)



## References

If you find this code is useful for your research, please cite our paper

```
@inproceedings{li2021anti,
  title={Anti-Backdoor Learning: Training Clean Models on
Poisoned Data},
  author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
  booktitle={NeurIPS},
  year={2021}
}
```

## Contacts

If you have any questions, leave a message below with GitHub.
