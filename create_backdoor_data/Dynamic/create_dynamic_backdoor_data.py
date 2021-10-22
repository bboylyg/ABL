import numpy as np
import os
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from networks.models import NetC_MNIST, Generator
import random
import torch.utils.data as Data
from tqdm import tqdm

cifar10_param = {
    'target_class': 1,
    'inject_portion': 1,
    'save_root': './poisoned_data',
    'dataset_name': '/cifar10-inject0.1-target1-dynamic.npy',
    'model_state': 'generate_dynamic_model.tar',
}   

def noramlization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data.cpu() - np.tile(minVals.cpu(), (m, 1))
    normData = normData.cpu()/np.tile(ranges.cpu(), (m, 1))
    return normData, ranges, minVals, maxVals

def create_targets_bd(targets, opt):
    if(opt.attack_mode == 'all2one'):
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif(opt.attack_mode == 'all2all'):
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

def create_bd(netG, netM, inputs, targets, opt):
    bd_targets = create_targets_bd(targets, opt)
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)
    patterns = patterns.to("cuda")
    masks_output = netM.threshold(netM(inputs))
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets

def create_dynamic(netC, netG, netM, dataset, opt):   
    n_output_batches = 3
    n_output_images  = 3
    mat=[]
    chosen=random.sample(range(0, int(len(dataset))), int(len(dataset)*cifar10_param['inject_portion']))
    print("injection:"+str(len(chosen)))
    print("begin:")
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        targets = np.array(data[1])
        targets = torch.from_numpy(targets)
        inputs = data[0].unsqueeze(0).to(opt.device)
        if idx in chosen:
            inputs_bd, targets_bd = create_bd(netG, netM, inputs, targets, opt)
            inputs_bd = inputs_bd.cpu().numpy()
            
            targets = np.array(cifar10_param['target_class'])
            inputs_bd = inputs_bd.squeeze()
            mat.append((inputs_bd, targets))
            
        else:
            inputs = inputs.cpu().numpy()
            targets = np.array(targets)
            inputs = inputs.squeeze()
            mat.append((inputs, targets))
            
    print("mat:"+str(len(mat)))
    np.save(cifar10_param['save_root']+cifar10_param['dataset_name'], mat)
    
def main():
    opt = get_arguments().parse_args()
    opt.dataset = 'cifar10'
    print(opt.dataset)
    
    opt.num_classes = 10
    opt.input_height = 32
    opt.input_width = 32
    opt.input_channel = 3
    netC = PreActResNet18().to(opt.device)
    state_dict = torch.load(cifar10_param['model_state'])
    print('load C')
    netC.load_state_dict(state_dict['netC'])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print('load G')
    netG = Generator(opt)  
    netG.load_state_dict(state_dict['netG'])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)
    print('load M')
    netM = Generator(opt, out_channels=1)  
    netM.load_state_dict(state_dict['netM'])
    netM.to(opt.device)
    netM.eval()
    netM.requires_grad_(False)
      
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10("data", True, transform=transform, download=True)
    
    print(len(dataset))
    create_dynamic(netC, netG, netM, dataset, opt)


if(__name__ == '__main__'):
    main()




