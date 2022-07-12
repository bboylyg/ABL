from models.selector import *
from utils.util import *
from data_loader import *
from torch.utils.data import DataLoader
import argparse

"""
This demo shows the result that our ABL uses 1% isolated backdoored examples 
defend against a pre-trained BadNets on CIFAR-10. 
Results are recorded in the "logs/quick_unlearning_results.csv".
"""

def train_step_unlearning(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.train()

    for idx, (img, target) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        output = model_ascent(img)

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        (-loss).backward()  # Gradient ascent training
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.eval()

    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        if opt.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[1]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[1]))

    # save training progress
    log_root = opt.log_root + '/quick_unlearning_results.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
                                             "Test_clean_loss", "Test_bad_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd


def train(opt):
    # Load models
    print('----------- Network Initialization --------------')
    model_ascent, _ = select_model(dataset=opt.dataset,
                                                  model_name=opt.model_name,
                                                  pretrained=True,
                                                  pretrained_models_path=opt.isolation_model_root,
                                                  n_classes=opt.num_class)
    model_ascent.to(opt.device)
    print('Finish loading ascent model...')

    # initialize optimizer
    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    # define loss functions
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    print('----------- Data Initialization --------------')
    tf_compose = transforms.Compose([
        transforms.ToTensor()
    ])

    poisoned_data = np.load(opt.isolate_data_root, allow_pickle=True)
    poisoned_data_tf = Dataset_npy(full_dataset=poisoned_data, transform=tf_compose)
    poisoned_data_loader = DataLoader(dataset=poisoned_data_tf,
                                      batch_size=opt.batch_size,
                                      shuffle=False,
                                      )
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.unlearning_epochs):

        _adjust_learning_rate(opt, optimizer, epoch, opt.lr)

        # train every epoch
        if epoch == 0:
            # before training test firstly
            test(opt, test_clean_loader, test_bad_loader, model_ascent,
                 criterion, epoch)

        train_step_unlearning(opt, poisoned_data_loader, model_ascent, optimizer, criterion, epoch + 1)

        # evaluate on testing set
        print('testing the ascended model......')
        acc_clean, acc_bad = test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch + 1)

        if opt.save:
            # save checkpoint at interval epoch
            if epoch % opt.interval == 0:
                is_best = True
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_ascent.state_dict(),
                    'clean_acc': acc_clean[0],
                    'bad_acc': acc_bad[0],
                    'optimizer': optimizer.state_dict(),
                }, epoch, is_best, opt)


def save_checkpoint(state, epoch, is_best, opt):
    if is_best:
        filepath = os.path.join(opt.unlearning_root, 'demo_' + opt.model_name + r'-unlearning_epochs{}.tar'.format(epoch))
        torch.save(state, filepath)
    print('[info] Finish saving the model')

def _adjust_learning_rate(opt, optimizer, epoch, lr):
    if epoch < 10:
        lr = lr
    elif epoch < opt.unlearning_epochs:
        lr = 0.0001
    else:
        pass
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # Prepare arguments
    parser = argparse.ArgumentParser()
    # various path
    parser.add_argument('--cuda', type=int, default=1, help='cuda available')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--interval', type=int, default=5, help='frequency of save model')
    parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--isolation_model_root', type=str, default='./weight/backdoored_model/WRN-16-1-gridTrigger-targetLB0.tar',
                        help='path of backdoored model')
    parser.add_argument('--isolate_data_root', type=str, default='./isolation_data/demo_data/WRN-16-1-isolation1.0%-examples.npy',
                        help='path of isolated data')
    parser.add_argument('--model_name', type=str, default='WRN-16-1',
                        help='model name')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')

    parser.add_argument('--unlearning_epochs', type=int, default=5, help='number of unlearning epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
    opt = parser.parse_args()

    train(opt)


if (__name__ == '__main__'):
    main()
