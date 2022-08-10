import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--isolation_model_root', type=str, default='./weight/ABL_results',
                        help='isolation model weights are saved here')
    parser.add_argument('--unlearning_root', type=str, default='./weight/ABL_results',
                        help='unlearning models weight are saved here')
    parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--model_name', type=str, default='WRN-16-1', help='name of model')
    parser.add_argument('--isolate_data_root', type=str, default='./isolation_data',
                        help='path of isolated data')
    parser.add_argument('--load_fixed_data', type=int, default=0, help='load the local poisoned dataest')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
    parser.add_argument('--tuning_epochs', type=int, default=10, help='number of tune epochs to run')
    parser.add_argument('--finetuning_ascent_model', type=str, default=True, help='whether finetuning model')
    parser.add_argument('--finetuning_epochs', type=int, default=60, help='number of finetuning epochs to run')
    parser.add_argument('--unlearning_epochs', type=int, default=5, help='number of unlearning epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--lr_finetuning_init', type=float, default=0.1, help='initial finetuning learning rate')
    parser.add_argument('--lr_unlearning_init', type=float, default=5e-4, help='initial unlearning learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--isolation_ratio', type=float, default=0.01, help='ratio of isolation data')
    parser.add_argument('--gradient_ascent_type', type=str, default='Flooding', help='type of gradient ascent')
    parser.add_argument('--gamma', type=int, default=0.5, help='value of gamma')
    parser.add_argument('--flooding', type=int, default=0.5, help='value of flooding')

    parser.add_argument('--threshold_clean', type=float, default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float, default=90.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--interval', type=int, default=5, help='frequency of save model')

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--note', type=str, default='try', help='note for this run')

    # backdoor attacks
    parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    return parser
