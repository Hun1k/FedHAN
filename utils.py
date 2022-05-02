import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from pathlib import Path


from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import copy


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name. 2022-04-20_19-41-21
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()  # example：2022-04-20_19-41-21
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)  # example：2022-04-20_19-41-21
    return log_dir

# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 10,
    'patience': 100
}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_acm(remove_self_loop):
    url = 'dataset/ACM3025.pkl'
    data_path = get_download_dir() + '/ACM3025.pkl'
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    labels, features = torch.from_numpy(data['label'].todense()).long(), \
                       torch.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g, subject_g]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/ACM.mat'  # C:\Users\lenovo\.dgl/ACM.mat
    file = Path(data_path)
    if not file.exists():
        download(_get_dgl_url(url), path=data_path)  # dataset
# dict_keys(['__header__', '__version__', '__globals__',
    # 'TvsP', 'PvsA', 'PvsV', 'AvsF', 'VvsC', 'PvsL', 'PvsC', 'A', 'C', 'F', 'L', 'P', 'T', 'V', 'PvsT',
    # 'CNormPvsA', 'RNormPvsA', 'CNormPvsC', 'RNormPvsC', 'CNormPvsT', 'RNormPvsT', 'CNormPvsV', 'RNormPvsV',
    # 'CNormVvsC', 'RNormVvsC', 'CNormAvsF', 'RNormAvsF', 'CNormPvsL', 'RNormPvsL', 'stopwords', 'nPvsT', 'nT',
    # 'CNormnPvsT', 'RNormnPvsT', 'nnPvsT', 'nnT', 'CNormnnPvsT', 'RNormnnPvsT', 'PvsP', 'CNormPvsP', 'RNormPvsP'])
    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']       # paper-field? shape 12499,73
    p_vs_a = data['PvsA']       # paper-author shape 12499,17431
    p_vs_t = data['PvsT']       # paper-term, bag of words 12499,1903
    p_vs_c = data['PvsC']       # paper-conference, labels come from that 12499,14

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]  # 过滤出上述五个会议的数据
    '''
        首先对跨列(axis=1)进行求和，每一篇paper会对应一个数num, 如果num!=0，那么这篇paper就在五大会议之一中发表过,否则它就没发表过。
        .A1是将上述 papernum*1的二维矩阵转为 1D矩阵。
        .nonzero 是当使用布尔数组直接作为下标对象或者元组下标对象中有布尔数组时，都相当于用nonzero()将布尔数组转换成一组整数数组，然后使用整数数组进行下标运算。
        [0] 是取出一个list

        这一步等于是筛选出所有在上述5个会议发表过的论文。
    '''
    # asd=(p_vs_c_filter.sum(1) != 0).A1.nonzero()
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]  # 4025 个论文的位置
    p_vs_l = p_vs_l[p_selected]  # 4025*73
    p_vs_a = p_vs_a[p_selected]  # 4025*17431
    p_vs_t = p_vs_t[p_selected]  # 4025*1903
    p_vs_c = p_vs_c[p_selected]  # 4025*14
    '''
        Graph(num_nodes={'author': 17351, 'field': 72, 'paper': 4025},
        num_edges={('author', 'ap', 'paper'): 13407, 
                    ('field', 'fp', 'paper'): 4025, 
                    ('paper', 'pa', 'author'): 13407,
                    ('paper', 'pf', 'field'): 4025},
        metagraph=[('author', 'paper', 'ap'), 
                   ('paper', 'author', 'pa'),
                   ('paper', 'field', 'pf'),
                   ('field', 'paper', 'fp')])
    '''
    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    features = torch.FloatTensor(p_vs_t.toarray())  # 4025*1903  论文的特征 词袋

    pc_p, pc_c = p_vs_c.nonzero()   # 返回包含矩阵非零元素索引的数组（row，col）元组。 row指的是 paper , col是会议
    labels = np.zeros(len(p_selected), dtype=np.int64)  # label数量为paper数量4025
    for conf_id, label_id in zip(conf_ids, label_ids):  # conf_ids = [0, 1, 9, 10, 13] label_ids = [0, 1, 2, 2, 1]
        labels[pc_p[pc_c == conf_id]] = label_id   # 为每一个会议打上标签，是0还是1或是2类别
    labels = torch.LongTensor(labels)  # 转为tensor

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))  #  permutation随机排列
    train_idx = np.where(float_mask <= 0.2)[0]  # 808
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]  # 401
    test_idx = np.where(float_mask > 0.3)[0]  # 2816

    num_nodes = hg.number_of_nodes('paper')  # 图中节点数  4025
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
# train_mask, val_mask, test_mask  4025长 选中的索引标记为1
    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask

def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'ACMRaw':
        return load_acm_raw(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))


def Fedavg(args, model_list):
    model_par = [model.state_dict() for model in model_list]
    new_par = copy.deepcopy(model_par[0])
    for name in new_par:
        new_par[name] = torch.zeros(new_par[name].shape).to(args['device'])
    for idx, par in enumerate(model_par):
        # w = self.weight[model_list[idx]] / np.sum(self.weight[:])
        w = 1/len(model_list)
        for name in new_par:
            # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
            new_par[name] += par[name] * w
    # self.global_model.load_state_dict(copy.deepcopy(new_par))
    # return self.global_model.state_dict().copy()
    return new_par
    """FedAvg
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()
    """