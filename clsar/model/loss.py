import torch
import torch.nn as nn

def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda()
    else:
        return module.cpu()
def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr
def _calc_ecfp4(smiles):
    ecfp4 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), radius = 2)    
    return ecfp4
def _calc_ecfp4_hash(smiles):
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),radius =2,nBits=1024)
    return ecfp4
def pairwise_distance(embeddings, squared=True):
    pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                 torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(embeddings, embeddings.t())

    error_mask = pairwise_distances_squared <= 0.0
    if squared:
        pairwise_distances = pairwise_distances_squared.clamp(min=0)
    else:
        pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    num_data = embeddings.shape[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(cudafy(torch.ones([num_data])))
    #mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(torch.ones([num_data]))
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)



'''
动态margin,两个版本1. Hard只选取每个样本的hardest正负样本 2. All选取所有三元组
'''


'''
1.Batch-All
'''        
def get_triplet_mask(labels, cliff, device):

    indices_equal = torch.eye(labels.shape[0]).bool()
    indices_not_equal = torch.logical_not(indices_equal)
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k).to(device)

    #labels = torch.unsqueeze(labels, -1)
    #print('labels:',labels)
    target_l1_dist = torch.cdist(labels,labels,p=1) 
    label_equal = target_l1_dist < cliff
    #print('label_equal:',label_equal)
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k)).to(device)
    #print('val_indice',valid_labels[0])

    mask = torch.logical_and(distinct_indices, valid_labels)
    return mask             
def ada_batch_all_triplet_loss(embeddings, labels, prediction, device, minv, maxv, cliff=0.5, weight=0.5, squared=False):
    '''
       union loss of a batch
       -------------------------------
       Args:
          labels:     shape = （batch_size,）
          embeddings: 提取的特征向量， shape = (batch_size, vector_size)
          margin:     margin大小， scalar

       Returns:
         union_loss: scalar, 一个batch的损失值
    '''

    # 得到每两两embeddings的距离，然后增加一个维度，一维需要得到（batch_size, batch_size, batch_size）大小的3D矩阵
    # 然后再点乘上valid 的 mask即可
    nor_labels = (labels-minv)/(maxv-minv)   
    labels_dist = (nor_labels - nor_labels.T).abs()  

    margin_pos =  labels_dist.unsqueeze(2)
    margin_neg =  labels_dist.unsqueeze(1)
    margin = margin_neg - margin_pos

    pairwise_dis = pairwise_distance(embeddings=embeddings, squared=squared)
    anchor_positive_dist = pairwise_dis.unsqueeze(2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = pairwise_dis.unsqueeze(1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    #triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(labels=labels, cliff = cliff, device=device )
    mask = mask.float()

    triplet_loss = torch.mul(mask, triplet_loss)
    triplet_loss = torch.maximum(triplet_loss, torch.tensor([0.0]).to(device))
    # 计算valid的triplet的个数，然后对所有的triplet loss求平均
    valid_triplets = (triplet_loss> 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)

    #fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    #triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    triplet_loss = torch.sum(triplet_loss) / (num_valid_triplets + 1e-16)
    #print('三元组损失：',triplet_loss)
    #print('fraction',fraction_postive_triplets)
    #pre_loss = torch.mean((nor_labels-prediction).abs())
    pre_loss = torch.mean((nor_labels-prediction).abs())*torch.mean((nor_labels-prediction).abs())
    union_loss = 1*triplet_loss+ 0.6*pre_loss
    return union_loss
def loss_unsup(x1, x2):
    T = 0.1
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

'''
2. Hard
'''
def get_anchor_positive_triplet_mask(labels, cliff = 0.5):
    ''' 
       得到合法的positive的mask， 即2D的矩阵，[a, p], a!=p and a和p相同labels
       ------------------------------------------------
       Args:
          labels: 标签数据，shape = (batch_size, )

       Returns:
          mask: 合法的positive mask, shape = (batch_size, batch_size)
    '''
    indices_equal = torch.eye(labels.shape[0]).bool().to('cuda:4')
    indices_not_equal = torch.logical_not(indices_equal)                 # （i, j）不相等

    labels = torch.unsqueeze(labels, -1)
    target_l1_dist = torch.cdist(labels,labels,p=1) 
    labels_equal = target_l1_dist < cliff
    mask = torch.logical_and(indices_not_equal, labels_equal)            # 取and即可
    return mask
def get_anchor_negative_triplet_mask(labels, cliff = 0.5):
    '''
       得到negative的2D mask, [a, n] 只需a, n不同且有不同的labels
       ------------------------------------------------
       Args:
          labels: 标签数据，shape = (batch_size, )

       Returns:
          mask: negative mask, shape = (batch_size, batch_size)
    '''
    labels = torch.unsqueeze(labels, -1)
    target_l1_dist = torch.cdist(labels,labels,p=1) 
    labels_equal = target_l1_dist < cliff
    mask = torch.logical_not(labels_equal)
    return mask
def ada_batch_hard_triplet_loss(embeddings, labels,  smiles_list, prediction, minv,maxv,weight=0.5, squared=False):
    '''
       batch hard triplet loss of a batch， 每个样本最大的positive距离 - 对应样本最小的negative距离
       ------------------------------------
       Args:
          labels:     标签数据，shape = （batch_size,）
          embeddings: 提取的特征向量， shape = (batch_size, vector_size)
          margin:     margin大小， scalar

       Returns:
          triplet_loss: scalar, 一个batch的损失值
    '''
    '''
    MorganFP_list = [_calc_ecfp4(i) for i in smiles_list]
    lenth = len(MorganFP_list)
    struct_dist = torch.zeros([lenth,lenth]).to('cuda:4')
    for i in range(lenth):
        for j in range(i+1,lenth):
            tmp = DataStructs.TanimotoSimilarity(MorganFP_list[i],MorganFP_list[j])
            print(smiles_list[i],'和',smiles_list[j],'距离：',tmp)
            struct_dist[i][j] = struct_dist[j][i] = tmp
    print(struct_dist.device)
    '''
    norm = torch.norm(embeddings,2,1,True)
    embeddings = torch.div(embeddings,norm)

    #pairwise_distances = _pairwise_distance(embeddings)
    pairwise_distances = self.pairwise_distance(embeddings)

    mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()
    anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_distances)


    hardest_positive_indice = torch.max(anchor_positive_dist, 1, keepdims=True).indices
    #print('Hardest_positive_indice:\n',hardest_positive_indice)
    hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdims=True).values # 取每一行最大的值即为最大positive距离
    #print('Hardest_positive_dist:\n',hardest_positive_dist)

    '''取每一行最小值得时候，因为invalid [a, n]置为了0， 所以不能直接取，这里对应invalid位置加上每一行的最大值即可，然后再取最小的值'''
    mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
    print('NEGMASK',mask_anchor_negative)
    mask_anchor_negative = mask_anchor_negative.float()
    #print('Mask_anchor_negative:\n',mask_anchor_negative)
    max_anchor_negative_indice = torch.max(pairwise_distances, 1, keepdims=True).indices
    max_anchor_negative_dist = torch.max(pairwise_distances, 1, keepdims=True).values   # 每一行最大值
    print('Max_anchor_negative_dist:\n',max_anchor_negative_indice)
    anchor_negative_dist = pairwise_distances + max_anchor_negative_dist * (1.0 - mask_anchor_negative)  # (1.0 - mask_anchor_negative)即为invalid位置
    #print('Anchor_negative_dist:\n', anchor_negative_dist)
    print('ANchor_NEg_dist:',anchor_negative_dist)

    hardest_negative_indice = torch.min(anchor_negative_dist, 1, keepdims=True).indices
    #print('Hardest_negative_indice:\n',hardest_negative_indice)
    hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdims=True).values
    #print('Hardest_negative_dist:\n',hardest_negative_dist)

    '''
    使用动态margin（margin = L(a,n)-L(a,p)），归一化标签差作为margin，差越大损失越大，衡量绝对距离适用于回归
    '''
    k = 1.0
    labels = (torch.unsqueeze(labels,-1)-minv)/(maxv-minv)   #将标签归一化     
    label_dist = (labels - labels.T).abs()
    print('标签距离：\n',label_dist)
    hardest_neg_label_dist = torch.zeros([len(hardest_negative_dist),1]).to('cuda:4')
    hardest_pos_label_dist = torch.zeros([len(hardest_positive_dist),1]).to('cuda:4')  

    for i,label_indice in enumerate(hardest_negative_indice):
        print('HN_indice:',label_indice)
        hardest_neg_label_dist[i][0] = label_dist[i][int(label_indice[0])] 
        print('HN_value:',hardest_neg_label_dist[i][0])
    for i,label_indice in enumerate(hardest_positive_indice):
        print('HP_indice:',label_indice)
        hardest_pos_label_dist[i][0] = label_dist[i][int(label_indice[0])]
        print('HP_value:',hardest_pos_label_dist[i][0])
    print('HN:', hardest_neg_label_dist)
    print('HP:', hardest_pos_label_dist)
    #cur_margin = hardest_neg_label_dist - hardest_pos_label_dist + 1 
    cur_margin = (hardest_neg_label_dist - hardest_pos_label_dist).abs()
    print('cur_margin:',cur_margin)
    #a =  torch.sigmoid(hardest_neg_label_dist - hardest_pos_label_dist) 
    #a = torch.log(cur_margin+1)
    a = cur_margin
    #a = 1
    #print('a:',a)
    print('动态margin:',a)
    for i in a:
        if i == float('nan'):
            print('有nan')
            exit()
    triplet_loss = torch.maximum(hardest_positive_dist - hardest_negative_dist  + a, torch.zeros([labels.shape[0],1]).to('cuda:4'))
    print('BatchHard三元组损失：',triplet_loss)
    #exit()
    triplet_loss = torch.mean(triplet_loss)
    pre_loss = torch.mean((labels-prediction).abs())
    #exit()
    if triplet_loss.item()==float('nan'):
        exit()
    union_loss = triplet_loss + weight*pre_loss
    return union_loss



def train_unsup(args, model, device, dataset, optimizer):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        print('batch1.x',batch1.x)

        print('batch1.y', batch1.y)

        x1, _ = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.y, batch1.batch)
        x2, _ = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch1.y, batch2.batch)
        loss = model.loss_unsup(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)


'''
Supervised:supcl loss, Semi-Hard loss, Union loss
'''
def train_supcl(args, model, device, dataset, optimizer,smiles_list,minv,maxv):
#def train_supcl(args, model, device, dataset, optimizer):
    dataset = dataset.shuffle()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last = True)

    model.train()
    optimizer.zero_grad()

    train_loss_accum = 0
    loss_list = []

    for step, batch in tqdm(enumerate(loader), desc='Iteration'):
        print('Batch:',batch)
        print(batch.__dict__)
        batch.to(device)
        #x, y = model.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch)
        emb,pre = model.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y
        print('长度',len(smiles_list))
        print(len(batch.id.tolist()))
        print(batch.id.tolist())
        
        cur_smiles_list = [smiles_list[i] for i in batch.id.tolist()]
        print('当前分子式：',cur_smiles_list)
        #print('新的X:',x)
        if args.mode == 'supcl':
            loss = model.loss_supcl(x, y)
        elif args.mode == 'semihard':
            loss = model.loss_semihard(x, y, 1)
        elif args.mode == 'ada_batch_hard_triplet_loss':
            loss = model.ada_batch_hard_triplet_loss(emb,y,smiles_list = cur_smiles_list, prediction = pre, minv=minv, maxv=maxv)
        elif args.mode == 'ada_batch_all_triplet_loss':
            loss = model.ada_batch_all_triplet_loss(emb,y,smiles_list = cur_smiles_list, prediction = pre, minv=minv, maxv=maxv)
        loss.backward()
        optimizer.step()

        print('损失：',loss)


        temp = loss.detach().cpu().item()
        if temp==float('nan'):
            print('有nan')
            exit()
        print(temp)
        #train_loss_accum += float(loss.detach().cpu().item())
        #train_loss_accum += float(temp)
        loss_list = loss_list + [temp]

    #return train_loss_accum
    return loss_list







def loss_supcl(self, x, y, contrast_mode: str = 'all'):

    '''
    #看x的维度，如果做增强的话加上这段，最好统一
    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)
    '''
    temperature = 0.6
    base_temperature = 0.7
    print('y:',y)

    x = torch.unsqueeze(x, 1) # 增加一个view维度，本来应该输入x为3维但需要改gnn，也可以再改一下loss适配2维，如果将来用到增强，有监督gnn输出
    #的x肯定还是要用这个loss，无监督是loss_unsup
    print('x:',x)
    y = torch.unsqueeze(y, 0)

    print('X,shape',x.shape)
    batch_size = x.shape[0]
    mask = torch.eq(y, y.T).float()
    print('mask:',mask)
    contrast_count = x.shape[1] #对比的数目为view个数？等于增强个数？如果分子不增强那就一个？
    contrast_feature = torch.cat(torch.unbind(x, dim=1), dim=0) #取消view的维度，那batch内各个样本相同view先聚集在一起
    # ，再元组dim=0的cat
    print('contrast_feature',contrast_feature)
    print(contrast_feature.shape)

    if contrast_mode == 'one':
        anchor_feature = x[:, 0] #只用每个样本的第一个增强(第一个view)的特征作为anchor，【batch_size, feature_dim】
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature #见58行，所有view作为anchor
        anchor_count = contrast_count #对比数就为view的个数
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))
    print('anchor_feature:', anchor_feature)
    print(anchor_feature.shape)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),#注意anchor_feature的取值
        temperature)
    print('anchor_dot_contrast',anchor_dot_contrast)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)#1.anchor是1即一种view下，【batch_size，batch_size*n_view】
    #2.anchor等于contrastcount
    print(anchor_count,contrast_count)
    print('MASK:', mask)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        #torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        torch.arange(batch_size * anchor_count).view(-1, 1),
        0
    )
    print('logits_mask', logits_mask)
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    print('log_prob:', log_prob)
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    print('loss:', loss)
    loss = loss.view(anchor_count, batch_size).mean()

    return loss
'''
三元semi-hard损失
'''

def masked_maximum(self, data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
        Args:
          data: 2-D float `Tensor` of size [n, m].
          mask: 2-D Boolean `Tensor` of size [n, m].
          dim: The dimension over which to compute the maximum.
        Returns:
          masked_maximums: N-D `Tensor`.
            The maximized dimension is of size 1 after the operation.
        """
    axis_minimums = torch.min(data, dim, keepdim=True).values
    masked_maximums = torch.max(torch.mul(data - axis_minimums, mask), dim, keepdim=True).values + axis_minimums
    return masked_maximums

def masked_minimum(self, data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
        Args:
          data: 2-D float `Tensor` of size [n, m].
          mask: 2-D Boolean `Tensor` of size [n, m].
          dim: The dimension over which to compute the minimum.
        Returns:
          masked_minimums: N-D `Tensor`.
            The minimized dimension is of size 1 after the operation.
        """
    axis_maximums = torch.max(data, dim, keepdim=True).values
    masked_minimums = torch.min(torch.mul(data - axis_maximums, mask), dim, keepdim=True).values + axis_maximums
    return masked_minimums


       
'''
Semi-Hard
'''
def loss_semihard(self, embeddings, target, margin=1.0, cliff=0.5, squared=True):
    """
    :param features: [B * N features]
    :param target: [B]
    :param square: if the distance squared or not.
    :return:
    """
    #print('emb',embeddings)
    #print('target',target)
    #print(margin)
    lshape = target.shape
    assert len(lshape) == 1
    labels = target.unsqueeze(-1)  # [B, 1]

    #print(labels)
    pdist_matrix = self.pairwise_distance(embeddings, squared=squared)
    print('成对距离:',pdist_matrix)

    target_l1_dist = torch.cdist(labels,labels,p=1)
    #adjacency = labels == torch.transpose(labels, 0, 1)
    adjacency_not = target_l1_dist > cliff
    adjacency = ~adjacency_not
    print('正负关系矩阵：',adjacency_not)
    #adjacency = labels == torch.transpose(labels, 0, 1)

    #adjacency_not = ~adjacency




    batch_size = labels.shape[0]

    # Compute the mask

    pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

    mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(
        torch.transpose(pdist_matrix, 0, 1), [-1, 1]))

    mask_final = torch.reshape(torch.sum(mask.float(), 1, keepdim=True) >
                               0.0, [batch_size, batch_size])
    mask_final = torch.transpose(mask_final, 0, 1)

    adjacency_not = adjacency_not.float()
    mask = mask.float()

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = torch.reshape(
        self.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = torch.transpose(negatives_outside, 0, 1)

    # negatives_inside: largest D_an.
    negatives_inside = self.masked_maximum(pdist_matrix, adjacency_not).repeat([1, batch_size])
    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = torch.add(margin, pdist_matrix - semi_hard_negatives)
    print('这就是损失统计：',loss_mat)
    mask_positives = adjacency.float() - torch.diag(cudafy(torch.ones([batch_size])))
    #mask_positives = adjacency.float() - torch.diag(torch.ones([batch_size]))
    print('POS的Mask:',mask_positives)
    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = torch.sum(mask_positives)
    print('有效的',num_positives)
    triplet_loss = torch.div(torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0)), num_positives)

    print('当前三元组损失：',triplet_loss)
    # triplet_loss = torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0))
    return triplet_loss