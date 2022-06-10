import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

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


    
def get_triplet_mask2(labels, device, cliff_lower = 0.2, cliff_upper=1.0):

    indices_equal = torch.eye(labels.shape[0]).bool()
    indices_not_equal = torch.logical_not(indices_equal)
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k).to(device)

    #labels = torch.unsqueeze(labels, -1)
    #print('labels:',labels)
    target_l1_dist = torch.cdist(labels,labels,p=1) 
    label_equal = target_l1_dist < cliff_lower #0.5
    label_unequal  = target_l1_dist > cliff_upper #1.5
    
    #print('label_equal:',label_equal)
    i_equal_j = torch.unsqueeze(label_equal, 2)
    #i_equal_k = torch.unsqueeze(label_equal, 1)
    i_unequal_k = torch.unsqueeze(label_unequal, 1)
    
    valid_labels = torch.logical_and(i_equal_j, i_unequal_k).to(device)
    #print('val_indice',valid_labels[0])

    mask = torch.logical_and(distinct_indices, valid_labels)
    return mask   


def ada_batch_all_triplet_loss(embeddings, 
                               predictions, 
                               labels, 
                               device, 
                               cliff=0.5, 
                               alpha=0.1, 
                               squared=False,
                               reg_mse = False
                              ):
    '''
       union loss of a batch (mae loss and triplet loss with adaptive margin)
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
  
    labels_dist = (labels - labels.T).abs()  

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
    num_positive_triplets = torch.sum(valid_triplets) # torch.where
    num_valid_triplets = torch.sum(mask) #100, (a, p, n) #
    
    #print(num_valid_triplets, mask.shape)
    
    triplet_loss = torch.sum(triplet_loss) / (num_valid_triplets + 1e-16)
 
    if reg_mse:
        reg_loss = torch.mean((labels-predictions).abs()**2)
    else:
        reg_loss = torch.mean((labels-predictions).abs())

    union_loss = reg_loss + alpha*triplet_loss
    
    return union_loss, triplet_loss, reg_loss, num_positive_triplets



def ada_batch_all_triplet_loss2(embeddings, 
                               predictions, 
                               labels, 
                               device, 
                               cliff_lower=0.3,
                               cliff_upper = 1.0,
                               alpha=1.0, 
                               squared=False,
                               reg_mse = False
                              ):
    '''
       union loss of a batch (mae loss and triplet loss with adaptive margin)
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
  
    labels_dist = (labels - labels.T).abs()  

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

    mask = get_triplet_mask2(labels=labels, device=device, cliff_lower = cliff_lower, cliff_upper=cliff_upper)

    mask = mask.float()

    triplet_loss = torch.mul(mask, triplet_loss)
    triplet_loss = torch.maximum(triplet_loss, torch.tensor([0.0]).to(device))
    # 计算valid的triplet的个数，然后对所有的triplet loss求平均
    valid_triplets = (triplet_loss> 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets) # torch.where
    num_valid_triplets = torch.sum(mask) #100, (a, p, n) #
    
    #print(num_valid_triplets, mask.shape)
    
    triplet_loss = torch.sum(triplet_loss) / (num_valid_triplets + 1e-16)
 
    if reg_mse:
        reg_loss = torch.mean((labels-predictions).abs()**2)
    else:
        reg_loss = torch.mean((labels-predictions).abs())
        
    union_loss = reg_loss + alpha*triplet_loss
    return union_loss, triplet_loss, reg_loss, num_positive_triplets


'''
ada_batch_hard_triplet_loss
'''
def get_anchor_positive_triplet_mask(labels, cliff):

    indices_equal = torch.eye(labels.shape[0]).bool().cuda() #--
    indices_not_equal = torch.logical_not(indices_equal)                 

    #labels = torch.unsqueeze(labels, -1)
    target_l1_dist = torch.cdist(labels,labels,p=1) 
    labels_equal = target_l1_dist < cliff
    mask = torch.logical_and(indices_not_equal, labels_equal)
    return mask
def get_anchor_negative_triplet_mask(labels, cliff):
    #labels = torch.unsqueeze(labels, -1)
    target_l1_dist = torch.cdist(labels,labels,p=1) 
    labels_equal = target_l1_dist < cliff
    mask = torch.logical_not(labels_equal)
    return mask
def ada_batch_hard_triplet_loss(embeddings, 
                               predictions, 
                               labels, 
                               device, 
                               cliff=0.5, 
                               alpha=1.0, 
                               squared=False,
                               reg_mse = False
                               ):
    '''
    batch hard triplet loss
    '''
    pairwise_distances = pairwise_distance(embeddings)
    mask_anchor_positive = get_anchor_positive_triplet_mask(labels,cliff).float()
    anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_distances)
    hardest_positive_dist,hardest_positive_indice = torch.max(anchor_positive_dist, dim=1, keepdims=True) 
    
    mask_anchor_negative = get_anchor_negative_triplet_mask(labels,cliff).float()
    max_anchor_negative_dist = torch.max(pairwise_distances, 1, keepdims=True).values  
    anchor_negative_dist = pairwise_distances + max_anchor_negative_dist * (1.0 - mask_anchor_negative) 
    hardest_negative_dist,hardest_negative_indice = torch.min(anchor_negative_dist, dim=1, keepdims=True)
    #labels = torch.unsqueeze(labels,-1)

    label_dist = (labels - labels.T).abs()
    
    positive_label_dist = torch.zeros([len(hardest_negative_dist),1]).to(device)
    negative_label_dist = torch.zeros([len(hardest_negative_dist),1]).to(device)
    
    for i,label_indice in enumerate(hardest_positive_indice):
        positive_label_dist[i][0] = label_dist[i][int(hardest_positive_indice[0])]
    for i,label_indice in enumerate(hardest_negative_indice):
        negative_label_dist[i][0] = label_dist[i][int(hardest_negative_indice[0])]

    margin = positive_label_dist - negative_label_dist
    
    triplet_loss = hardest_positive_dist - hardest_negative_dist + margin
    triplet_loss = torch.maximum(triplet_loss, torch.tensor([0.0]).to(device))
    
    valid_triplets = (triplet_loss> 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    
    triplet_loss = torch.sum(triplet_loss*valid_triplets) / (num_positive_triplets + 1e-16)

    if reg_mse:
        reg_loss = torch.mean((labels-predictions).abs()**2)
    else:
        reg_loss = torch.mean((labels-predictions).abs())
        
    union_loss = reg_loss + alpha*triplet_loss
    return union_loss, triplet_loss, reg_loss, num_positive_triplets

