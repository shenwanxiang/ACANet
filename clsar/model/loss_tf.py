# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:06:05 2022

@author: wanxiang.shen@u.nus.edu

Implement the ACA loss for Tensorflow 2.x
"""

import tensorflow as tf
from tensorflow.keras import losses
import numpy as np


class ACALoss(losses.Loss):
    
    r"""Creates a criterion that measures the activity cliff awareness (ACA) loss given an input
    tensors :math:`y_true`, :math:`y_pred`, :math:`y_emb`, an awareness factor :math:`𝑎` 
    and two cliff parameters :math:`cliff_lower` and :math:`cliff_upper` with a value greater than :math:`0`.

    This is used for increasing the activty cliff awareness in regression tasks of molecular property prediction. 
    It consists of two parts, the natural regression loss of mae or mse and an active cliff perception loss in latent space. 
    The ACALoss is described in detail in the paper `Online triplet contrastive learning enables efficient cliff 
    awareness in regression tasks of molecular property prediction`.


    The loss function for each sample in the mini-batch is:

    .. math::
        L(aca) = L(mae/mse) + 𝑎 * L(tsm)
        L(tsm) = ∑_(j=1)^M[|f_j^a-f_j^p |-|f_j^a-f_j^n |+m_j ]_+ 
        
    where the L_mae is the MAE loss, the L_tsm is the triplet loss with soft margin, 𝑎 is the awareness factor, 
    N is the number of samples in each batch, M is the number of the mined (i.e., the valid) triplets in each batch, 
    the y_j, f_j are the j-th true label and latent vectors, respectively. 
    The item m_j is the soft margin of the mined j-th triplet and is defined by:
    
    .. math::
        m_j=|y_j^a-y_j^n |-|y_j^a-y_j^p |

    where `a`, `p` and `n` are `anchor`, `positive` and `negative` examples of a mined triplet in a min-batch, respectively

    It can be seen that the L_tsm term is only determined by the true labels and the embedding vectors in the latent space.
    Therefore, this term is forcing the model to learn active cliffs in the latent space.
    

    Args:
        alpha (float, optional): awareness factor. Default: :math:`0.1`.
        cliff_lower (float, optional): The threshold for mining the postive samples. Default: ``1.0``
        cliff_upper (float, optional): The threshold for mining the negative samples. Default: ``1.0``
        squared (bool, optional): if True, the mse loss will be used, otherwise mae. The L(tsm) will also be squared.
        p (float, optional) – p value for the p-norm distance to calculate the distance of latent vectors ∈[0,∞]. Default: ``2.0``
        dev_mode (bool, optional): if False, only return the union loss
    Examples::
    ## developer mode
    >>> aca_loss = ACALoss(alpha=1e-1, cliff_lower = 0.2, cliff_upper = 1.0, squared = True, p =1.0, dev_mode = True)
    >>> loss, reg_loss, tsm_loss, n_mined_triplets, n_pos_triplets = aca_loss(labels, predictions, embeddings)
    >>> loss.backward()
    ## normal mode
    >>> aca_loss = ACALoss(dev_mode = False)
    >>> loss = aca_loss(labels, predictions, embeddings)
    >>> loss.backward()
    
    """
    
    def __init__(self, 
                 alpha: float = 1e-1, 
                 cliff_lower: float = 1.0, 
                 cliff_upper: float = 1.0,
                 squared: bool = False, 
                 p: float = 2.0,
                 dev_mode = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.cliff_lower = cliff_lower
        self.cliff_upper = cliff_upper
        self.squared = squared
        self.p = p
        self.dev_mode = dev_mode
        
    def call(self, labels, predictions, embeddings):
        # implement the forward pass of the ACA loss here
        # return the loss tensor
        return _aca_loss(labels, predictions, embeddings, alpha=self.alpha, 
                        cliff_lower=self.cliff_lower, cliff_upper=self.cliff_upper,
                        squared=self.squared, p = self.p, dev_mode = self.dev_mode)


def pairwise_distance(embeddings, squared=True, p = 1):
    pdist = tf.norm(embeddings - embeddings[:, tf.newaxis], ord=p, axis=-1)
    
    ## normalized l1/l2 distance along the vector size
    # N = np.power(embeddings.shape[1], 1/p)
    # pdist = pdist / N

    if squared:
        pdist = pdist**2
    return pdist



def get_triplet_mask(labels, cliff_lower=0.2, cliff_upper=1.0):
    indices_equal = tf.linalg.eye(tf.shape(labels)[0]).numpy().astype(bool)
    indices_not_equal = ~indices_equal
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(
        tf.logical_and(i_not_equal_j, i_not_equal_k),
        j_not_equal_k
    )
    target_l1_dist = tf.norm(labels - labels[:, tf.newaxis], ord=1, axis=-1)
    label_equal = target_l1_dist < cliff_lower
    label_unequal = target_l1_dist >= cliff_upper
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_unequal_k = tf.expand_dims(label_unequal, 1)
    valid_labels = tf.logical_and(i_equal_j, i_unequal_k)
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask
 

def _aca_loss(labels,
              predictions, 
              embeddings,
              alpha=1e-1,
              cliff_lower=0.2,
              cliff_upper=1.0,
              squared = False,
              p = 2.0,
              dev_mode = True
              ):
    if squared:
        reg_loss = tf.math.reduce_mean(tf.math.abs(labels-predictions)**2)
    else:
        reg_loss = tf.math.reduce_mean(tf.math.abs(labels-predictions))


    # implement the rest of the ACA loss 
    labels_dist = pairwise_distance(embeddings=labels, squared=squared, p = p)
    margin_pos = tf.expand_dims(labels_dist, 2)
    margin_neg = tf.expand_dims(labels_dist, 1)
    margin = margin_neg - margin_pos
    pairwise_dis = pairwise_distance(embeddings=embeddings, squared=squared, p = p)
    anchor_positive_dist = tf.expand_dims(pairwise_dis, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(
        anchor_positive_dist.shape)
    anchor_negative_dist = tf.expand_dims(pairwise_dis, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(
        anchor_negative_dist.shape)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = get_triplet_mask(labels=labels, 
                            cliff_lower=cliff_lower, 
                            cliff_upper=cliff_upper)
        
    mask = tf.cast(mask, triplet_loss.dtype)

    n_mined_triplets = tf.reduce_sum(mask)  # total number of mined triplets
    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, tf.constant(0.0, shape=(1,), dtype=triplet_loss.dtype))
    pos_triplets = tf.cast(triplet_loss > 1e-16, triplet_loss.dtype)
    n_pos_triplets = tf.reduce_sum(pos_triplets)  # torch.where
    tsm_loss = tf.reduce_sum(triplet_loss) / (n_mined_triplets + 1e-16)
    loss = reg_loss + alpha*tsm_loss
    if dev_mode:
        return loss, reg_loss, tsm_loss, n_mined_triplets, n_pos_triplets
    else:
        return loss
         

def get_best_cliff(labels, cliffs = list(np.arange(0.1, 3.2, 0.1).round(2))):
    '''
    Get the best cliff lower and upper values. Under these value, we can mine the maximal No. of triplets.
    '''
    low_up_trps = []
    n = 0
    best_lower = 0
    best_upper = 0
    for lower in cliffs:
        for upper in cliffs:
            if upper >= lower:
                mask = get_triplet_mask(
                    labels, cliff_lower=lower, cliff_upper=upper)
                mask = tf.cast(mask, tf.float32)
                n_mined_trps = tf.math.reduce_sum(mask).numpy()
                if n_mined_trps > n:
                    n = n_mined_trps
                    best_lower = lower
                    best_upper = upper

    return best_lower, best_upper, n