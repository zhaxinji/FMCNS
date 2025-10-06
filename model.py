import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
import math
import copy
import importlib
import itertools
from time import time
from logging import getLogger
from utils import get_local_time, early_stopping, dict2str, TopKEvaluator, build_sim, compute_normalized_laplacian

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding) or isinstance(module, nn.Parameter):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class EmbLoss(nn.Module):
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss

class AbstractRecommender(nn.Module):
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def __str__(self):
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class GeneralRecommender(AbstractRecommender):
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        self.batch_size = config['train_batch_size']
        self.device = config['device']

        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class FlowModel(nn.Module):
    def __init__(self, config):
        super(FlowModel, self).__init__()
        self.embedding_dim = config['embedding_size']
        self.time_emb_dim = config['time_embedding_size']
        
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.embedding_dim + self.time_emb_dim, config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_size'], self.embedding_dim)
        )
        
        self.apply(xavier_normal_initialization)

    def forward(self, x, t):
        time_emb = timestep_embedding(t, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        h = torch.cat([x, emb], dim=-1)
        output = self.mlp_layers(h)
        return output

class FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_steps = config['n_steps']
        self.s_steps = config['s_steps']
        self.embedding_dim = config['embedding_size']
        self.time_steps = torch.linspace(0, 1, self.n_steps + 1)
        
        self.flow_model = FlowModel(config)
        
        self.w_q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)

    def q_sample(self, x_start, t_batch, x0=None):
        if x0 is None:
            x0 = torch.zeros_like(x_start)
        
        t = t_batch.unsqueeze(-1)
        random_mask = torch.rand_like(x_start, dtype=torch.float32) <= t
        xt = torch.where(random_mask, x_start, x0)
        return xt

    def p_losses(self, x_start, t_start, v_start, t_batch, x0=None):
        if x0 is None:
            x0 = torch.zeros_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t_batch=t_batch, x0=x0)
        
        t_emb = timestep_embedding(t_batch, self.embedding_dim).cuda()
        flow_input = torch.cat([x_noisy.unsqueeze(1), t_start.unsqueeze(1), v_start.unsqueeze(1), t_emb.unsqueeze(1)], dim=1)
        predicted_x = self.self_attention(flow_input)
        
        loss = F.mse_loss(x_start, predicted_x)
        return loss, predicted_x

    @torch.no_grad()
    def p_sample(self, x_t, t_t, v_t, t_batch, t_index):
        t_batch = t_batch.cuda()
        
        t_emb = timestep_embedding(t_batch, self.embedding_dim).cuda()
        flow_input = torch.cat([x_t.unsqueeze(1), t_t.unsqueeze(1), v_t.unsqueeze(1), t_emb.unsqueeze(1)], dim=1)
        x_start = self.self_attention(flow_input)
        
        if t_index == 0:
            return x_start
        else:
            return x_start

    @torch.no_grad()
    def sample(self, x0, t_start, v_start, item_frequencies):
        x_t = x0
        t_t = t_start
        v_t = v_start
        
        start_step = max(0, self.n_steps - self.s_steps)
        
        for n in range(start_step, self.n_steps):
            if n == self.n_steps - 1:
                t_tensor = torch.full((x_t.shape[0],), n / self.n_steps, dtype=torch.float32)
                x_t = self.p_sample(x_t, t_t, v_t, t_tensor, 0)
                break
                
            t_tensor = torch.full((x_t.shape[0],), n / self.n_steps, dtype=torch.float32)
            x1_pred = self.p_sample(x_t, t_t, v_t, t_tensor, 1)
            
            t_curr = n / self.n_steps
            v = (x1_pred - x_t) / (1 - t_curr + 1e-8)
            x_t = x_t + v / self.n_steps
            x_t = (x_t >= 0.5).float()
            x_t = torch.logical_or(x0.bool(), x_t.bool()).float()
                
        return x_t

    def self_attention(self, features):
        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)

        attn = q.mul(self.embedding_dim ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v
        y = features.mean(dim=-2)

        return y

class LightGCN_Encoder(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCN_Encoder, self).__init__(config, dataset)
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.n_layers = 3 if config['n_layers'] is None else config['n_layers']
        self.layers = [self.latent_size] * self.n_layers

        self.drop_ratio = 1.0
        self.drop_flag = True

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size)))
        })

        return embedding_dict

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse_coo_tensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - rate))

    def forward(self, inputs):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        users, items = inputs[0], inputs[1]
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings

class FMCNS(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FMCNS, self).__init__(config, dataset)
        self.config = config
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']
        self.aug_weight = config['aug_weight']
        
        self.epoch_idx = 0

        self.n_nodes = self.n_users + self.n_items
        self.flow_weight = config['flow_weight']
        self.w = config['w']
        self.c = config['c']
        self.weight = config['weight']
        self.sample_k = config['sample_k']
        self.intervention_beta = config['intervention_beta']
        
        self.flow_matching = FlowMatching(self.config)
        self.n_steps = config['n_steps']
        self.s_steps = config['s_steps']
        
        self.sample_x = None
        self.x_start = None
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.interaction_matrix_csr = dataset.inter_matrix(form='csr').astype(np.float32)
        self.interaction_matrix_dense = torch.tensor(self.interaction_matrix_csr.todense())
        
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_flowcf_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, weights_only=True).to(self.device)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        self.item_frequencies = self.get_item_frequencies()
        self.freq_mean = self.item_frequencies.mean()
        self.freq_std = self.item_frequencies.std()

    def get_item_frequencies(self):
        item_counts = torch.zeros(self.n_items, device=self.device)
        for row, col in zip(self.interaction_matrix.row, self.interaction_matrix.col):
            item_counts[col] += 1
        item_frequencies = item_counts / self.n_users
        return item_frequencies

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)
    
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self, epoch_idx):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        
        self.epoch_idx = epoch_idx
        
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_indices[:, degree_idx]
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse_coo_tensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj, predicted_x, items):
        h = self.item_id_embedding.weight
            
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
            
        h_flow = h.clone()
        predicted_x = self.w * predicted_x + (1 - self.w) * h[items, :]
        h_flow[items, :] = predicted_x
    
        ego_embeddings = torch.cat((self.user_embedding.weight, h_flow), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        items = torch.cat((pos_items, neg_items), dim=0)
        
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        
        t_batch = torch.rand(items.shape[0], device=items.device)
        
        flow_items = self.item_id_embedding.weight[items, :]
        flow_v = image_feats[items, :]
        flow_t = text_feats[items, :]
        
        x0 = torch.bernoulli(self.item_frequencies[items].unsqueeze(-1).expand(-1, self.embedding_dim)).to(items.device)
        
        flow_loss, predicted_x = self.flow_matching.p_losses(flow_items, flow_t, flow_v, t_batch, x0)
        
        ua_embeddings, ia_embeddings = self.forward(self.masked_adj, predicted_x, items)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        neg_i_g_embeddings_flow = ia_embeddings[neg_items]
        
        if self.epoch_idx > 0:   
            neg_flow_items = self.sample_neg_items(pos_items, users, ia_embeddings)
            neg_i_g_embeddings_flow = ia_embeddings[neg_flow_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        batch_mf_loss_flow = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings_flow)

        return batch_mf_loss * (1 - self.weight) + self.weight * batch_mf_loss_flow + self.flow_weight * flow_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        items = torch.arange(self.item_id_embedding.weight.shape[0])
        restore_user_e, restore_item_e = self.forward(self.norm_adj, self.sample_x, items)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    
    def sample(self):
        text_feats, image_feats = None, None
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)

        x0 = torch.bernoulli(self.item_frequencies.unsqueeze(-1).expand(-1, self.embedding_dim)).to(self.device)
        
        predicted_x = self.flow_matching.sample(
            x0, text_feats, image_feats, self.item_frequencies
        )
        self.sample_x = predicted_x

    def sample_neg_items(self, pos_items, users, ia_embeddings):
        freq_bias = self.item_frequencies[pos_items].unsqueeze(-1)
        causal_mask = torch.sigmoid(-self.intervention_beta * (freq_bias - self.freq_mean) / (self.freq_std + 1e-8))
        
        flow_enhanced_emb = self.sample_x[pos_items] if self.sample_x is not None else ia_embeddings[pos_items]
        debiased_embeddings = causal_mask * flow_enhanced_emb + (1 - causal_mask) * ia_embeddings[pos_items]
        
        num_candidates = min(int(0.15 * ia_embeddings.shape[0]), 1000)
        candidate_idx = torch.randperm(ia_embeddings.shape[0], device=self.device)[:num_candidates]
        candidate_embs = ia_embeddings[candidate_idx]
        
        similarity_matrix = torch.matmul(debiased_embeddings, candidate_embs.t())
        
        user_history = self.interaction_matrix_dense[users.cpu()][:, candidate_idx.cpu()].to(self.device)
        masked_similarity = similarity_matrix.masked_fill(user_history.bool(), float('-inf'))
        
        k = max(1, int(self.sample_k * num_candidates))
        _, top_k_idx = torch.topk(masked_similarity, k=k, dim=1)
        
        random_select = torch.randint(0, k, (len(pos_items),), device=self.device)
        final_candidates = top_k_idx[torch.arange(len(pos_items)), random_select]
        
        return candidate_idx[final_candidates]

class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        raise NotImplementedError('Method [next] should be implemented.')

class Trainer(AbstractTrainer):
    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)
        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.flow_lr = config['flow_lr']

        self.start_epoch = 0
        self.cur_step = 0
        self.softmax = nn.Softmax(dim=-1)
        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        lr_scheduler = config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']

    def _build_optimizer(self):
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_inter = interaction.clone()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                return loss, torch.tensor(0.0)
            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                first_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses
                    
                if self._check_nan(loss):
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()
            
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, is_test=False):
        valid_result = self.evaluate(valid_data, is_test)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            self.model.pre_epoch_processing(epoch_idx)
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            
            if torch.is_tensor(train_loss):
                break
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            
            post_info = self.model.post_epoch_processing()

            if (epoch_idx + 1) % self.eval_step == 0:
                self.model.sample()
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, is_test=False)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                
                _, test_result = self._valid_epoch(test_data, is_test=True)
                
                if verbose:
                    log_str = f"Epoch {epoch_idx+1}/{self.epochs}, "
                    metrics_str = []
                    for metric in ['recall', 'ndcg']:
                        for k in [10, 20]:
                            key = f'{metric}@{k}'
                            if key in test_result:
                                metrics_str.append(f"{key}: {test_result[key]:.4f}")
                    log_str += ", ".join(metrics_str)
                    self.logger.info(log_str)
                    self.logger.info("")
                    
                if update_flag:
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    if verbose:
                        self.logger.info(f'Training stopped at epoch {epoch_idx + 1}')
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        self.model.eval()
        
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
            batch_matrix_list.append(topk_index)
            
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

def get_model(model_name):
    if model_name in globals():
        return globals()[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")