import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
import torch.nn.functional as F

def add_full_rrwp(data, walk_length):
    pes = []
    for ids in range(data.shape[0]):
        dt = data[ids].squeeze()
        pe = add_every_rrwp(dt, walk_length)
        pes.append(pe)
    return torch.stack(pes)

def add_every_rrwp(data,

                  walk_length=8,
                  # attr_name_abs="rrwp",  # name: 'rrwp'
                  # attr_name_rel="rrwp",  # name: ('rrwp_idx', 'rrwp_val')
                  add_identity=True,
                  spd=False,
                  **kwargs
                  ):
    # edge_index = torch.column_stack(torch.where(data > 0.3)).T.contiguous()
    #
    # device = edge_index.device
    # ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
    # num_nodes = data.shape[0]
    # edge_weight = torch.zeros((num_nodes, num_nodes), device=device)
    # edge_weight[edge_index[0], edge_index[1]] = 1.0
    # adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=device), (num_nodes, num_nodes))
    # adj = adj.to_dense()
    # data = (data + 1.) / 2.
    edge_index = torch.column_stack(torch.where(data > 0.3)).T.contiguous()

    device = edge_index.device
    num_nodes = data.shape[0]
    edge_weight = data[edge_index[0], edge_index[1]]
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    adj = adj.to_dense()
    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=device))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1).cuda()  # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

    # rel_pe = SparseTensor.from_dense(pe, has_value=True)

    # rel_pe = torch.sparse_coo_tensor(pe.nonzero().t(), pe[pe.nonzero()].view(-1), pe.size())

    # rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    # rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    # if spd:
    #     spd_idx = walk_length - torch.arange(walk_length)
    #     val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
    #     val = torch.argmax(val, dim=-1)
    #     rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
    #     abs_pe = torch.zeros_like(abs_pe)

    # data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    # data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    # data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    # data.log_deg = torch.log(deg + 1)
    # data.deg = deg.type(torch.long)

    return abs_pe

class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=4,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        self.pos_embed_dim = config.model.pos_embed_dim
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)
        if self.pos_encoding == 'rrwp':
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim



        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        if self.pos_encoding == 'rrwp':
            pos_emb = add_full_rrwp(node_feature, self.pos_embed_dim)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
