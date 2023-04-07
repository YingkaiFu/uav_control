# -*- coding: utf-8 -*

import numpy as np
import copy
from typing import Optional, List
# from model.atten.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import time
import torch
import torch.nn as nn

from model.common_opr.common_block import conv_bn_relu
from model.module_base import ModuleBase
from model.task_head.taskhead_base import TRACK_HEADS, VOS_HEADS
from pipeline.utils.bbox import xyxy2cxywh

torch.set_printoptions(precision=8)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, src_search, mask_search, pos_temp, pos_search):
        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = None
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = None
        mask_temp = None
        mask_search = None

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search,
                                                  src1_key_padding_mask=mask_temp,
                                                  src2_key_padding_mask=mask_search,
                                                  pos_src1=pos_temp,
                                                  pos_src2=pos_search)
        hs = self.decoder(memory_search, memory_temp,
                          tgt_key_padding_mask=mask_search,
                          memory_key_padding_mask=mask_temp,
                          pos_enc=pos_temp, pos_dec=pos_search)
        return hs.unsqueeze(0).transpose(1, 2)


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        output1 = src1
        output2 = src2

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)

        return output1, output2


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):
        q1 = k1 = self.with_pos_embed(src1, pos_src1)
        src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
                               key_padding_mask=src1_key_padding_mask)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)

        q2 = k2 = self.with_pos_embed(src2, pos_src2)
        src22 = self.self_attn2(q2, k2, value=src2, attn_mask=src2_mask,
                               key_padding_mask=src2_key_padding_mask)[0]
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)


        src12 = self.multihead_attn1(query=self.with_pos_embed(src1, pos_src1),
                                   key=self.with_pos_embed(src2, pos_src2),
                                   value=src2, attn_mask=src2_mask,
                                   key_padding_mask=src2_key_padding_mask)[0]
        src22 = self.multihead_attn2(query=self.with_pos_embed(src2, pos_src2),
                                   key=self.with_pos_embed(src1, pos_src1),
                                   value=src1, attn_mask=src1_mask,
                                   key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)

        return src1, src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(settings):
    return FeatureFusionNetwork(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward,
        num_featurefusion_layers=settings.featurefusion_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Attention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # add pos to src
        if self.divide_norm:
            # print("encoder divide by norm")
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def check_inf(tensor):
    return torch.isinf(tensor.detach()).any()


def check_nan(tensor):
    return torch.isnan(tensor.detach()).any()


def check_valid(tensor, type_name):
    if check_inf(tensor):
        print("%s is inf." % type_name)
    if check_nan(tensor):
        print("%s is nan" % type_name)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, divide_norm=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        decoder_norm = nn.LayerNorm(d_model)
        if num_decoder_layers == 0:
            self.decoder = None
        else:
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, mask, query_embed, pos_embed, mode="all", return_encoder_output=False,memo=None):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)
        :return:
        """
        assert mode in ["all", "encoder","decoder"]
        if self.encoder is None:
            memory = feat
        else:
            memory = self.encoder(feat, src_key_padding_mask=None, pos=None)
        if mode == "encoder":
            return memory
        elif mode == "all":
            assert len(query_embed.size()) in [2, 3]
            if len(query_embed.size()) == 2:
                bs = feat.size(1)
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N,C) --> (N,1,C) --> (N,B,C)
            if self.decoder is not None:
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                                  pos=None, query_pos=query_embed)
            else:
                hs = query_embed.unsqueeze(0)
            if return_encoder_output:
                return hs.transpose(1, 2), memory # (1, B, N, C)
            else:
                return hs.transpose(1, 2) # (1, B, N, C)
        elif mode=='decoder':
            assert len(query_embed.size()) in [2, 3]
            if len(query_embed.size()) == 2:
                bs = feat.size(1)
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N,C) --> (N,1,C) --> (N,B,C)
            if self.decoder is not None:
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, memo, memory_key_padding_mask=None,
                                  pos=None, query_pos=query_embed)
            else:
                hs = query_embed.unsqueeze(0)
            if return_encoder_output:
                return hs.transpose(1, 2), memory # (1, B, N, C)
            else:
                return hs.transpose(1, 2) # (1, B, N, C)

class Transformer1(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, divide_norm=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer_no_attention(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer_no_attention(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        decoder_norm = nn.LayerNorm(d_model)
        if num_decoder_layers == 0:
            self.decoder = None
        else:
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, mask, query_embed, pos_embed, mode="all", return_encoder_output=False,memo=None):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)
        :return:
        """
        assert mode in ["all", "encoder","decoder"]
        if self.encoder is None:
            memory = feat
        else:
            memory = self.encoder(feat, src_key_padding_mask=None, pos=None)
        if mode == "encoder":
            return memory
        elif mode == "all":
            assert len(query_embed.size()) in [2, 3]
            if len(query_embed.size()) == 2:
                bs = feat.size(1)
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N,C) --> (N,1,C) --> (N,B,C)
            if self.decoder is not None:
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                                  pos=None, query_pos=query_embed)
            else:
                hs = query_embed.unsqueeze(0)
            if return_encoder_output:
                return hs.transpose(1, 2), memory # (1, B, N, C)
            else:
                return hs.transpose(1, 2) # (1, B, N, C)
        elif mode=='decoder':
            assert len(query_embed.size()) in [2, 3]
            if len(query_embed.size()) == 2:
                bs = feat.size(1)
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N,C) --> (N,1,C) --> (N,B,C)
            if self.decoder is not None:
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, memo, memory_key_padding_mask=None,
                                  pos=None, query_pos=query_embed)
            else:
                hs = query_embed.unsqueeze(0)
            if return_encoder_output:
                return hs.transpose(1, 2), memory # (1, B, N, C)
            else:
                return hs.transpose(1, 2) # (1, B, N, C)

class Transformer_MAN_No_attention(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, divide_norm=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer_no_attention(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer_no_attention(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        decoder_norm = nn.LayerNorm(d_model)
        if num_decoder_layers == 0:
            self.decoder = None
        else:
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, mask, query_embed, pos_embed, mode="all", return_encoder_output=False,memo=None):
        """

        :param feat: (H1W1+H2W2, bs, C)
        :param mask: (bs, H1W1+H2W2)
        :param query_embed: (N, C) or (N, B, C)
        :param pos_embed: (H1W1+H2W2, bs, C)
        :param mode: run the whole transformer or encoder only
        :param return_encoder_output: whether to return the output of encoder (together with decoder)
        :return:
        """
        assert mode in ["all", "encoder","decoder"]
        if self.encoder is None:
            memory = feat
        else:
            memory = self.encoder(feat, src_key_padding_mask=None, pos=None)
        if mode == "encoder":
            return memory
        elif mode == "all":
            assert len(query_embed.size()) in [2, 3]
            if len(query_embed.size()) == 2:
                bs = feat.size(1)
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N,C) --> (N,1,C) --> (N,B,C)
            if self.decoder is not None:
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                                  pos=None, query_pos=query_embed)
            else:
                hs = query_embed.unsqueeze(0)
            if return_encoder_output:
                return hs.transpose(1, 2), memory # (1, B, N, C)
            else:
                return hs.transpose(1, 2) # (1, B, N, C)
        elif mode=='decoder':
            assert len(query_embed.size()) in [2, 3]
            if len(query_embed.size()) == 2:
                bs = feat.size(1)
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (N,C) --> (N,1,C) --> (N,B,C)
            if self.decoder is not None:
                tgt = torch.zeros_like(query_embed)
                hs = self.decoder(tgt, memo, memory_key_padding_mask=None,
                                  pos=None, query_pos=query_embed)
            else:
                hs = query_embed.unsqueeze(0)
            if return_encoder_output:
                return hs.transpose(1, 2), memory # (1, B, N, C)
            else:
                return hs.transpose(1, 2) # (1, B, N, C)




class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_intermediate=False):
        if return_intermediate:
            output_list = []
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)
                if self.norm is None:
                    output_list.append(output)
            if self.norm is not None:
                output = self.norm(output)
                output_list.append(output)
            return output_list
        else:
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)

            if self.norm is not None:
                output = self.norm(output)

            return output


class TransformerEncoderLite(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        assert self.num_layers == 1

    def forward(self, seq_dict, return_intermediate=False, part_att=False):
        if return_intermediate:
            output_list = []

            # for layer in self.layers:
            output = self.layers[-1](seq_dict, part_att=part_att)
            if self.norm is None:
                output_list.append(output)
            if self.norm is not None:
                output = self.norm(output)
                output_list.append(output)
            return output_list
        else:
            output = self.layers[-1](seq_dict, part_att=part_att)

            if self.norm is not None:
                output = self.norm(output)

            return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # add pos to src
        if self.divide_norm:
            # print("encoder divide by norm")
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayer_no_attention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(src, pos)  # add pos to src
        # if self.divide_norm:
        #     # print("encoder divide by norm")
        #     q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
        #     k = k / torch.norm(k, dim=-1, keepdim=True)
        # src2 = self.self_attn(q, k, src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoderLayerLite(nn.Module):
    """search region features as queries, concatenated features as keys and values"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, seq_dict, part_att=False):
        """
        seq_dict: sequence dict of both the search region and the template (concatenated)
        """
        if part_att:
            # print("using part attention")
            q = self.with_pos_embed(seq_dict["feat_x"], seq_dict["pos_x"])  # search region as query
            k = self.with_pos_embed(seq_dict["feat_z"], seq_dict["pos_z"])  # template as key
            v = seq_dict["feat_z"]
            key_padding_mask = seq_dict["mask_z"]
            # print(q.size(), k.size(), v.size())
        else:
            q = self.with_pos_embed(seq_dict["feat_x"], seq_dict["pos_x"])  # search region as query
            k = self.with_pos_embed(seq_dict["feat"], seq_dict["pos"])  # concat as key
            v = seq_dict["feat"]
            key_padding_mask = seq_dict["mask"]
        if self.divide_norm:
            raise ValueError("divide norm is not supported.")
        # s = time.time()
        src2 = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)[0]
        if torch.isnan(src2).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        src = q + self.dropout1(src2)
        src = self.norm1(src)
        # e1 = time.time()
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # e2 = time.time()
        # print("self-attention time: %.1f" % ((e1-s) * 1000))
        # print("MLP time: %.1f" % ((e2-e1) * 1000))
        return src

    def forward(self, seq_dict, part_att=False):
        if self.normalize_before:
            raise ValueError("PRE-NORM is not supported now")
        return self.forward_post(seq_dict, part_att=part_att)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # self-attention
        q = k = self.with_pos_embed(tgt, query_pos)  # Add object query to the query and key
        if self.divide_norm:
            q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
            k = k / torch.norm(k, dim=-1, keepdim=True)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # mutual attention
        queries, keys = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
        if self.divide_norm:
            queries = queries / torch.norm(queries, dim=-1, keepdim=True) * self.scale_factor
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)
        tgt2 = self.multihead_attn(queries, keys, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayer_no_attention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # self-attention
        # q = k = self.with_pos_embed(tgt, query_pos)  # Add object query to the query and key
        # if self.divide_norm:
        #     q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
        #     k = k / torch.norm(k, dim=-1, keepdim=True)
        # tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        # mutual attention
        queries, keys = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(memory, pos)
        if self.divide_norm:
            queries = queries / torch.norm(queries, dim=-1, keepdim=True) * self.scale_factor
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)
        tgt2 = self.multihead_attn(queries, keys, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        nhead=cfg.MODEL.TRANSFORMER.NHEADS,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
        num_decoder_layers=cfg.MODEL.TRANSFORMER.DEC_LAYERS,
        normalize_before=cfg.MODEL.TRANSFORMER.PRE_NORM,
        return_intermediate_dec=False,  # we use false to avoid DDP error,
        divide_norm=cfg.MODEL.TRANSFORMER.DIVIDE_NORM
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x.to(prob_vec.device) * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y.to(prob_vec.device) * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

def get_xy_ctr(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in torch)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = torch.linspace(0., fm_height - 1., fm_height).reshape(
        1, fm_height, 1, 1).repeat(1, 1, fm_width,
                                   1)  # .broadcast([1, fm_height, fm_width, 1])
    x_list = torch.linspace(0., fm_width - 1., fm_width).reshape(
        1, 1, fm_width, 1).repeat(1, fm_height, 1,
                                  1)  # .broadcast([1, fm_height, fm_width, 1])
    xy_list = score_offset + torch.cat([x_list, y_list], 3) * total_stride
    xy_ctr = xy_list.repeat(batch, 1, 1, 1).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    # TODO: consider use float32 type from the beginning of this function
    xy_ctr = xy_ctr.type(torch.Tensor)
    return xy_ctr


def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = np.linspace(0., fm_height - 1.,
                         fm_height).reshape(1, fm_height, 1, 1)
    y_list = y_list.repeat(fm_width, axis=2)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
    x_list = x_list.repeat(fm_height, axis=1)
    xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
    xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    # TODO: consider use float32 type from the beginning of this function
    xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
    return xy_ctr


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred

def get_box_cxy(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred

@TRACK_HEADS.register
@VOS_HEADS.register
class DenseboxHead(ModuleBase):
    r"""
    Densebox Head for siamfcpp

    Hyper-parameter
    ---------------
    total_stride: int
        stride in backbone
    score_size: int
        final feature map
    x_size: int
        search image size
    num_conv3x3: int
        number of conv3x3 tiled in head
    head_conv_bn: list
        has_bn flag of conv3x3 in head, list with length of num_conv3x3
    head_width: int
        feature width in head structure
    conv_weight_std: float
        std for conv init
    """
    default_hyper_params = dict(
        total_stride=8,
        score_size=17,
        x_size=303,
        num_conv3x3=3,
        head_conv_bn=[False, False, False],
        head_width=128,
        conv_weight_std=0.0001,
        input_size_adapt=False,
    )

    def __init__(self):
        super(DenseboxHead, self).__init__()
        head_numbers = self._hyper_params['head_width']
        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))
        self.bottleneck=nn.Conv2d(head_numbers, 128, kernel_size=1)
        self.bottleneck1=nn.Conv2d(head_numbers, 128, kernel_size=1)

        # self.fusion = FeatureFusionNetwork(d_model=128,nhead=4,num_featurefusion_layers=2)
        # self.mlp_head = MLP(head_numbers,head_numbers,4,3)
        self.query_embed = nn.Embedding(1, 128)
        self.query_embed1 = nn.Embedding(1, 128)
        self.transformer = Transformer(d_model=128,dropout=0.1,nhead=4,num_encoder_layers=3,num_decoder_layers=3,
                                       normalize_before=False,divide_norm=False)
        self.transformer1 = Transformer(d_model=128,dropout=0.1,nhead=4,num_encoder_layers=2,num_decoder_layers=1,
                                       normalize_before=False,divide_norm=False)
        self.cls_convs = []
        # self.self_attention = SimplifiedScaledDotProductAttention(d_model=head_numbers,h=8)
        self.box_head = Corner_Predictor(inplanes=128, channel=128, feat_sz=17, stride=17.8)
        self.adjust = conv_bn_relu(head_numbers,128,kszie=3,stride=2,pad=0,has_bn=False,has_relu=True).cuda()
        self.bbox_convs = []


    def forward_box_head(self, memory):
        enc_opt = memory.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, 17, 17)
        # run the corner head
        outputs_coord = self.box_head(opt_feat)
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}
        return out, outputs_coord_new


    def forward(self, c_out, c_x):
        # 5_8_no_bottle
        # num_conv3x3 = self._hyper_params['num_conv3x3']
        # cls = c_out
        # cc_out = self.adjust(c_x)
        # for i in range(0, num_conv3x3):
        #     cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
        #     # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        # # r_out_feat = self.bottleneck(cls)
        # r_out_feat = cls

        # # c_out_feat = self.bottleneck(bbox)
        # r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        # r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        # embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)
        # enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        # enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        # dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        # att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        # opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
        #     (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        # bs, Nq, C, HW = opt1.size()
        # opt_feat1 = opt1.view(-1, C, 17, 17)

        # 移动位置
        # num_conv3x3 = self._hyper_params['num_conv3x3']
        # cls = c_out
        # cc_out = self.adjust(c_x)
        # for i in range(0, num_conv3x3):
        #     cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
        #     # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        # # r_out_feat = self.bottleneck(cls)
        # r_out_feat = cls

        # # c_out_feat = self.bottleneck(bbox)
        # r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        # r_vec1 = c_x.flatten(2).permute(2, 0, 1)
        # # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        # embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)
        # enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        # enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        # dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        # att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        # opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
        #     (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        # bs, Nq, C, HW = opt1.size()
        # opt_feat1 = opt1.view(-1, C, 17, 17)


        # 最好性能的
        # '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
#         print(cls.shape,c_x.shape)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)
        
        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)  # (1, d, 1, C) (hw, d, C)
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        # '''

        # nips_ab_no_self_attention
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)  # (1, d, 1, C) (hw, d, C)
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # alexnet
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        # cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            c_x = getattr(self, 'bbox_p5_conv%d' % (i + 1))(c_x)
        r_out_feat = self.bottleneck(cls)
        cc_out = self.bottleneck1(c_x)
        # print(r_out_feat.shape)
        # print(cc_out.shape)
        # print(c_x.shape)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)  # (1, d, 1, C) (hw, d, C)
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''


        # best
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)
        # print(r_out_feat.shape)
        # print(cc_out.shape)
        # print(c_x.shape)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)  # (1, d, 1, C) (hw, d, C)
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''


 # short_cut
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)
        # print(r_out_feat.shape)
        # print(cc_out.shape)
        # print(c_x.shape)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)  # (1, d, 1, C) (hw, d, C)
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_mem1 = enc_mem1+r_vec1
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # nips_reverse_fusion
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        enc_mem1 = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True,mode='encoder')  # (1, d, 1, C) (hw, d, C)
        embed, enc_mem = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True)
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # nips_ab_only_matmul
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)  # (1, d, 1, C) (hw, d, C)
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # MLP_HEAD
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)
        # enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        # enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.transpose(3, 1).squeeze()  # (B, C, N)
        # att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        # opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
        #     (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        # bs, Nq, C, HW = opt1.size()
        # opt_feat1 = opt1.view(-1, C, 17, 17)
        outputs_coord = self.mlp_head(dec_opt1)
        outputs_coord_new = outputs_coord.view(dec_opt1.size()[0], 1, 4)
        out = {'pred_boxes': outputs_coord_new}
        # print(out)
        return out, outputs_coord_new
        '''

        # nips_ab_attention_in_man
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''


        # nips_ab_fuse_add
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True,mode='encoder')
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc = enc_mem + enc_mem1
        enc_opt1 = enc.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        # dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        # att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # nips_ab_fuse_with_eca_and_cfa
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)
        out = self.fusion(r_out_feat,None,cc_out,None,None,None)
        # print(out.shape,r_out_feat.shape)
        opt1 = out.permute(
            (1, 0, 3, 2)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # nips_ab_only_tan_self_fusion
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        # cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        # r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)
        # enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # nips_ab_only_man_no_fusion
        '''
        # num_conv3x3 = self._hyper_params['num_conv3x3']
        # cls = c_out
        cc_out = self.adjust(c_x)
        # for i in range(0, num_conv3x3):
        #     cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        # r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        # r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        # embed, enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)
        enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        # dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        # att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # ab: nips_ab_only_tan_no_decoder_and_fusion
        '''
        num_conv3x3 = self._hyper_params['num_conv3x3']
        cls = c_out
        # cc_out = self.adjust(c_x)
        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
            # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        r_out_feat = self.bottleneck(cls)

        # c_out_feat = self.bottleneck(bbox)
        r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        # r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        enc_mem = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True,mode='encoder')
        # enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True,mode='encoder')
        enc_opt1 = enc_mem.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        # dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        # att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        opt1 = (enc_opt1.unsqueeze(-1)).permute(
            (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt1.size()
        opt_feat1 = opt1.view(-1, C, 17, 17)
        '''

        # # 这是仅用上面分支的ablation study，encoder和decoder做了attention
        # num_conv3x3 = self._hyper_params['num_conv3x3']
        # cls = c_out
        # # cc_out = self.adjust(c_x)
        # for i in range(0, num_conv3x3):
        #     cls = getattr(self, 'cls_p5_conv%d' % (i + 1))(cls)
        #     # bbox = getattr(self, 'bbox_p5_conv%d' % (i + 1))(bbox)
        # r_out_feat = self.bottleneck(cls)
        # # # c_out_feat = self.bottleneck(bbox)
        # r_vec = r_out_feat.flatten(2).permute(2, 0, 1)
        # # r_vec1 = cc_out.flatten(2).permute(2, 0, 1)
        # # feat_vector = torch.cat([r_vec, c_vec], dim=0)
        # embed, enc_mem1 = self.transformer(r_vec, None, self.query_embed.weight, None, return_encoder_output=True)
        # # embed, enc_mem1 = self.transformer1(r_vec1, None, self.query_embed1.weight, None, return_encoder_output=True)
        # enc_opt1 = enc_mem1.transpose(0, 1)  # encoder output for the search region (B, HW, C)
        # dec_opt1 = embed.squeeze(0).transpose(1, 2)  # (B, C, N)
        # att1 = torch.matmul(enc_opt1, dec_opt1)  # (B, HW, N)
        # opt1 = (enc_opt1.unsqueeze(-1) * att1.unsqueeze(-2)).permute(
        #     (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        # bs, Nq, C, HW = opt1.size()
        # opt_feat1 = opt1.view(-1, C, 17, 17)


        # enc_opt2 = enc_mem.transpose(0, 1)
        # opt2 = (enc_opt2.unsqueeze(-1)).permute(
        #     (0, 3, 2, 1)).contiguous()
        # opt_feat2 = opt2.view(-1, C, 17, 17)
        # opt_feat = torch.cat([opt_feat1,opt_feat2],dim=1)

        # run the corner head
        outputs_coord = self.box_head(opt_feat1)
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}
        # print(out)
        return out, outputs_coord_new


        # # classification score
        # cls_score = self.cls_score_p5(cls)  #todo
        # cls_score = cls_score.permute(0, 2, 3, 1)
        # cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)
        # # center-ness score
        # ctr_score = self.ctr_score_p5(cls)  #todo
        # ctr_score = ctr_score.permute(0, 2, 3, 1)
        # ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)
        # # regression
        # offsets = self.bbox_offsets_p5(bbox)
        # offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride
        # if raw_output:
        #     return [cls_score, ctr_score, offsets]
        # # bbox decoding
        # if self._hyper_params["input_size_adapt"] and x_size > 0:
        #     score_offset = (x_size - 1 -
        #                     (offsets.size(-1) - 1) * self.total_stride) // 2
        #     fm_ctr = get_xy_ctr_np(offsets.size(-1), score_offset,
        #                            self.total_stride)
        #     fm_ctr = fm_ctr.to(offsets.device)
        # else:
        #     fm_ctr = self.fm_ctr.to(offsets.device)
        # bbox = get_box(fm_ctr, offsets)
        #
        # return [cls_score, ctr_score, bbox, cls]

    def update_params(self):
        x_size = self._hyper_params["x_size"]
        score_size = self._hyper_params["score_size"]
        total_stride = self._hyper_params["total_stride"]
        score_offset = (x_size - 1 - (score_size - 1) * total_stride) // 2
        self._hyper_params["score_offset"] = score_offset

        self.score_size = self._hyper_params["score_size"]
        self.total_stride = self._hyper_params["total_stride"]
        self.score_offset = self._hyper_params["score_offset"]
        # ctr = get_xy_ctr_np(self.score_size, self.score_offset,
        #                     self.total_stride)
        # self.fm_ctr = ctr
        # self.fm_ctr.require_grad = False

        self._make_conv3x3()
        self._make_conv_output()
        self._initialize_conv()

    def _make_conv3x3(self):
        num_conv3x3 = self._hyper_params['num_conv3x3']
        head_conv_bn = self._hyper_params['head_conv_bn']
        head_width = self._hyper_params['head_width']
        self.cls_conv3x3_list = []
        self.bbox_conv3x3_list = []
        for i in range(num_conv3x3):
            cls_conv3x3 = conv_bn_relu(head_width,
                                       head_width,
                                       stride=1,
                                       kszie=3,
                                       pad=0,
                                       has_bn=head_conv_bn[i])

            bbox_conv3x3 = conv_bn_relu(head_width,
                                        head_width,
                                        stride=1,
                                        kszie=4,
                                        pad=0,
                                        has_bn=head_conv_bn[i])
            setattr(self, 'cls_p5_conv%d' % (i + 1), cls_conv3x3)
            setattr(self, 'bbox_p5_conv%d' % (i + 1), bbox_conv3x3)
            self.cls_conv3x3_list.append(cls_conv3x3)
            # self.bbox_conv3x3_list.append(bbox_conv3x3)

    def _make_conv_output(self):
        head_width = self._hyper_params['head_width']
        # self.cls_score_p5 = conv_bn_relu(head_width,
        #                                  1,
        #                                  stride=1,
        #                                  kszie=1,
        #                                  pad=0,
        #                                  has_relu=False)
        # self.ctr_score_p5 = conv_bn_relu(head_width,
        #                                  1,
        #                                  stride=1,
        #                                  kszie=1,
        #                                  pad=0,
        #                                  has_relu=False)
        # self.bbox_offsets_p5 = conv_bn_relu(head_width,
        #                                     4,
        #                                     stride=1,
        #                                     kszie=1,
        #                                     pad=0,
        #                                     has_relu=False)

    def _initialize_conv(self, ):
        num_conv3x3 = self._hyper_params['num_conv3x3']
        conv_weight_std = self._hyper_params['conv_weight_std']

        # initialze head
        conv_list = []
        for i in range(num_conv3x3):
            conv_list.append(getattr(self, 'cls_p5_conv%d' % (i + 1)).conv)
            conv_list.append(getattr(self, 'bbox_p5_conv%d' % (i + 1)).conv)

        # conv_list.append(self.cls_score_p5.conv)
        # conv_list.append(self.ctr_score_p5.conv)
        # conv_list.append(self.bbox_offsets_p5.conv)
        # conv_classifier = [self.cls_score_p5.conv]
        # assert all(elem in conv_list for elem in conv_classifier)

        pi = 0.01
        bv = -np.log((1 - pi) / pi)
        for ith in range(len(conv_list)):
            # fetch conv from list
            conv = conv_list[ith]
            # torch.nn.init.normal_(conv.weight, std=0.01) # from megdl impl.
            torch.nn.init.normal_(
                conv.weight, std=conv_weight_std)  # conv_weight_std = 0.0001
            # nn.init.kaiming_uniform_(conv.weight, a=np.sqrt(5))  # from PyTorch default implementation
            # nn.init.kaiming_uniform_(conv.weight, a=0)  # from PyTorch default implementation
            # if conv in conv_classifier:
            #     torch.nn.init.constant_(conv.bias, torch.tensor(bv))
            # else:
                # torch.nn.init.constant_(conv.bias, 0)  # from PyTorch default implementation
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(conv.bias, -bound, bound)
