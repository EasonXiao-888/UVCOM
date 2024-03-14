from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from einops import rearrange, repeat

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class VITA(nn.Module):
    """
    capturing the T-series information of the object queries return by Deformable Transformers
    Specifically, The query for decoder is Language queries
    """
    def __init__(
        self,
        config,
        # input_dim: int,
        #window_size: int,
        #num_frame_queries: int,
        #num_frames: int,
        #num_queries: int,
        #nheads: int,
        #dim_feedforward: int,
        #enc_layers: int,
        #dec_layers: int,
        pre_norm: bool = False,
        aux_loss: bool = False

    ) -> None:
        super().__init__()
        self.window_size = config['window_size']
        self.num_frame_queries = config['num_frame_queries']
        self.num_queries = config['num_queries']
        self.num_heads = config['nheads']
        self.num_layers = config['dec_layers']
        self.num_frames = config['num_frames']
        self.aux_loss = aux_loss

        ### for decoder ------------------------------------------
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        ###------------------------------------------------------

        self.src_embed = nn.Identity()
        self.fq_pos = nn.Embedding(self.num_frame_queries, config['input_dim'])

        #### learnable query p.e.
        self.query_embed = nn.Embedding(self.num_queries, config['input_dim'])
        # self.query_feat = nn.Embedding(self.num_queries, config['input_dim'])

        self.decoder_norm = nn.LayerNorm(config['input_dim'])

        self.enc_layers = config['enc_layers']
        if self.enc_layers > 0:
            self.enc_self_attn = nn.ModuleList()
            self.enc_ffn = nn.ModuleList()
            for _ in range(self.enc_layers):
                self.enc_self_attn.append(
                    SelfAttentionLayer(
                        d_model=config['input_dim'],
                        nhead=config['nheads'],
                        dropout=0.1,
                        normalize_before=pre_norm,
                    ),
                )
                self.enc_ffn.append(
                    FFNLayer(
                        d_model=config['input_dim'],
                        dim_feedforward=config['dim_feedforward'],
                        dropout=0.1,
                        normalize_before=pre_norm,
                    )
                )
        
        ### this is for decoder layer
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=config['input_dim'],
                    nhead=config['nheads'],
                    dropout=0.1,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=config['input_dim'],
                    nhead=config['nheads'],
                    dropout=0.1,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=config['input_dim'],
                    dim_feedforward=config['dim_feedforward'],
                    dropout=0.1,
                    normalize_before=pre_norm,
                )
            )

    def forward(self, frame_query, language_query):
        """
        params:
        @frame_query: [L, T, B, query, C]
        @language_query: [B, C]
        """
        if not self.training:
            frame_query = frame_query[[-1]] #inference take the last layer
            # language_query = language_query[[-1]]
        
        L, T, B, NQ, C = frame_query.shape
        # B = BT // self.num_frames if self.training else 1
        # T = self.num_frames if self.training else BT // B

        frame_query = frame_query.reshape(L*B, T, NQ, C)
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous() #[t, query, L*B, C]

        if self.window_size > 0:
            pad = int(ceil(T / self.window_size)) * self.window_size - T
            _T = pad + T
            frame_query = F.pad(frame_query, (0,0,0,0,0,0,0,pad))   # _T, fQ, LB, C
            enc_mask = frame_query.new_ones(L*B, _T).bool()         # LB, _T
            enc_mask[:, :T] = False
        else:
            enc_mask = None
        
        frame_query = self.encode_frame_query(frame_query, enc_mask) #windows attention
        frame_query = frame_query[:T].flatten(0,1)              # TfQ, LB, C

        src = self.src_embed(frame_query) #[T*NQ, LB, C] = [L B C]
        dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L*B, 1).flatten(0, 1)
        
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L*B, 1) # cQ, LB, C
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L*B, 1) # cQ, LB, C
        # output = self.query_feat.weight.unsqueeze(1).repeat(1, L*B, 1) # cQ, LB, C
        language_query = language_query.unsqueeze(1).unsqueeze(1) #[B C]
        output = language_query.repeat(1, L, self.num_queries, 1) #[B, L, NQ, C]
        output = rearrange(output, 'b l nq c -> nq (l b) c')

        decoder_outputs = []
        for i in range(self.num_layers):
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=dec_pos, query_pos=query_embed
            )

            ### self-attention
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
        
            output = self.transformer_ffn_layers[i](
                    output
                )
        
            if (self.training and self.aux_loss) or (i == self.num_layers - 1):
                dec_out = self.decoder_norm(output) # cQ, LB, C
                dec_out = dec_out.transpose(0, 1)   # LB, cQ, C
                decoder_outputs.append(dec_out.view(L, B, self.num_queries, C))

        decoder_outputs = torch.stack(decoder_outputs, dim=0)   # D, L, B, cQ, C

        return decoder_outputs[-1] # L B queries C


    def encode_frame_query(self, frame_query, attn_mask):
            """
            input shape (frame_query)   : T, fQ, LB, C
            output shape (frame_query)  : T, fQ, LB, C
            """

            # Not using window-based attention if self.window_size == 0.
            if self.window_size == 0:
                return_shape = frame_query.shape        # T, fQ, LB, C
                frame_query = frame_query.flatten(0, 1) # TfQ, LB, C

                for i in range(self.enc_layers):
                    frame_query = self.enc_self_attn[i](frame_query)
                    frame_query = self.enc_ffn[i](frame_query)

                frame_query = frame_query.view(return_shape)
                return frame_query
            # Using window-based attention if self.window_size > 0.
            else:
                T, fQ, LB, C = frame_query.shape
                W = self.window_size
                Nw = T // W
                half_W = int(ceil(W / 2))

                window_mask = attn_mask.view(LB*Nw, W)[..., None].repeat(1, 1, fQ).flatten(1)

                _attn_mask  = torch.roll(attn_mask, half_W, 1)
                _attn_mask  = _attn_mask.view(LB, Nw, W)[..., None].repeat(1, 1, 1, W)    # LB, Nw, W, W
                _attn_mask[:,  0] = _attn_mask[:,  0] | _attn_mask[:,  0].transpose(-2, -1)
                _attn_mask[:, -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(-2, -1)
                _attn_mask[:, 0, :half_W, half_W:] = True
                _attn_mask[:, 0, half_W:, :half_W] = True
                _attn_mask  = _attn_mask.view(LB*Nw, 1, W, 1, W, 1).repeat(1, self.num_heads, 1, fQ, 1, fQ).view(LB*Nw*self.num_heads, W*fQ, W*fQ)
                shift_window_mask = _attn_mask.float() * -1000

                for layer_idx in range(self.enc_layers):
                    if self.training or layer_idx % 2 == 0:
                        frame_query = self._window_attn(frame_query, window_mask, layer_idx)
                    else:
                        frame_query = self._shift_window_attn(frame_query, shift_window_mask, layer_idx)
                return frame_query

    def _window_attn(self, frame_query, attn_mask, layer_idx):
        T, fQ, LB, C = frame_query.shape
        # LBN, WTfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W

        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
        frame_query = self.enc_ffn[layer_idx](frame_query)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        return frame_query
    
    def _shift_window_attn(self, frame_query, attn_mask, layer_idx):
        T, fQ, LB, C = frame_query.shape
        # LBNH, WfQ, WfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W
        half_W = int(ceil(W / 2))

        frame_query = torch.roll(frame_query, half_W, 0)
        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_mask=attn_mask)
        frame_query = self.enc_ffn[layer_idx](frame_query)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        frame_query = torch.roll(frame_query, -half_W, 0)

        return frame_query

if __name__ == '__main__':
    vita = VITA(
        input_dim = 256,
        window_size = 4,
        num_frame_queries = 50,
        num_frames = 8,
        num_queries = 5,
        nheads = 8,
        dim_feedforward = 2048,
        enc_layers = 0,
        dec_layers = 3,
        pre_norm = False, 
    )
    frame_query = torch.randn(size = (3, 8, 50, 256))
    lanaguae_query = torch.randn(size= (3, 1, 256))
    output = vita(frame_query, lanaguae_query)
    print(output.size())
