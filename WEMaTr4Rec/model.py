import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
import ptwt  # 导入 ptwt
import numpy as np
from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class WEMaTr4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(WEMaTr4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.attn_heads = config['attn_heads'] if 'attn_heads' in config else 4
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.mamba_layers = nn.ModuleList([
            BiMaTrLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
                max_seq_length=self.max_seq_length,
                attn_heads=self.attn_heads
            ) for _ in range(self.num_layers)
        ])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class BiMaTrLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, max_seq_length, attn_heads=4):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.attn_heads = attn_heads
        self.filter_layer = DualPathFilterLayer(
            max_seq_length=max_seq_length,
            hidden_size=d_model,
            dropout_prob=dropout
        )

        self.norms_forward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms_backward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.attn_norms_forward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.attn_norms_backward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.mamba_forwards = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)
        ])
        self.mamba_backwards = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)
        ])

        self.attention_forwards = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.attention_backwards = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.glu = GLU(d_model=d_model, dropout=dropout)

        self.multi_query_transformer_block = MultiQueryTransformerBlock(
            d_model=d_model,
            nhead=attn_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )

    def forward(self, input_tensor):
        x = input_tensor
        x = self.filter_layer(x)

        for i in range(self.num_layers):
            forward_states = self.mamba_forwards[i](x)

            attn_output, _ = self.attention_forwards[i](
                forward_states, forward_states, forward_states
            )

            forward_states = forward_states + self.dropout(attn_output)
            forward_states = self.attn_norms_forward[i](forward_states)

            forward_states = self.norms_forward[i](forward_states + x)

            reversed_input = torch.flip(x, [1])

            backward_states = self.mamba_backwards[i](reversed_input)

            attn_output, _ = self.attention_backwards[i](
                backward_states, backward_states, backward_states
            )

            backward_states = backward_states + self.dropout(attn_output)
            backward_states = self.attn_norms_backward[i](backward_states)

            backward_states = torch.flip(backward_states, [1])

            backward_states = self.norms_backward[i](backward_states + x)

            x = forward_states + backward_states

        x = self.glu(x)
        return x


class DualPathFilterLayer(nn.Module):

    def __init__(self, max_seq_length, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length

        self.fft_path = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        self.wavelet_path = WaveletPath(hidden_size)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout_prob,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout_prob)
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.pool = nn.AdaptiveAvgPool1d(max_seq_length)

    def forward(self, x):
        x_fft = self._fft_filter(x)

        x_wavelet = self.wavelet_path(x)

        if x_fft.shape[1] != x_wavelet.shape[1]:
            min_len = min(x_fft.shape[1], x_wavelet.shape[1])
            x_fft = x_fft[:, :min_len, :]
            x_wavelet = x_wavelet[:, :min_len, :]

        fused, _ = self.cross_attn(
            query=x_fft,
            key=x_wavelet,
            value=x_wavelet
        )

        fused = fused + x

        gate_input = torch.cat([fused, x], dim=-1)
        gated_output = self.gate(gate_input)

        return self.layer_norm(gated_output)

    def _fft_filter(self, x):
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        real = x_fft.real
        imag = x_fft.imag

        combined = torch.cat([real, imag], dim=-1)
        filtered = self.fft_path(combined)

        real_f, imag_f = torch.chunk(filtered, 2, dim=-1)

        reconstructed = torch.fft.irfft(
            torch.complex(real_f, imag_f),
            n=self.max_seq_length,
            dim=1,
            norm='ortho'
        )

        if reconstructed.shape[1] != self.max_seq_length:
            reconstructed = reconstructed.permute(0, 2, 1)
            reconstructed = self.pool(reconstructed)
            reconstructed = reconstructed.permute(0, 2, 1)

        return reconstructed


# =============================================================================
#  Begin Modified Section: WaveletPath
# =============================================================================

class WaveletPath(nn.Module):
    def __init__(self, d_model, wavelet='db4', max_level=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )
        self.wavelet = wavelet
        self.max_level = max_level
        self.d_model = d_model

    def forward(self, x):
        # x 现在的形状是 (B, S, D)
        B, S, D = x.shape

        # 检查是否可以执行变换
        # pywt 仍然可以用于计算 max_level (这是一个 CPU 标量计算，没有性能影响)
        max_possible_level = pywt.dwt_max_level(S, self.wavelet)
        level = min(max_possible_level, self.max_level)
        if level < 1:
            return torch.zeros_like(x)

        # 1. 调整输入形状以适应 ptwt
        # ptwt.wavedec 需要 (B, D, S)
        x_permuted = x.permute(0, 2, 1)  # 形状变为 (B, D, S)

        # 2. 【已删除】移除手动填充块
        #    ptwt 将自动处理填充

        # 3. 在 GPU 上执行小波变换
        #    ptwt 现在将自动处理基于 'zero' 模式的填充
        coeffs = ptwt.wavedec(
            x_permuted,
            self.wavelet,
            level=level,
            axis=2,
            mode='zero'  # 指定填充模式以匹配 'constant' 行为
        )

        # 4. 获取近似系数 (approx)
        approx_tensor = coeffs[0]  # (B, D, S_approx)

        # 5. 在 GPU 上执行卷积
        conv_out = self.conv(approx_tensor)  # (B, D, S_approx)

        # 6. 上采样/插值回原始序列长度 S
        if conv_out.size(2) != S:
            conv_out = F.interpolate(
                conv_out,
                size=S,
                mode='linear',
                align_corners=False
            )

        # 7. 恢复原始形状 (B, S, D)
        return conv_out.permute(0, 2, 1)


# =============================================================================
#  End Modified Section
# =============================================================================


class MultiQueryTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.glu = GLU(d_model, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x_transposed = x.permute(1, 0, 2)
        attn_output, _ = self.multihead_attn(x_transposed, x_transposed, x_transposed)
        attn_output = attn_output.permute(1, 0, 2)
        x = x + attn_output
        x = self.norm1(x)
        x = self.dropout1(x)

        glu_output = self.glu(x)
        x = x + glu_output
        x = self.norm2(x)
        x = self.dropout2(x)

        return x


class GLU(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        x_transformed = self.fc1(x)
        value, gate = x_transformed.chunk(2, dim=-1)
        gated_value = value * torch.sigmoid(gate)
        gated_value = self.fc2(gated_value)
        return self.LayerNorm(self.dropout(gated_value + x))