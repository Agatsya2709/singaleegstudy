import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, features], adj: [nodes, nodes]
        support = torch.matmul(x, self.weight)
        out = torch.einsum("ij,bjk->bik", adj, support)
        return out + self.bias


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat

        self.weight = nn.Parameter(torch.empty(heads, in_features, out_features))
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.last_attention = None

        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(-1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, features]
        wh = torch.einsum("bni,hio->bhno", x, self.weight)

        src_scores = torch.einsum("bhno,ho->bhn", wh, self.attn_src)
        dst_scores = torch.einsum("bhno,ho->bhn", wh, self.attn_dst)
        e = self.leaky_relu(src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2))

        if adj.dim() == 2:
            mask = (adj > 0).unsqueeze(0).unsqueeze(0)
        else:
            mask = (adj > 0).unsqueeze(1)

        e = e.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(e, dim=-1)
        attention = self.attn_dropout(attention)
        self.last_attention = attention.mean(dim=1)

        out = torch.einsum("bhij,bhjo->bhio", attention, wh)
        if self.concat:
            out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.heads * self.out_features)
        else:
            out = out.mean(dim=1)
        return out


class HybridCNNLSTMGCN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        adjacency: torch.Tensor,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.register_buffer("adjacency", adjacency)

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.node_time_proj = nn.Linear(n_times, 96)
        self.gcn1 = GraphConvolution(96, 96)
        self.gcn2 = GraphConvolution(96, 96)

        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(48 + 192 + 96, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(160, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        cnn = self.temporal(x.unsqueeze(1))
        cnn = self.spatial(cnn).squeeze(2)  # [batch, 48, time]
        cnn_feat = F.adaptive_avg_pool1d(cnn, output_size=1).squeeze(-1)

        lstm_in = cnn.transpose(1, 2)  # [batch, time, 48]
        lstm_out, _ = self.lstm(lstm_in)
        lstm_feat = lstm_out[:, -1, :]

        node_feat = self.node_time_proj(x)  # [batch, channels, 96]
        node_feat = F.relu(self.gcn1(node_feat, self.adjacency))
        node_feat = self.dropout(node_feat)
        node_feat = F.relu(self.gcn2(node_feat, self.adjacency))
        gnn_feat = torch.mean(node_feat, dim=1)

        fused = torch.cat([cnn_feat, lstm_feat, gnn_feat], dim=1)
        fused = self.dropout(fused)
        return self.fusion(fused)


class HybridCNNLSTMGAT(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        adjacency: torch.Tensor,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.last_attention_matrix = None

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.node_time_proj = nn.Linear(n_times, 96)
        self.gat1 = MultiHeadGraphAttention(96, 24, heads=4, dropout=dropout, concat=True)
        self.gat2 = MultiHeadGraphAttention(96, 24, heads=4, dropout=dropout, concat=True)
        self.node_norm1 = nn.LayerNorm(96)
        self.node_norm2 = nn.LayerNorm(96)

        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(48 + 192 + 96, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        cnn = self.temporal(x.unsqueeze(1))
        cnn = self.spatial(cnn).squeeze(2)  # [batch, 48, time]
        cnn_feat = F.adaptive_avg_pool1d(cnn, output_size=1).squeeze(-1)

        lstm_in = cnn.transpose(1, 2)  # [batch, time, 48]
        lstm_out, _ = self.lstm(lstm_in)
        lstm_feat = lstm_out[:, -1, :]

        node_feat = self.node_time_proj(x)
        node_feat = self.node_norm1(F.elu(self.gat1(node_feat, self.adjacency)))
        node_feat = self.dropout(node_feat)
        node_feat = self.node_norm2(F.elu(self.gat2(node_feat, self.adjacency)))
        gnn_feat = torch.mean(node_feat, dim=1)

        if self.gat1.last_attention is not None and self.gat2.last_attention is not None:
            self.last_attention_matrix = (self.gat1.last_attention + self.gat2.last_attention) / 2.0

        fused = torch.cat([cnn_feat, lstm_feat, gnn_feat], dim=1)
        fused = self.dropout(fused)
        return self.fusion(fused)

    def get_last_attention(self):
        if self.last_attention_matrix is None:
            return None
        return self.last_attention_matrix.detach().mean(dim=0).cpu()


class CompactCNNGAT(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        adjacency: torch.Tensor,
        dropout: float = 0.45,
        dropedge: float = 0.15,
    ):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.dropedge = dropedge
        self.last_attention_matrix = None

        # Channel-wise temporal encoder with low parameter count.
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, n_channels * 2, kernel_size=11, padding=5, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels * 2),
            nn.ELU(inplace=True),
            nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=7, padding=3, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels * 4),
            nn.ELU(inplace=True),
        )

        self.node_proj = nn.Linear(4, 32)
        self.gat1 = MultiHeadGraphAttention(32, 8, heads=4, dropout=dropout, concat=True)   # 32
        self.gat2 = MultiHeadGraphAttention(32, 8, heads=4, dropout=dropout, concat=True)   # 32
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def _dropedge_adj(self):
        if (not self.training) or self.dropedge <= 0:
            return self.adjacency

        keep = (torch.rand_like(self.adjacency) > self.dropedge).float()
        keep = torch.triu(keep, diagonal=1)
        keep = keep + keep.transpose(0, 1)
        keep = keep + torch.eye(self.adjacency.size(0), device=self.adjacency.device)
        adj = self.adjacency * keep
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        z = self.temporal(x)  # [B, C*4, T]
        b, _, t = z.shape
        z = z.view(b, -1, 4, t)  # [B, C, 4, T]
        z = torch.mean(z, dim=-1)  # [B, C, 4]
        z = self.node_proj(z)      # [B, C, 32]

        adj = self._dropedge_adj()
        z = self.norm1(F.elu(self.gat1(z, adj)))
        z = self.dropout(z)
        z = self.norm2(F.elu(self.gat2(z, adj)))

        if self.gat1.last_attention is not None and self.gat2.last_attention is not None:
            self.last_attention_matrix = (self.gat1.last_attention + self.gat2.last_attention) / 2.0

        pooled_mean = torch.mean(z, dim=1)
        pooled_max = torch.max(z, dim=1).values
        feat = torch.cat([pooled_mean, pooled_max], dim=1)
        feat = self.dropout(feat)
        return self.classifier(feat)

    def get_last_attention(self):
        if self.last_attention_matrix is None:
            return None
        return self.last_attention_matrix.detach().mean(dim=0).cpu()


class GraphEnhancedCNNGAT(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        adjacency: torch.Tensor,
        dropout: float = 0.35,
        dropedge: float = 0.10,
    ):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.dropedge = dropedge
        self.last_attention_matrix = None

        # Keep the strong CNN backbone behavior.
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
        )

        # Lightweight graph branch over per-channel temporal signatures.
        self.node_proj = nn.Linear(n_times, 32)
        self.gat1 = MultiHeadGraphAttention(32, 8, heads=4, dropout=dropout, concat=True)  # 32
        self.gat2 = MultiHeadGraphAttention(32, 8, heads=4, dropout=dropout, concat=True)  # 32
        self.gnorm1 = nn.LayerNorm(32)
        self.gnorm2 = nn.LayerNorm(32)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(48 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def _dropedge_adj(self):
        if (not self.training) or self.dropedge <= 0:
            return self.adjacency
        keep = (torch.rand_like(self.adjacency) > self.dropedge).float()
        keep = torch.triu(keep, diagonal=1)
        keep = keep + keep.transpose(0, 1)
        keep = keep + torch.eye(self.adjacency.size(0), device=self.adjacency.device)
        return self.adjacency * keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        temp = self.temporal(x.unsqueeze(1))      # [B, 32, C, T]

        # CNN pooled representation (baseline-like).
        cnn_feat = self.spatial(temp).squeeze(2)  # [B, 48, T]
        cnn_feat = F.adaptive_avg_pool1d(cnn_feat, output_size=1).squeeze(-1)  # [B, 48]

        # Graph representation.
        node_signal = torch.mean(temp, dim=1)  # [B, C, T]
        node_feat = self.node_proj(node_signal)  # [B, C, 32]
        adj = self._dropedge_adj()
        node_feat = self.gnorm1(F.elu(self.gat1(node_feat, adj)))
        node_feat = self.dropout(node_feat)
        node_feat = self.gnorm2(F.elu(self.gat2(node_feat, adj)))
        graph_feat = torch.mean(node_feat, dim=1)  # [B, 32]

        if self.gat1.last_attention is not None and self.gat2.last_attention is not None:
            self.last_attention_matrix = (self.gat1.last_attention + self.gat2.last_attention) / 2.0

        fused = torch.cat([cnn_feat, graph_feat], dim=1)
        fused = self.dropout(fused)
        return self.head(fused)

    def get_last_attention(self):
        if self.last_attention_matrix is None:
            return None
        return self.last_attention_matrix.detach().mean(dim=0).cpu()


# Backward-compatible alias used by training scripts
HybridCNNLSTMGNN = HybridCNNLSTMGAT
