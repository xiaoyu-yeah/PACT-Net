import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add


def maybe_num_nodes(index, num_nodes=None):
    """Return the requested node count or infer it from an edge index."""
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(
    edge_index, edge_weight=None, fill_value=1, num_nodes=None
):
    """Add the identity contribution to every node's diagonal weight."""
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col
    inv_mask = torch.logical_not(mask)
    loop_weight = torch.full(
        (num_nodes,),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device,
    )

    if edge_weight is not None:
        if edge_weight.numel() != edge_index.size(1):
            raise ValueError("edge_weight and edge_index must contain the same number of edges")
        existing_loop_weight = edge_weight[inv_mask]
        if existing_loop_weight.numel() > 0:
            # The paper uses \tilde{W}=W+I, so an existing diagonal weight
            # receives the identity contribution instead of being replaced.
            loop_weight[row[inv_mask]] = existing_loop_weight + fill_value
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(
        num_nodes, dtype=row.dtype, device=row.device
    ).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
    return edge_index, edge_weight


class NewSGConv(SGConv):

    def __init__(self, num_features, num_classes, K=3, cached=False, bias=True):
        super().__init__(
            num_features,
            num_classes,
            K=K,
            cached=cached,
            bias=bias,
        )

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1), dtype=dtype, device=edge_index.device
            )

        fill_value = 2 if improved else 1
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )
        row, col = edge_index
        degree = scatter_add(
            torch.abs(edge_weight), row, dim=0, dim_size=num_nodes
        )
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0
        normalized_weight = (
            degree_inv_sqrt[row] * edge_weight * degree_inv_sqrt[col]
        )
        return edge_index, normalized_weight

    def forward(self, x, edge_index, edge_weight=None):
        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype
            )
            for _ in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=norm)
            self.cached_result = x
        return self.lin(self.cached_result)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class GraphLearner(nn.Module):

    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        # The paper's graph-learner diagram and Eq. (3) use three linear
        # projections with ReLU activations between them.
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_features):
        # node_features: (batch, num_nodes, feature_dim)
        _, num_nodes, _ = node_features.shape
        source = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        target = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        pair_features = torch.cat([source, target], dim=-1)
        return self.mlp(pair_features).squeeze(-1)


class EEGNet_Modified(nn.Module):

    def __init__(
        self,
        num_channels,
        time_points,
        F1=16,
        D=2,
        F2=32,
        dropout=0.25,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(
            F1, D * F1, (num_channels, 1), groups=F1
        )
        self.batchnorm2 = nn.BatchNorm2d(D * F1)
        self.pooling2 = nn.AvgPool2d((1, 4), stride=(1, 4))

        self.conv3 = nn.Conv2d(
            D * F1, D * F1, (1, 32), padding=(0, 16), groups=D * F1
        )
        self.conv4 = nn.Conv2d(D * F1, D * F1 * 2, (1, 1))
        self.batchnorm3 = nn.BatchNorm2d(D * F1 * 2)
        self.pooling3 = nn.AvgPool2d((1, 2), stride=(1, 2))

        self.conv5 = nn.Conv2d(
            D * F1 * 2, F2, (1, 16), padding=(0, 8)
        )
        self.batchnorm4 = nn.BatchNorm2d(F2)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

        self.time_reduction_factor = 8
        self.output_time_points = time_points

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.batchnorm1(x))
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(self.batchnorm2(x))
        x = self.pooling2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.activation(self.batchnorm3(x))
        x = self.pooling3(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.activation(self.batchnorm4(x))
        return x


class TransformerEncoder_Modified(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=4,
        num_heads=4,
        dropout=0.25,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _sinusoidal_encoding(seq_len, hidden_dim, device, dtype):
        position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(
            seq_len, hidden_dim, device=device, dtype=dtype
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(self, x):
        batch_size, num_nodes, seq_len, feat_dim = x.shape
        x = x.reshape(batch_size, num_nodes * seq_len, feat_dim)
        x = self.input_proj(x)
        x = x + self._sinusoidal_encoding(
            x.size(1), x.size(2), x.device, x.dtype
        )
        x = self.layer_norm(x)
        x = self.transformer(x)
        x = x.reshape(batch_size, num_nodes, seq_len, -1)
        return torch.mean(x, dim=2)


class DomainDiscriminator(nn.Module):

    def __init__(self, input_dim=32, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() == 2:
            if x.size(1) != self.input_dim:
                raise ValueError(
                    f"Expected domain features with dimension {self.input_dim}, "
                    f"got {x.size(1)}"
                )
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.size(1) != 1 and x.size(1) == self.input_dim:
                x = x.transpose(1, 2)
        else:
            raise ValueError("Domain features must have shape (B, D) or (B, C, L)")

        x = self.feature_extractor(x).squeeze(-1)
        return self.classifier(x)


class CrossModalAttention(nn.Module):

    def __init__(self, graph_dim, temporal_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.key_proj = nn.Linear(temporal_dim, hidden_dim)
        self.value_proj = nn.Linear(temporal_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph_feats, temporal_feats):
        query = self.graph_proj(graph_feats).unsqueeze(1)
        key = self.key_proj(temporal_feats)
        value = self.value_proj(temporal_feats)
        attention_output, _ = self.attention(query, key, value)
        attention_output = attention_output.squeeze(1)

        return F.relu(self.out_proj(attention_output) + query.squeeze(1))


class FeatureFusion_Improved(nn.Module):

    def __init__(self, graph_dim, temporal_dim, hidden_dim, output_dim):
        super().__init__()
        self.cross_attn = CrossModalAttention(
            graph_dim, temporal_dim, hidden_dim, num_heads=4
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, graph_feats, temporal_feats):
        fused = self.cross_attn(graph_feats, temporal_feats)
        vis_features = self.relu(self.fc1(fused))
        output = self.fc2(vis_features)
        return output, vis_features


class TemporalRGNN_Improved(nn.Module):

    def __init__(
        self,
        num_nodes,
        num_features,
        num_hiddens,
        num_classes,
        K=3,
        time_points=256,
        dropout=0.25,
        lambda_dyn=0.5,
        use_graph=True,
        use_temporal=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.time_points = time_points
        self.dropout = dropout
        self.lambda_dyn = lambda_dyn
        self.use_graph = use_graph
        self.use_temporal = use_temporal

        xs, ys = torch.tril_indices(num_nodes, num_nodes, offset=0)
        self.register_buffer("lt_xs", xs)
        self.register_buffer("lt_ys", ys)
        self.delta_param = nn.Parameter(torch.zeros(xs.size(0)))

        hidden_dim = num_hiddens[0]

        if use_graph:
            self.graph_conv = NewSGConv(
                num_features=num_features,
                num_classes=hidden_dim,
                K=K,
            )
            self.graph_fc = nn.Linear(hidden_dim, num_classes)

        if use_temporal:
            self.eegnet = EEGNet_Modified(
                num_channels=num_nodes,
                time_points=time_points,
                F2=hidden_dim // 2,
                dropout=dropout,
            )
            self.transformer_encoder = TransformerEncoder_Modified(
                input_dim=hidden_dim // 2,
                hidden_dim=hidden_dim,
                num_layers=4,
                num_heads=4,
                dropout=dropout,
            )
            self.temporal_fc = nn.Linear(hidden_dim, num_classes)

        if use_graph and use_temporal:
            self.graph_learner = GraphLearner(
                node_dim=num_features, hidden_dim=hidden_dim // 2
            )
            self.fusion_module = FeatureFusion_Improved(
                graph_dim=hidden_dim,
                temporal_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=num_classes,
            )
        elif use_graph:
            self.proj_graph_to_vis = nn.Linear(hidden_dim, hidden_dim // 2)
        elif use_temporal:
            self.proj_temp_to_vis = nn.Linear(hidden_dim, hidden_dim // 2)
        else:
            self.fallback_fc1 = nn.Linear(num_features, hidden_dim // 2)
            self.fallback_fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def _build_base_weight(self, edge_index, edge_attr, batch, num_nodes):
        if edge_attr.dim() == 2 and edge_attr.size(-1) == 1:
            edge_weight = edge_attr.squeeze(-1)
        else:
            edge_weight = edge_attr

        batch_size = int(batch.max().item()) + 1
        base_weight = torch.zeros(
            batch_size,
            num_nodes,
            num_nodes,
            device=edge_index.device,
            dtype=edge_weight.dtype,
        )
        source, target = edge_index
        batch_index = source // num_nodes
        source_node = source % num_nodes
        target_node = target % num_nodes
        base_weight[batch_index, source_node, target_node] = edge_weight
        base_weight[batch_index, target_node, source_node] = edge_weight
        return base_weight

    def _adaptive_adjacency(self, x, edge_index, edge_attr, batch, num_nodes):
        base_weight = self._build_base_weight(
            edge_index, edge_attr, batch, num_nodes
        )
        batch_size = base_weight.size(0)

        delta = base_weight.new_zeros((num_nodes, num_nodes))
        delta[self.lt_xs, self.lt_ys] = self.delta_param
        delta = delta + delta.t() - torch.diag(delta.diagonal())
        delta = delta.unsqueeze(0).expand(batch_size, -1, -1)

        node_features = x.view(batch_size, num_nodes, -1)
        dynamic_weight = self.graph_learner(node_features)
        dynamic_weight = 0.5 * (
            dynamic_weight + dynamic_weight.transpose(1, 2)
        )
        return base_weight + delta + self.lambda_dyn * dynamic_weight

    def forward(self, x, raw_eeg, batch, edge_index, edge_attr):
        batch_size = int(batch.max().item()) + 1
        num_nodes = x.size(0) // batch_size
        raw_eeg = raw_eeg.view(batch_size, num_nodes, self.time_points)

        if not self.use_graph and not self.use_temporal:
            pooled_x = x.view(batch_size, num_nodes, -1).mean(dim=1)
            vis_features = F.relu(self.fallback_fc1(pooled_x))
            fusion_output = self.fallback_fc2(vis_features)
            return {
                "graph": torch.zeros_like(fusion_output),
                "temporal": torch.zeros_like(fusion_output),
                "fusion": fusion_output,
                "vis": vis_features,
            }

        graph_output = None
        temporal_output = None
        graph_pooled = None
        temporal_pooled = None
        temporal_features = None

        if self.use_graph:
            adjacency = self._adaptive_adjacency(
                x, edge_index, edge_attr, batch, num_nodes
            )
            undirected_edge_index = torch.cat(
                [edge_index, edge_index.flip(0)], dim=1
            )
            source, target = undirected_edge_index
            edge_batch = source // num_nodes
            source_node = source % num_nodes
            target_node = target % num_nodes
            edge_weights = adjacency[edge_batch, source_node, target_node]

            graph_features = F.relu(
                self.graph_conv(x, undirected_edge_index, edge_weights)
            )
            graph_pooled = global_add_pool(graph_features, batch)
            graph_pooled = F.dropout(
                graph_pooled, p=self.dropout, training=self.training
            )
            graph_output = self.graph_fc(graph_pooled)

        if self.use_temporal:
            eeg_features = self.eegnet(raw_eeg.unsqueeze(1))
            eeg_features = eeg_features.permute(0, 2, 3, 1)
            temporal_features = self.transformer_encoder(eeg_features)
            temporal_pooled = torch.mean(temporal_features, dim=1)
            temporal_output = self.temporal_fc(temporal_pooled)

        if self.use_graph and self.use_temporal:
            fusion_output, vis_features = self.fusion_module(
                graph_pooled, temporal_features
            )
        elif self.use_graph:
            fusion_output = graph_output
            vis_features = self.proj_graph_to_vis(graph_pooled)
        else:
            fusion_output = temporal_output
            vis_features = self.proj_temp_to_vis(temporal_pooled)

        if graph_output is None:
            graph_output = torch.zeros_like(fusion_output)
        if temporal_output is None:
            temporal_output = torch.zeros_like(fusion_output)

        return {
            "graph": graph_output,
            "temporal": temporal_output,
            "fusion": fusion_output,
            "vis": vis_features,
        }


class MultiTaskLoss(nn.Module):

    def __init__(self, class_weights=None, use_graph=True, use_temporal=True):
        super().__init__()
        self.main_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, outputs, labels):
        fusion_loss = self.main_loss(outputs["fusion"], labels)
        zero = fusion_loss.new_zeros(())
        return {
            "total": fusion_loss,
            "graph": zero,
            "temporal": zero,
            "fusion": fusion_loss,
        }

