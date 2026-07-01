import pandas as pd
import numpy as np
import gzip
import networkx as nx
import random

random.seed(10)

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear, to_hetero
from torch_geometric.utils import negative_sampling, structured_negative_sampling

from sklearn.metrics import precision_score, recall_score, f1_score


class GCN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_dim)
        self.conv2 = GCNConv(-1, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_dim)
        self.conv2 = GATConv((-1, -1), output_dim, add_self_loops=False)
        self.lin2 = Linear(-1, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


class Two_Hop_GAT(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_dim)
        self.conv2 = GATConv((-1, -1), output_dim, add_self_loops=False)
        self.lin2 = Linear(-1, output_dim)

    def forward(self, x, edge_index, edge_index_2_hop):
        x = self.conv1(x, edge_index) + self.conv1(x, edge_index_2_hop)
        x = self.lin1(x)
        x = x.relu()

        x = self.conv2(x, edge_index) + self.conv2(x, edge_index_2_hop)
        x = self.lin2(x)
        return x


class GNN:
    def __init__(self):
        self.model = None
        self.gnn_variant = None
        self.epochs = 800
        self.node_embed_size = 200
        self.hidden_dim = 200
        self.output_dim = 200
        self.seed = 10
        torch.manual_seed(self.seed)

    def _init_model(self, GNN_variant, device):
        self.device = device
        self.gnn_variant = GNN_variant

        if GNN_variant == "GCN":
            self.model = GCN(self.hidden_dim, self.output_dim).to(self.device)
        elif GNN_variant == "GraphSAGE":
            self.model = GraphSAGE(self.hidden_dim, self.output_dim).to(self.device)
        elif GNN_variant == "GAT":
            self.model = GAT(self.hidden_dim, self.output_dim).to(self.device)
        elif GNN_variant in ["2-Hop GAT", "GAT Reasoner"]:
            self.model = Two_Hop_GAT(self.hidden_dim, self.output_dim).to(self.device)
        else:
            raise ValueError(f"Unsupported GNN variant: {GNN_variant}")

    def save_checkpoint(self, path):
        checkpoint = {
            "gnn_variant": self.gnn_variant,
            "epochs": self.epochs,
            "node_embed_size": self.node_embed_size,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "seed": self.seed,
            "node_embeds": self.node_embeds.detach().cpu(),
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, device, fallback_variant=None):
        ensure_legacy_pyg_checkpoint_compatibility()
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "This checkpoint was saved as a full Python object with an older "
                "PyTorch Geometric version and cannot be loaded in the current "
                "environment. Retrain the model in this environment to generate "
                "a new checkpoint with GNN.save_checkpoint(...)."
            ) from exc

        if isinstance(checkpoint, cls):
            checkpoint.device = device
            if checkpoint.model is not None:
                checkpoint.model = checkpoint.model.to(device)
            if hasattr(checkpoint, "node_embeds"):
                checkpoint.node_embeds = checkpoint.node_embeds.to(device)
            return checkpoint

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint format in {path}")

        model = cls()
        model.epochs = checkpoint.get("epochs", model.epochs)
        model.node_embed_size = checkpoint.get("node_embed_size", model.node_embed_size)
        model.hidden_dim = checkpoint.get("hidden_dim", model.hidden_dim)
        model.output_dim = checkpoint.get("output_dim", model.output_dim)
        model.seed = checkpoint.get("seed", model.seed)

        variant = checkpoint.get("gnn_variant") or fallback_variant
        if variant is None:
            raise ValueError(
                "Checkpoint does not contain gnn_variant; provide fallback_variant."
            )

        model._init_model(variant, device)
        model.node_embeds = checkpoint["node_embeds"].to(device)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _train(
        self,
        device,
        GNN_variant,
        g_train,
        g_subclass_filter=None,
        g_assertion_filter=None,
    ):
        self._init_model(GNN_variant, device)

        adj = nx.to_scipy_sparse_array(g_train)
        pos_edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
        neg_edge_index = negative_sampling(pos_edge_index)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        num_nodes = g_train.number_of_nodes()

        self.node_embeds = torch.rand(num_nodes, self.node_embed_size).to(self.device)

        if GNN_variant == "2-Hop GAT":
            adj_2hop = adj.dot(adj)
            edge_index_2hop = torch_geometric.utils.from_scipy_sparse_matrix(adj_2hop)[
                0
            ].to(self.device)
        elif GNN_variant == "GAT Reasoner":
            adj_2hop = adj.dot(adj)
            adj_assertion = nx.to_scipy_sparse_array(g_assertion_filter)
            adj_subclass = nx.to_scipy_sparse_array(g_subclass_filter)
            adj_subclass_2hop = adj_subclass.dot(adj_subclass)

            edge_index_2hop = torch_geometric.utils.from_scipy_sparse_matrix(adj_2hop)[
                0
            ].to(self.device)
            edge_index_assertion = torch_geometric.utils.from_scipy_sparse_matrix(
                adj_assertion
            )[0].to(self.device)
            edge_index_subclass = torch_geometric.utils.from_scipy_sparse_matrix(
                adj_subclass
            )[0].to(self.device)
            edge_index_subclass_2hop = torch_geometric.utils.from_scipy_sparse_matrix(
                adj_subclass_2hop
            )[0].to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.005, weight_decay=5e-4
        )
        targets = torch.cat(
            [torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])]
        )
        edge_index, targets = shuffle_predictions_targets(
            edge_index, targets, self.device
        )

        for i in range(self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            if GNN_variant == "2-Hop GAT":
                embeds = self.model(self.node_embeds, edge_index, edge_index_2hop).to(
                    self.device
                )
            elif GNN_variant == "GAT Reasoner":
                embeds_assertion = self.model(
                    self.node_embeds, edge_index_assertion, edge_index_subclass_2hop
                ).to(self.device)
                embeds_subclass = self.model(
                    self.node_embeds, edge_index_subclass, edge_index_subclass_2hop
                ).to(self.device)
                embeds_1hop = self.model(
                    self.node_embeds, edge_index, edge_index_2hop
                ).to(self.device)
                embeds = torch.stack(
                    [embeds_assertion, embeds_subclass, embeds_1hop]
                ).mean(dim=0)
            else:
                embeds = self.model(self.node_embeds, edge_index).to(self.device)

            u = torch.index_select(embeds, 0, edge_index[0, :])
            v = torch.index_select(embeds, 0, edge_index[1, :])
            pred = torch.sum(u * v, dim=-1)
            pred = (pred - pred.min()) / (pred.max() - pred.min())

            loss = mse_loss(pred, targets)
            loss.backward()
            optimizer.step()

            if i % 400 == 0:
                print(f"Epoch: {i}, Loss: {loss:.4f}")

    def _eval(
        self,
        max_num,
        GNN_variant,
        g_test,
        g_message,
        g_subclass_filter=None,
        g_assertion_filter=None,
        print_results=False,
    ):
        with torch.no_grad():
            self.model.eval()

            test_adj = nx.to_scipy_sparse_array(g_test)
            pos_edge_index = torch_geometric.utils.from_scipy_sparse_matrix(test_adj)[0]
            message_adj = nx.to_scipy_sparse_array(g_message)
            edge_index = torch_geometric.utils.from_scipy_sparse_matrix(message_adj)[
                0
            ].to(self.device)

            if GNN_variant == "2-Hop GAT":
                adj_2hop = message_adj.dot(message_adj)
                edge_index_2hop = torch_geometric.utils.from_scipy_sparse_matrix(
                    adj_2hop
                )[0].to(self.device)

            if GNN_variant == "GAT Reasoner":
                adj_2hop = message_adj.dot(message_adj)
                adj_assertion = nx.to_scipy_sparse_array(g_assertion_filter)
                adj_subclass = nx.to_scipy_sparse_array(g_subclass_filter)
                adj_subclass_2hop = adj_subclass.dot(adj_subclass)

                edge_index_2hop = torch_geometric.utils.from_scipy_sparse_matrix(
                    adj_2hop
                )[0].to(self.device)
                edge_index_assertion = torch_geometric.utils.from_scipy_sparse_matrix(
                    adj_assertion
                )[0].to(self.device)
                edge_index_subclass = torch_geometric.utils.from_scipy_sparse_matrix(
                    adj_subclass
                )[0].to(self.device)
                edge_index_subclass_2hop = (
                    torch_geometric.utils.from_scipy_sparse_matrix(adj_subclass_2hop)[
                        0
                    ].to(self.device)
                )

            if GNN_variant == "2-Hop GAT":
                output = self.model(self.node_embeds, edge_index, edge_index_2hop).to(
                    self.device
                )
            elif GNN_variant == "GAT Reasoner":
                embeds_assertion = self.model(
                    self.node_embeds, edge_index_assertion, edge_index_subclass_2hop
                ).to(self.device)
                embeds_subclass = self.model(
                    self.node_embeds, edge_index_subclass, edge_index_subclass_2hop
                ).to(self.device)
                embeds_1hop = self.model(
                    self.node_embeds, edge_index, edge_index_2hop
                ).to(self.device)
                output = torch.stack(
                    [embeds_assertion, embeds_subclass, embeds_1hop]
                ).mean(dim=0)
            else:
                output = self.model(self.node_embeds, edge_index).to(self.device)

            ###Model as Binary Classification Problem###
            # u = torch.index_select(output, 0, edge_index[0, :])
            # v = torch.index_select(output, 0, edge_index[1, :])
            # pred = torch.sum(u * v, dim=-1)
            # pred = (pred - pred.min()) / (pred.max() - pred.min())

            # pred = pred.detach().numpy()
            # pred = np.where(pred >= 0.5, 1, 0)
            # targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])]).detach().numpy()

            # precision = precision_score(targets, pred)
            # recall = recall_score(targets, pred)
            # f1 = f1_score(targets, pred)

            # print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}')
            # print()
            ######

            mrr, hits1, hits5, hits10 = eval_ranking_metrics(
                tail_pred=1,
                g_test=g_test,
                pos_edge_index=pos_edge_index.to(self.device),
                output=output,
                max_num=max_num,
                device=self.device,
            )
            metrics = {
                "MRR": mrr,
                "Hits@1": hits1,
                "Hits@5": hits5,
                "Hits@10": hits10,
            }
            if print_results:
                print(format_metrics(metrics))
                print("--------")
            return metrics


###HELPER FUNCIONS###


def mse_loss(pred, target):
    return (pred - target.to(pred.dtype)).pow(2).mean()


def shuffle_predictions_targets(edge_index, targets, device):
    edge_index = edge_index.to(device)
    targets = targets.to(device)
    perm = torch.randperm(edge_index.size(1), device=device)
    return edge_index[:, perm], targets[perm]


def eval_ranking_metrics(tail_pred, g_test, pos_edge_index, output, max_num, device):
    reciprocal_rank_sum = 0.0
    top1 = 0
    top5 = 0
    top10 = 0
    n = pos_edge_index.size(1)
    candidate_nodes = relation_candidate_nodes(pos_edge_index, tail_pred)

    for idx in range(n):
        if tail_pred == 1:
            x = torch.index_select(output, 0, pos_edge_index[0, idx])
        else:
            x = torch.index_select(output, 0, pos_edge_index[1, idx])

        candidates, candidates_embeds = sample_negative_edges_idx(
            idx=idx,
            tail_pred=tail_pred,
            g_test=g_test,
            pos_edge_index=pos_edge_index,
            output=output,
            max_num=max_num,
            device=device,
            candidate_nodes=candidate_nodes,
        )

        scores = torch.sum(candidates_embeds * x, dim=-1)
        score_dict = {cand: score for cand, score in zip(candidates, scores)}

        sorted_keys = [
            cand
            for cand, _ in sorted(
                score_dict.items(), key=lambda item: item[1], reverse=True
            )
        ]

        ranks_dict = {sorted_keys[i]: i for i in range(0, len(sorted_keys))}
        rank = ranks_dict[pos_edge_index[1, idx].item()] + 1

        reciprocal_rank_sum += 1.0 / rank

        if rank <= 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
        if rank <= 10:
            top10 += 1
    return reciprocal_rank_sum / n, top1 / n, top5 / n, top10 / n


def sample_negative_edges_idx(
    idx,
    tail_pred,
    g_test,
    pos_edge_index,
    output,
    max_num,
    device,
    candidate_nodes,
):
    candidates = []
    nodes = list(candidate_nodes)
    random.shuffle(nodes)

    for node in nodes:
        if len(candidates) >= max_num:
            break
        if tail_pred == 1:
            t = node
            h = pos_edge_index[0, idx].item()
            if (h, t) not in g_test.edges():
                candidates.append(t)
        else:
            t = pos_edge_index[1, idx].item()
            h = node
            if (h, t) not in g_test.edges():
                candidates.append(h)

    candidates_embeds = torch.index_select(
        output, 0, torch.tensor(candidates, dtype=torch.long, device=device)
    )

    if tail_pred == 1:
        true_tail = pos_edge_index[1, idx]
        candidates.append(true_tail.item())
        candidates_embeds = torch.concat(
            [candidates_embeds, torch.index_select(output, 0, true_tail)]
        )
    else:
        true_head = pos_edge_index[0, idx]
        candidates.append(true_head.item())
        candidates_embeds = torch.concat(
            [candidates_embeds, torch.index_select(output, 0, true_head)]
        )
    return candidates, candidates_embeds.to(device)


def relation_candidate_nodes(pos_edge_index, tail_pred):
    candidate_row = 1 if tail_pred == 1 else 0
    return set(pos_edge_index[candidate_row, :].detach().cpu().tolist())


def ensure_legacy_pyg_checkpoint_compatibility():
    try:
        from torch_geometric.nn.dense.linear import Linear as PyGLinear

        if not hasattr(PyGLinear, "_lazy_load_hook"):
            PyGLinear._lazy_load_hook = None
    except ImportError:
        pass


def format_metrics(metrics):
    return (
        f"MRR: {metrics['MRR']:.3f}, "
        f"hits@1: {metrics['Hits@1']:.3f}, "
        f"hits@5: {metrics['Hits@5']:.3f}, "
        f"hits@10: {metrics['Hits@10']:.3f}"
    )
