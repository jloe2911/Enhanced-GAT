import copy
import re
import random
from pathlib import Path

import numpy as np
import rdflib
import torch
import torch.nn.functional as F
from rdflib import OWL, RDF, RDFS, Namespace
from torch.nn import Parameter
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, Linear
from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    negative_sampling,
    to_scipy_sparse_matrix,
)


HIDDEN_CHANNELS = 200
LEARNING_RATE = 0.01
DROPOUT = 0.2
REGULARIZATION = 1e-2


def checkpoint_safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_namespace(dataset_name):
    normalized_name = dataset_name.lower()
    if normalized_name.startswith("family") or "family" in normalized_name:
        return Namespace("http://www.example.com/genealogy.owl#")
    if (
        dataset_name.startswith("OWL2DL-")
        or dataset_name.startswith("OWL2Bench")
        or "OWL2DL-" in dataset_name
        or "OWL2Bench" in dataset_name
    ):
        return Namespace("https://kracr.iiitd.edu.in/OWL2Bench#")
    if normalized_name.startswith("pizza") or "pizza" in normalized_name:
        return Namespace("http://www.co-ode.org/ontologies/pizza/pizza.owl#")
    raise ValueError(f"Unknown NSORN namespace for dataset '{dataset_name}'")


def dataset_names(dataset):
    test_name = Path(dataset["test_file"]).stem
    dataset_name = dataset.get("model_file") or dataset.get("file")
    file_name = test_name.removesuffix("_test")
    if file_name.startswith("_test_"):
        file_name = file_name.removeprefix("_test_")
    if file_name.endswith("-test"):
        file_name = file_name.removesuffix("-test")
    return dataset_name, file_name


def validation_path(dataset):
    if dataset.get("val_file"):
        return Path(dataset["path"]) / dataset["val_file"]

    base_path = Path(dataset["path"])
    train_path = Path(dataset["train_file"])
    train_name = train_path.name
    if train_name.startswith("_train_"):
        return base_path / train_name.replace("_train_", "_valid_", 1)
    if train_name.endswith("_train.owl"):
        return base_path / train_name.replace("_train.owl", "_val.owl")
    if train_name.endswith("-train.nt"):
        return base_path / train_name.replace("-train.nt", "-valid.nt")
    return base_path / f"{train_path.stem}_val{train_path.suffix}"


def load_rdflib_graph(path):
    if is_functional_syntax_file(path):
        return load_functional_syntax_graph(path)

    graph = rdflib.Graph()
    suffix = Path(path).suffix.lower()
    parse_format = "nt" if suffix == ".nt" else None
    graph.parse(path, format=parse_format)
    return graph


def is_functional_syntax_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        prefix = handle.read(256).lstrip()
    return prefix.startswith(
        ("SubClassOf(", "ClassAssertion(", "ObjectPropertyAssertion(")
    )


def as_rdflib_term(token):
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        return rdflib.URIRef(token[1:-1])
    if token == "owl:Thing":
        return OWL.Thing
    if token == "owl:Nothing":
        return OWL.Nothing
    if ":" in token:
        return rdflib.URIRef(token)
    return rdflib.URIRef(token)


def load_functional_syntax_graph(path):
    graph = rdflib.Graph()
    pattern = re.compile(
        r"^(SubClassOf|ClassAssertion|ObjectPropertyAssertion)\((.*)\)$"
    )
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            axiom, body = match.groups()
            terms = re.findall(r"<[^>]+>|owl:Thing|owl:Nothing|[^\s()]+", body)
            if axiom == "SubClassOf" and len(terms) >= 2:
                graph.add(
                    (
                        as_rdflib_term(terms[0]),
                        RDFS.subClassOf,
                        as_rdflib_term(terms[1]),
                    )
                )
            elif axiom == "ClassAssertion" and len(terms) >= 2:
                graph.add(
                    (as_rdflib_term(terms[1]), RDF.type, as_rdflib_term(terms[0]))
                )
            elif axiom == "ObjectPropertyAssertion" and len(terms) >= 3:
                graph.add(
                    (
                        as_rdflib_term(terms[1]),
                        as_rdflib_term(terms[0]),
                        as_rdflib_term(terms[2]),
                    )
                )
    return graph


def rdf_edges_to_tensors(graph, node2id, rel2id):
    src = []
    dst = []
    rel = []

    for s, p, o in graph:
        if s not in node2id:
            node2id[s] = len(node2id)
        if o not in node2id:
            node2id[o] = len(node2id)
        if p not in rel2id:
            rel2id[p] = len(rel2id)
        src.append(node2id[s])
        dst.append(node2id[o])
        rel.append(rel2id[p])

    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(
        rel, dtype=torch.long
    )


def empty_edge_tensors():
    return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)


def load_fixed_split_data(dataset, vocabulary_datasets=None):
    dataset_name, _ = dataset_names(dataset)
    base_path = Path(dataset["path"])
    train_graph = load_rdflib_graph(base_path / dataset["train_file"])
    test_graph = load_rdflib_graph(base_path / dataset["test_file"])

    val_path = validation_path(dataset)
    val_graph = load_rdflib_graph(val_path) if val_path.exists() else rdflib.Graph()

    node2id = {}
    rel2id = {}
    train_edge_index, train_edge_type = rdf_edges_to_tensors(
        train_graph, node2id, rel2id
    )
    val_edge_index, val_edge_type = rdf_edges_to_tensors(val_graph, node2id, rel2id)
    test_edge_index, test_edge_type = rdf_edges_to_tensors(test_graph, node2id, rel2id)

    if vocabulary_datasets:
        for vocab_dataset in vocabulary_datasets:
            vocab_test_path = Path(vocab_dataset["path"]) / vocab_dataset["test_file"]
            if vocab_test_path == base_path / dataset["test_file"]:
                continue
            vocab_graph = load_rdflib_graph(vocab_test_path)
            rdf_edges_to_tensors(vocab_graph, node2id, rel2id)

    data = HeteroData()
    data.train_pos_edge_index = train_edge_index
    data.train_edge_type = train_edge_type
    data.val_pos_edge_index = val_edge_index
    data.val_edge_type = val_edge_type
    data.test_pos_edge_index = test_edge_index
    data.test_edge_type = test_edge_type
    data.edge_index = torch.cat(
        [train_edge_index, val_edge_index, test_edge_index], dim=1
    )
    data.edge_type = torch.cat([train_edge_type, val_edge_type, test_edge_type])
    data.has_validation_edges = val_edge_index.numel() > 0
    return data, node2id, rel2id


def load_shared_train_data(dataset, evaluation_datasets):
    dataset_name, _ = dataset_names(dataset)
    base_path = Path(dataset["path"])
    train_graph = load_rdflib_graph(base_path / dataset["train_file"])

    val_path = validation_path(dataset)
    val_graph = load_rdflib_graph(val_path) if val_path.exists() else rdflib.Graph()

    node2id = {}
    rel2id = {}
    train_edge_index, train_edge_type = rdf_edges_to_tensors(
        train_graph, node2id, rel2id
    )
    val_edge_index, val_edge_type = rdf_edges_to_tensors(val_graph, node2id, rel2id)

    all_test_edge_indices = []
    all_test_edge_types = []
    test_edges_by_file = {}
    for eval_dataset in evaluation_datasets:
        test_graph = load_rdflib_graph(
            Path(eval_dataset["path"]) / eval_dataset["test_file"]
        )
        test_edge_index, test_edge_type = rdf_edges_to_tensors(
            test_graph, node2id, rel2id
        )
        all_test_edge_indices.append(test_edge_index)
        all_test_edge_types.append(test_edge_type)
        test_edges_by_file[eval_dataset["file"]] = (test_edge_index, test_edge_type)

    if all_test_edge_indices:
        all_test_edge_index = torch.cat(all_test_edge_indices, dim=1)
        all_test_edge_type = torch.cat(all_test_edge_types)
    else:
        all_test_edge_index, all_test_edge_type = empty_edge_tensors()

    data = HeteroData()
    data.train_pos_edge_index = train_edge_index
    data.train_edge_type = train_edge_type
    data.val_pos_edge_index = val_edge_index
    data.val_edge_type = val_edge_type
    data.test_pos_edge_index, data.test_edge_type = empty_edge_tensors()
    data.edge_index = torch.cat(
        [train_edge_index, val_edge_index, all_test_edge_index], dim=1
    )
    data.edge_type = torch.cat([train_edge_type, val_edge_type, all_test_edge_type])
    data.has_validation_edges = val_edge_index.numel() > 0
    return data, node2id, rel2id, test_edges_by_file


def with_test_edges(shared_data, dataset, test_edges_by_file):
    test_edge_index, test_edge_type = test_edges_by_file[dataset["file"]]
    data = shared_data.clone()
    data.test_pos_edge_index = test_edge_index.to(shared_data.edge_index.device)
    data.test_edge_type = test_edge_type.to(shared_data.edge_type.device)
    return data


def two_hop_edges(edge_index, num_nodes, device):
    if edge_index.numel() == 0:
        return edge_index
    adjacency = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=num_nodes)
    return from_scipy_sparse_matrix(adjacency.dot(adjacency))[0].to(device)


def composed_edges(first_edge_index, second_edge_index, num_nodes, device):
    if first_edge_index.numel() == 0 or second_edge_index.numel() == 0:
        return first_edge_index[:, :0].to(device)
    first_adjacency = to_scipy_sparse_matrix(
        first_edge_index.cpu(), num_nodes=num_nodes
    )
    second_adjacency = to_scipy_sparse_matrix(
        second_edge_index.cpu(), num_nodes=num_nodes
    )
    return from_scipy_sparse_matrix(first_adjacency.dot(second_adjacency))[0].to(device)


def unique_edges(edge_index, num_nodes, device):
    if edge_index.numel() == 0:
        return edge_index.to(device)
    adjacency = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=num_nodes)
    return from_scipy_sparse_matrix(adjacency)[0].to(device)


def relation_edge_index(data, relation_id):
    if relation_id is None:
        return data.train_pos_edge_index[:, :0]
    mask = data.train_edge_type == relation_id
    return data.train_pos_edge_index[:, mask]


class GatEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels):
        super().__init__()
        self.output_dim = hidden_channels
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        self.conv1 = GATConv(
            (-1, -1), hidden_channels, add_self_loops=False, dropout=DROPOUT
        )
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv(
            (-1, -1), hidden_channels, add_self_loops=False, dropout=DROPOUT
        )
        self.lin2 = Linear(-1, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.lin1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin2.reset_parameters()

    def _conv(self, conv, x, edge_index, output_dim):
        if edge_index.numel() == 0:
            return torch.zeros(x.size(0), output_dim, dtype=x.dtype, device=x.device)
        return conv(x, edge_index)

    def forward(self, edge_index, edge_type=None):
        x = self._conv(
            self.conv1, self.node_emb, edge_index, HIDDEN_CHANNELS
        ) + self.lin1(self.node_emb)
        x = x.relu()
        x = F.dropout(x, p=DROPOUT, training=self.training)
        return self._conv(self.conv2, x, edge_index, HIDDEN_CHANNELS) + self.lin2(x)


class TwoHopGatEncoder(GatEncoder):
    def __init__(self, num_nodes, hidden_channels):
        super().__init__(num_nodes, hidden_channels)
        self.two_hop_relation_ids = None

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, edge_index, edge_type=None):
        two_hop_base_edges = edge_index
        if self.two_hop_relation_ids is not None and edge_type is not None:
            relation_ids = torch.tensor(
                sorted(self.two_hop_relation_ids),
                dtype=edge_type.dtype,
                device=edge_type.device,
            )
            two_hop_base_edges = edge_index[:, torch.isin(edge_type, relation_ids)]
        edge_index_2hop = two_hop_edges(
            two_hop_base_edges,
            self.node_emb.size(0),
            edge_index.device,
        )
        return self.forward_with_2hop(edge_index, edge_index_2hop)

    def set_two_hop_relations(self, relation_ids):
        self.two_hop_relation_ids = set(relation_ids)

    def forward_with_2hop(self, edge_index, edge_index_2hop):
        x = (
            self._conv(self.conv1, self.node_emb, edge_index, HIDDEN_CHANNELS)
            + self.lin1(self.node_emb)
            + self._conv(self.conv1, self.node_emb, edge_index_2hop, HIDDEN_CHANNELS)
        )
        x = x.relu()
        x = F.dropout(x, p=DROPOUT, training=self.training)
        return (
            self._conv(self.conv2, x, edge_index, HIDDEN_CHANNELS)
            + self.lin2(x)
            + self._conv(self.conv2, x, edge_index_2hop, HIDDEN_CHANNELS)
        )


class FilteredTwoHopGatEncoder(TwoHopGatEncoder):
    def __init__(self, num_nodes, hidden_channels):
        torch.nn.Module.__init__(self)
        self.output_dim = hidden_channels * 3
        self.standard_encoder = GatEncoder(num_nodes, hidden_channels)
        self.subclass_encoder = TwoHopGatEncoder(num_nodes, hidden_channels)
        self.assertion_encoder = TwoHopGatEncoder(num_nodes, hidden_channels)

    def forward(self, edge_index, edge_type=None):
        if edge_type is None:
            return self.standard_encoder(edge_index, edge_type)

        subclass_edges = relation_edge_index(self.data, self.subclass_id).to(
            edge_index.device
        )
        assertion_edges = relation_edge_index(self.data, self.rdf_type_id).to(
            edge_index.device
        )
        num_nodes = self.standard_encoder.node_emb.size(0)
        subclass_2hop = two_hop_edges(
            subclass_edges,
            num_nodes,
            edge_index.device,
        )
        assertion_subclass_2hop = composed_edges(
            assertion_edges,
            subclass_edges,
            num_nodes,
            edge_index.device,
        )
        standard_view = self.standard_encoder(edge_index, edge_type)
        subclass_view = self.subclass_encoder.forward_with_2hop(
            subclass_edges, subclass_2hop
        )
        assertion_view = self.assertion_encoder.forward_with_2hop(
            assertion_edges, assertion_subclass_2hop
        )
        return torch.cat([standard_view, subclass_view, assertion_view], dim=1)

    def set_filters(self, data, rdf_type_id, subclass_id):
        self.data = data
        self.rdf_type_id = rdf_type_id
        self.subclass_id = subclass_id


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


class FilteredDistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels, rdf_type_id, subclass_id):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.rdf_type_id = rdf_type_id
        self.subclass_id = subclass_id
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.view_logits = Parameter(torch.empty(num_relations, 3))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)
        torch.nn.init.zeros_(self.view_logits)
        with torch.no_grad():
            if self.rdf_type_id is not None:
                self.view_logits[self.rdf_type_id] = torch.tensor([-0.5, -1.0, 1.0])
            if self.subclass_id is not None:
                self.view_logits[self.subclass_id] = torch.tensor([-0.5, 1.0, -1.0])

    def _combine_views(self, z, node_index, edge_type):
        standard, subclass, assertion = z.split(self.hidden_channels, dim=1)
        weights = F.softmax(self.view_logits[edge_type], dim=1)
        return (
            weights[:, 0:1] * standard[node_index]
            + weights[:, 1:2] * subclass[node_index]
            + weights[:, 2:3] * assertion[node_index]
        )

    def forward(self, z, edge_index, edge_type):
        z_src = self._combine_views(z, edge_index[0], edge_type)
        z_dst = self._combine_views(z, edge_index[1], edge_type)
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


class NsornGatReasoner(torch.nn.Module):
    def __init__(
        self,
        variant,
        device,
        num_nodes,
        num_relations,
        rdf_type_id,
        subclass_id,
        link_relation_ids,
        rule_aux_weight=0.0,
    ):
        super().__init__()
        self.variant = variant
        self.device = device
        self.rdf_type_id = rdf_type_id
        self.subclass_id = subclass_id
        self.link_relation_ids = set(link_relation_ids)
        self.rule_aux_weight = rule_aux_weight
        self.class_nodes = None
        self.individual_nodes = None
        self.forbidden_tails_by_rel = None
        if variant == "GAT":
            self.encoder = GatEncoder(num_nodes, HIDDEN_CHANNELS)
        elif variant == "2-Hop GAT":
            self.encoder = TwoHopGatEncoder(num_nodes, HIDDEN_CHANNELS)
        elif variant == "Filtered 2-Hop GAT":
            self.encoder = FilteredTwoHopGatEncoder(num_nodes, HIDDEN_CHANNELS)
        else:
            raise ValueError(f"Unsupported NSORN GAT variant: {variant}")
        if isinstance(self.encoder, FilteredTwoHopGatEncoder):
            self.decoder = FilteredDistMultDecoder(
                num_relations,
                HIDDEN_CHANNELS,
                rdf_type_id,
                subclass_id,
            )
        else:
            self.decoder = DistMultDecoder(num_relations, self.encoder.output_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def save_checkpoint(
        self,
        path,
        best_val_loss=None,
        seed=None,
        epochs=None,
        node2id=None,
        rel2id=None,
        dataset=None,
        evaluation_datasets=None,
    ):
        checkpoint = {
            "protocol": "nsorn",
            "variant": self.variant,
            "num_nodes": self.encoder.standard_encoder.node_emb.size(0)
            if isinstance(self.encoder, FilteredTwoHopGatEncoder)
            else self.encoder.node_emb.size(0),
            "num_relations": self.decoder.rel_emb.size(0),
            "rdf_type_id": self.rdf_type_id,
            "subclass_id": self.subclass_id,
            "link_relation_ids": sorted(self.link_relation_ids),
            "rule_aux_weight": self.rule_aux_weight,
            "hidden_channels": HIDDEN_CHANNELS,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "regularization": REGULARIZATION,
            "seed": seed,
            "epochs": epochs,
            "best_val_loss": best_val_loss,
            "node2id": {str(node): idx for node, idx in (node2id or {}).items()},
            "rel2id": {str(rel): idx for rel, idx in (rel2id or {}).items()},
            "dataset": dict(dataset or {}),
            "evaluation_datasets": [
                dict(evaluation_dataset)
                for evaluation_dataset in (evaluation_datasets or [])
            ],
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    def encode(self, data):
        if isinstance(self.encoder, FilteredTwoHopGatEncoder):
            self.encoder.set_filters(data, self.rdf_type_id, self.subclass_id)
        elif isinstance(self.encoder, TwoHopGatEncoder):
            two_hop_relation_ids = [
                relation_id
                for relation_id in [
                    self.rdf_type_id,
                    self.subclass_id,
                    *self.link_relation_ids,
                ]
                if relation_id is not None
            ]
            self.encoder.set_two_hop_relations(two_hop_relation_ids)
        return self.encoder(data.train_pos_edge_index, data.train_edge_type)

    def decode(self, z, edge_index, edge_type):
        return self.decoder(z, edge_index, edge_type)

    def _ensure_candidate_nodes(self, data):
        if (
            self.class_nodes is not None
            and self.individual_nodes is not None
            and self.forbidden_tails_by_rel is not None
        ):
            return

        class_parts = []
        individual_parts = []
        if self.rdf_type_id is not None:
            type_mask = data.edge_type == self.rdf_type_id
            class_parts.append(data.edge_index[1, type_mask])
            individual_parts.append(data.edge_index[0, type_mask])
        if self.subclass_id is not None:
            subclass_mask = data.edge_type == self.subclass_id
            class_parts.extend(
                [data.edge_index[0, subclass_mask], data.edge_index[1, subclass_mask]]
            )
        for relation_id in self.link_relation_ids:
            link_mask = data.edge_type == relation_id
            individual_parts.extend(
                [data.edge_index[0, link_mask], data.edge_index[1, link_mask]]
            )

        all_nodes = torch.arange(
            data.edge_index.max().item() + 1, device=data.edge_index.device
        )
        self.class_nodes = unique_or_all(class_parts, all_nodes)
        self.individual_nodes = unique_or_all(individual_parts, all_nodes)
        self.forbidden_tails_by_rel = build_forbidden_tails_by_rel(
            data,
            self.rdf_type_id,
            self.subclass_id,
        )

    def _train(self, data, epoch_seed):
        self.train()
        self.optimizer.zero_grad()
        set_seed(epoch_seed)

        z = self.encode(data)
        self._ensure_candidate_nodes(data)

        pos_out = self.decode(z, data.train_pos_edge_index, data.train_edge_type)
        neg_edge_index, neg_edge_type = relation_aware_negative_edges(
            data.train_pos_edge_index,
            data.train_edge_type,
            z.size(0),
            self.rdf_type_id,
            self.subclass_id,
            self.link_relation_ids,
            self.class_nodes,
            self.individual_nodes,
            self.forbidden_tails_by_rel,
        )
        neg_out = self.decode(z, neg_edge_index, neg_edge_type)

        out = torch.cat([pos_out, neg_out])
        target = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        loss_ce = F.binary_cross_entropy_with_logits(out, target)
        if self.variant == "Filtered 2-Hop GAT" and self.rule_aux_weight > 0:
            aux_edge_index, aux_edge_type = rule_auxiliary_edges(
                data,
                self.rdf_type_id,
                self.subclass_id,
            )
            if aux_edge_index.size(1) > 0:
                aux_pos_out = self.decode(z, aux_edge_index, aux_edge_type)
                aux_neg_edge_index, aux_neg_edge_type = relation_aware_negative_edges(
                    aux_edge_index,
                    aux_edge_type,
                    z.size(0),
                    self.rdf_type_id,
                    self.subclass_id,
                    self.link_relation_ids,
                    self.class_nodes,
                    self.individual_nodes,
                    self.forbidden_tails_by_rel,
                )
                aux_neg_out = self.decode(z, aux_neg_edge_index, aux_neg_edge_type)
                aux_out = torch.cat([aux_pos_out, aux_neg_out])
                aux_target = torch.cat(
                    [torch.ones_like(aux_pos_out), torch.zeros_like(aux_neg_out)]
                )
                aux_loss = F.binary_cross_entropy_with_logits(aux_out, aux_target)
                loss_ce = loss_ce + self.rule_aux_weight * aux_loss
        reg_loss = z.pow(2).mean() + self.decoder.rel_emb.pow(2).mean()
        loss = loss_ce + REGULARIZATION * reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return loss.detach().item()

    def validation_loss(self, data):
        self.eval()
        with torch.no_grad():
            z = self.encode(data)
            self._ensure_candidate_nodes(data)
            if data.has_validation_edges:
                pos_edge_index = data.val_pos_edge_index
                edge_type = data.val_edge_type
            else:
                pos_edge_index = data.train_pos_edge_index
                edge_type = data.train_edge_type

            pos_out = self.decode(z, pos_edge_index, edge_type)
            neg_edge_index, neg_edge_type = relation_aware_negative_edges(
                pos_edge_index,
                edge_type,
                z.size(0),
                self.rdf_type_id,
                self.subclass_id,
                self.link_relation_ids,
                self.class_nodes,
                self.individual_nodes,
                self.forbidden_tails_by_rel,
            )
            neg_out = self.decode(z, neg_edge_index, neg_edge_type)

            out = torch.cat([pos_out, neg_out])
            target = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
            loss_ce = F.binary_cross_entropy_with_logits(out, target)
            reg_loss = z.pow(2).mean() + self.decoder.rel_emb.pow(2).mean()
            return (loss_ce + REGULARIZATION * reg_loss).item()

    def _eval(self, data, target_type, multiple):
        self.eval()
        with torch.no_grad():
            z = self.encode(data)
            edge_index, edge_type = specific_test_edges(data, target_type, multiple)
            return self._eval_edges(edge_index, edge_type, z)

    def _eval_edges(self, edge_index, edge_type, z=None):
        self.eval()
        with torch.no_grad():
            if z is None:
                raise ValueError("z must be provided for _eval_edges")
            return full_ranking_metrics(
                edge_index,
                edge_type,
                z,
                self.decoder,
                rdf_type_id=self.rdf_type_id,
                subclass_id=self.subclass_id,
                link_relation_ids=self.link_relation_ids,
                class_nodes=self.class_nodes,
                individual_nodes=self.individual_nodes,
                device=self.device,
            )


def unique_or_all(tensors, fallback):
    tensors = [tensor for tensor in tensors if tensor.numel() > 0]
    if not tensors:
        return fallback
    return torch.cat(tensors).unique()


def add_forbidden_edges(forbidden_tails_by_rel, edge_index, edge_type):
    for idx in range(edge_index.size(1)):
        relation_id = int(edge_type[idx].item())
        source_id = int(edge_index[0, idx].item())
        tail_id = int(edge_index[1, idx].item())
        forbidden_tails_by_rel.setdefault(relation_id, {}).setdefault(
            source_id, set()
        ).add(tail_id)


def build_forbidden_tails_by_rel(data, rdf_type_id, subclass_id):
    forbidden_tails_by_rel = {}
    known_edge_index = torch.cat(
        [data.train_pos_edge_index, data.val_pos_edge_index], dim=1
    )
    known_edge_type = torch.cat([data.train_edge_type, data.val_edge_type])
    add_forbidden_edges(
        forbidden_tails_by_rel, known_edge_index.cpu(), known_edge_type.cpu()
    )

    num_nodes = data.edge_index.max().item() + 1
    if subclass_id is not None:
        subclass_edges = relation_edge_index(data, subclass_id)
        subclass_2hop = two_hop_edges(
            subclass_edges,
            num_nodes,
            data.edge_index.device,
        )
        subclass_edge_type = torch.full(
            (subclass_2hop.size(1),),
            subclass_id,
            dtype=torch.long,
            device=subclass_2hop.device,
        )
        add_forbidden_edges(
            forbidden_tails_by_rel,
            subclass_2hop.cpu(),
            subclass_edge_type.cpu(),
        )

    if rdf_type_id is not None and subclass_id is not None:
        assertion_edges = relation_edge_index(data, rdf_type_id)
        subclass_edges = relation_edge_index(data, subclass_id)
        assertion_subclass_edges = composed_edges(
            assertion_edges,
            subclass_edges,
            num_nodes,
            data.edge_index.device,
        )
        assertion_subclass_edge_type = torch.full(
            (assertion_subclass_edges.size(1),),
            rdf_type_id,
            dtype=torch.long,
            device=assertion_subclass_edges.device,
        )
        add_forbidden_edges(
            forbidden_tails_by_rel,
            assertion_subclass_edges.cpu(),
            assertion_subclass_edge_type.cpu(),
        )

    return forbidden_tails_by_rel


def rule_auxiliary_edges(data, rdf_type_id, subclass_id):
    aux_edge_indices = []
    aux_edge_types = []

    if rdf_type_id is not None and subclass_id is not None:
        membership_edges = membership_rule_edges(data, rdf_type_id, subclass_id)
        if membership_edges.size(1) > 0:
            aux_edge_indices.append(membership_edges)
            aux_edge_types.append(
                torch.full(
                    (membership_edges.size(1),),
                    rdf_type_id,
                    dtype=torch.long,
                    device=membership_edges.device,
                )
            )

    if subclass_id is not None:
        subclass_edges = subsumption_rule_edges(data, subclass_id)
        if subclass_edges.size(1) > 0:
            aux_edge_indices.append(subclass_edges)
            aux_edge_types.append(
                torch.full(
                    (subclass_edges.size(1),),
                    subclass_id,
                    dtype=torch.long,
                    device=subclass_edges.device,
                )
            )

    if not aux_edge_indices:
        return data.train_pos_edge_index[:, :0], data.train_edge_type[:0]

    return torch.cat(aux_edge_indices, dim=1), torch.cat(aux_edge_types)


def relation_aware_negative_edges(
    pos_edge_index,
    edge_type,
    num_nodes,
    rdf_type_id,
    subclass_id,
    link_relation_ids,
    class_nodes,
    individual_nodes,
    forbidden_tails_by_rel,
):
    neg_edge_index_list = []
    neg_edge_type_list = []
    for rel in edge_type.unique():
        rel_mask = edge_type == rel
        pos_edges_rel = pos_edge_index[:, rel_mask]
        num_neg = pos_edges_rel.size(1)

        relation_id = rel.item()
        candidates = None
        if relation_id in {rdf_type_id, subclass_id}:
            candidates = class_nodes
        elif relation_id in link_relation_ids:
            candidates = individual_nodes

        if candidates is not None and len(candidates) > 0:
            src = pos_edges_rel[0]
            dst = sample_tails_excluding_forbidden(
                src,
                candidates,
                relation_id,
                forbidden_tails_by_rel,
            )
            neg_edges_rel = torch.stack([src, dst], dim=0)
        else:
            neg_edges_rel = negative_sampling(
                pos_edges_rel, num_nodes=num_nodes, num_neg_samples=num_neg
            )

        neg_edge_index_list.append(neg_edges_rel)
        neg_edge_type_list.append(
            torch.full((num_neg,), rel, dtype=torch.long, device=pos_edges_rel.device)
        )

    return torch.cat(neg_edge_index_list, dim=1), torch.cat(neg_edge_type_list, dim=0)


def sample_tails_excluding_forbidden(
    src,
    candidates,
    relation_id,
    forbidden_tails_by_rel,
):
    relation_forbidden = forbidden_tails_by_rel.get(relation_id, {})
    candidate_list = candidates.detach().cpu().tolist()
    sampled = []
    for source_id in src.detach().cpu().tolist():
        forbidden = relation_forbidden.get(int(source_id), set())
        allowed = [
            candidate for candidate in candidate_list if candidate not in forbidden
        ]
        if not allowed:
            allowed = candidate_list
        sampled.append(random.choice(allowed))
    return torch.tensor(sampled, dtype=torch.long, device=src.device)


def specific_test_edges(data, target_type, multiple):
    if multiple:
        target_tensor = torch.tensor(
            target_type,
            dtype=data.test_edge_type.dtype,
            device=data.test_edge_type.device,
        )
        mask = torch.isin(data.test_edge_type, target_tensor)
    else:
        mask = data.test_edge_type == target_type
    return data.test_pos_edge_index[:, mask], data.test_edge_type[mask]


@torch.no_grad()
def full_ranking_metrics(
    edge_index,
    edge_type,
    z,
    decoder,
    rdf_type_id,
    subclass_id,
    link_relation_ids,
    class_nodes,
    individual_nodes,
    device,
):
    if edge_index.size(1) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    src, dst = edge_index
    pos_scores = decoder(z, edge_index, edge_type)
    ranks = []

    all_nodes = torch.arange(z.size(0), device=device)
    for idx in range(edge_index.size(1)):
        relation_id = edge_type[idx].item()
        true_tail = dst[idx].item()
        if relation_id in {rdf_type_id, subclass_id} and class_nodes is not None:
            candidates = class_nodes
        elif relation_id in link_relation_ids and individual_nodes is not None:
            candidates = individual_nodes
        else:
            candidates = all_nodes

        candidates = candidates[candidates != true_tail]
        candidate_edge_index = torch.stack(
            [
                src[idx].repeat(candidates.size(0)),
                candidates,
            ],
            dim=0,
        )
        candidate_edge_type = torch.full(
            (candidates.size(0),),
            relation_id,
            dtype=edge_type.dtype,
            device=device,
        )
        scores = decoder(z, candidate_edge_index, candidate_edge_type)
        ranks.append((scores >= pos_scores[idx]).sum().item() + 1)

    ranks = torch.tensor(ranks, dtype=torch.float, device=device)
    return (
        (1.0 / ranks).mean().item(),
        (ranks <= 1).float().mean().item(),
        (ranks <= 5).float().mean().item(),
        (ranks <= 10).float().mean().item(),
    )


def is_generic_object_relation(relation):
    relation_text = str(relation)
    excluded_prefixes = (
        str(RDF),
        str(RDFS),
        str(OWL),
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2002/07/owl#",
    )
    return not relation_text.startswith(excluded_prefixes)


def target_relations(dataset_name, rel2id):
    try:
        namespace = get_namespace(dataset_name)
        link_relations = [
            rel_id
            for rel, rel_id in rel2id.items()
            if str(rel).startswith(str(namespace))
        ]
    except ValueError:
        link_relations = [
            rel_id for rel, rel_id in rel2id.items() if is_generic_object_relation(rel)
        ]

    return {
        "Membership": (rel2id.get(RDF.type), False),
        "Subsumption": (rel2id.get(RDFS.subClassOf), False),
        "Link Prediction": (link_relations, True),
    }


def cleaned_two_hop_relation_ids(dataset_name, rel2id):
    targets = target_relations(dataset_name, rel2id)
    relation_ids = [
        rel2id.get(RDF.type),
        rel2id.get(RDFS.subClassOf),
        *targets["Link Prediction"][0],
    ]
    return [relation_id for relation_id in relation_ids if relation_id is not None]


def edge_count(edge_index):
    return int(edge_index.size(1))


def membership_rule_edges(data, rdf_type_id, subclass_id):
    if rdf_type_id is None or subclass_id is None:
        return data.train_pos_edge_index[:, :0]
    assertion_edges = relation_edge_index(data, rdf_type_id)
    subclass_edges = relation_edge_index(data, subclass_id)
    return composed_edges(
        assertion_edges,
        subclass_edges,
        data.edge_index.max().item() + 1,
        data.edge_index.device,
    )


def subsumption_rule_edges(data, subclass_id):
    if subclass_id is None:
        return data.train_pos_edge_index[:, :0]
    subclass_edges = relation_edge_index(data, subclass_id)
    return two_hop_edges(
        subclass_edges,
        data.edge_index.max().item() + 1,
        data.edge_index.device,
    )


def test_edge_count(test_edge_type, relation_id):
    if relation_id is None:
        return 0
    return int((test_edge_type == relation_id).sum().item())


def covered_test_edges(test_edge_index, test_edge_type, relation_id, derived_edges):
    if relation_id is None:
        return 0
    test_edges = test_edge_index[:, test_edge_type == relation_id]
    if test_edges.numel() == 0 or derived_edges.numel() == 0:
        return 0
    derived_pairs = set(map(tuple, derived_edges.cpu().t().tolist()))
    test_pairs = set(map(tuple, test_edges.cpu().t().tolist()))
    return len(test_pairs & derived_pairs)


def build_nsorn_diagnostics(dataset, data, rel2id, test_edges_by_file):
    dataset_name, _ = dataset_names(dataset)
    rdf_type_id = rel2id.get(RDF.type)
    subclass_id = rel2id.get(RDFS.subClassOf)
    clean_relation_ids = cleaned_two_hop_relation_ids(dataset_name, rel2id)
    clean_relation_tensor = torch.tensor(
        clean_relation_ids,
        dtype=data.train_edge_type.dtype,
        device=data.train_edge_type.device,
    )
    if clean_relation_ids:
        clean_two_hop_base = data.train_pos_edge_index[
            :, torch.isin(data.train_edge_type, clean_relation_tensor)
        ]
    else:
        clean_two_hop_base = data.train_pos_edge_index[:, :0]

    num_nodes = data.edge_index.max().item() + 1
    raw_2hop = two_hop_edges(
        data.train_pos_edge_index, num_nodes, data.edge_index.device
    )
    clean_2hop = two_hop_edges(clean_two_hop_base, num_nodes, data.edge_index.device)
    subclass_edges = relation_edge_index(data, subclass_id)
    assertion_edges = relation_edge_index(data, rdf_type_id)
    subsumption_edges = subsumption_rule_edges(data, subclass_id)
    membership_edges = membership_rule_edges(data, rdf_type_id, subclass_id)
    test_edge_index, test_edge_type = test_edges_by_file[dataset["file"]]

    return {
        "train_edges": edge_count(data.train_pos_edge_index),
        "clean_two_hop_base_edges": edge_count(clean_two_hop_base),
        "raw_two_hop_edges": edge_count(raw_2hop),
        "clean_two_hop_edges": edge_count(clean_2hop),
        "train_subclass_edges": edge_count(subclass_edges),
        "train_membership_edges": edge_count(assertion_edges),
        "subclass_rule_edges": edge_count(subsumption_edges),
        "membership_rule_edges": edge_count(membership_edges),
        "test_subsumption_edges": test_edge_count(test_edge_type, subclass_id),
        "test_membership_edges": test_edge_count(test_edge_type, rdf_type_id),
        "test_subsumption_covered_by_rule": covered_test_edges(
            test_edge_index,
            test_edge_type,
            subclass_id,
            subsumption_edges,
        ),
        "test_membership_covered_by_rule": covered_test_edges(
            test_edge_index,
            test_edge_type,
            rdf_type_id,
            membership_edges,
        ),
    }


def train_variant(
    dataset,
    evaluation_datasets,
    variant,
    device,
    seed,
    epochs,
    models_dir,
    rule_aux_weight=0.0,
):
    dataset_name, _ = dataset_names(dataset)
    set_seed(seed)
    data, node2id, rel2id, test_edges_by_file = load_shared_train_data(
        dataset, evaluation_datasets
    )
    data = data.to(device)
    link_relation_ids = target_relations(dataset_name, rel2id)["Link Prediction"][0]

    model = NsornGatReasoner(
        variant=variant,
        device=device,
        num_nodes=data.edge_index.max().item() + 1,
        num_relations=len(rel2id),
        rdf_type_id=rel2id.get(RDF.type),
        subclass_id=rel2id.get(RDFS.subClassOf),
        link_relation_ids=link_relation_ids,
        rule_aux_weight=rule_aux_weight,
    ).to(device)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs + 1):
        loss = model._train(data, epoch_seed=seed + epoch)
        if epoch % 50 == 0 or epoch == epochs:
            val_loss = model.validation_loss(data)
            print(
                f"{variant} seed {seed} epoch {epoch}, "
                f"loss: {loss:.4f}, val_loss: {val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"{variant} seed {seed} using best val_loss: {best_val_loss:.4f}")

    checkpoint_name = "_".join(
        [
            checkpoint_safe_name(dataset["model_file"]),
            "nsorn",
            checkpoint_safe_name(variant),
            f"seed{seed}",
        ]
    )
    checkpoint_path = Path(models_dir) / checkpoint_name
    model.save_checkpoint(
        checkpoint_path,
        best_val_loss=best_val_loss,
        seed=seed,
        epochs=epochs,
        node2id=node2id,
        rel2id=rel2id,
        dataset=dataset,
        evaluation_datasets=evaluation_datasets,
    )
    print(f"Saved nsorn checkpoint to {checkpoint_path}")

    return model, data, node2id, rel2id, test_edges_by_file


def evaluate_variant(model, shared_data, dataset, rel2id, test_edges_by_file):
    dataset_name, _ = dataset_names(dataset)
    data = with_test_edges(shared_data, dataset, test_edges_by_file)

    metrics = {}
    for name, (target_type, multiple) in target_relations(dataset_name, rel2id).items():
        if target_type is None or target_type == []:
            metrics[name] = (float("nan"), float("nan"), float("nan"), float("nan"))
        else:
            metrics[name] = model._eval(data, target_type, multiple)
    return metrics


def format_metric_tuple(metrics):
    return " & ".join(f"{value:.3f}" for value in metrics)


def write_diagnostics(handle, diagnostics):
    if not diagnostics:
        return

    handle.write("Diagnostics:\n")
    labels = {
        "train_edges": "Train 1-hop edges",
        "clean_two_hop_base_edges": "Cleaned 2-hop base edges",
        "raw_two_hop_edges": "Raw 2-hop edges",
        "clean_two_hop_edges": "Cleaned 2-hop edges",
        "train_membership_edges": "Train rdf:type edges",
        "train_subclass_edges": "Train subClassOf edges",
        "membership_rule_edges": "Rule rdf:type+subClassOf edges",
        "subclass_rule_edges": "Rule subClassOf+subClassOf edges",
        "test_membership_edges": "Test membership edges",
        "test_membership_covered_by_rule": "Test membership edges covered by rule",
        "test_subsumption_edges": "Test subsumption edges",
        "test_subsumption_covered_by_rule": "Test subsumption edges covered by rule",
    }
    for key, label in labels.items():
        handle.write(f"{label}: {diagnostics[key]}\n")
    handle.write("\n")


def write_nsorn_results(path, dataset, results, diagnostics=None, rule_aux_weight=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_name, file_name = dataset_names(dataset)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("NSORN-Compatible GAT Experiment Results\n")
        handle.write(f"Dataset: {dataset_name}\n")
        handle.write(f"File: {file_name}\n")
        handle.write(
            "Protocol: clean train graph, clean validation graph, selected "
            "clean/noisy test graph held out, 5 seeds by default, DistMult "
            "decoder, best validation-loss checkpoint, typed full tail ranking\n\n"
        )
        handle.write(
            f"Filtered auxiliary rule-supervision weight: {rule_aux_weight}\n\n"
        )
        write_diagnostics(handle, diagnostics)

        for variant, task_results in results.items():
            handle.write(f"Model: {variant}\n")
            for task_name in ("Membership", "Subsumption", "Link Prediction"):
                runs = task_results[task_name]
                handle.write(f"{task_name}:\n")
                for run_idx, metrics in enumerate(runs, start=1):
                    handle.write(f"Run {run_idx}: {format_metric_tuple(metrics)}\n")
                mean = np.nanmean(np.array(runs), axis=0)
                handle.write(f"Mean: {format_metric_tuple(mean)}\n\n")
            handle.write("-" * 40 + "\n")


def training_key(dataset):
    return (
        dataset["loader"],
        dataset["path"],
        dataset["train_file"],
        str(validation_path(dataset)),
    )


def run_nsorn_protocol(
    device,
    datasets,
    results_dir,
    models_dir,
    epochs=300,
    runs=5,
    rule_aux_weight=0.0,
):
    variants = ("GAT", "2-Hop GAT", "Filtered 2-Hop GAT")
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    dataset_groups = {}
    for dataset in datasets:
        dataset_groups.setdefault(training_key(dataset), []).append(dataset)

    for _, group_datasets in dataset_groups.items():
        results_by_dataset = {
            dataset["file"]: {
                variant: {
                    "Membership": [],
                    "Subsumption": [],
                    "Link Prediction": [],
                }
                for variant in variants
            }
            for dataset in group_datasets
        }

        train_dataset = group_datasets[0]
        diagnostics_data, _, diagnostics_rel2id, diagnostics_test_edges = (
            load_shared_train_data(train_dataset, group_datasets)
        )
        diagnostics_by_dataset = {
            dataset["file"]: build_nsorn_diagnostics(
                dataset,
                diagnostics_data,
                diagnostics_rel2id,
                diagnostics_test_edges,
            )
            for dataset in group_datasets
        }
        for variant in variants:
            for run_idx in range(runs):
                seed = 42 + run_idx
                (
                    model,
                    shared_data,
                    node2id,
                    rel2id,
                    test_edges_by_file,
                ) = train_variant(
                    train_dataset,
                    group_datasets,
                    variant,
                    device,
                    seed,
                    epochs,
                    models_dir,
                    rule_aux_weight,
                )
                for dataset in group_datasets:
                    metrics = evaluate_variant(
                        model, shared_data, dataset, rel2id, test_edges_by_file
                    )
                    for task_name, task_metrics in metrics.items():
                        results_by_dataset[dataset["file"]][variant][task_name].append(
                            task_metrics
                        )

        for dataset in group_datasets:
            results_path = Path(results_dir) / f"{dataset['file']}_nsorn_protocol.txt"
            write_nsorn_results(
                results_path,
                dataset,
                results_by_dataset[dataset["file"]],
                diagnostics_by_dataset[dataset["file"]],
                rule_aux_weight,
            )
            print(f"Wrote nsorn results to {results_path}")
