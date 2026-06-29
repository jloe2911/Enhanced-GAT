import pandas as pd
import networkx as nx
import xml.etree.ElementTree as ET
import time

import re


RDF_NS = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDF_ABOUT = f"{RDF_NS}about"
RDF_NODE_ID = f"{RDF_NS}nodeID"
RDF_RESOURCE = f"{RDF_NS}resource"
RDFS_SUBCLASS_URI = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDF_TYPE_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
SCHEMA_TYPE_URIS = {
    "http://www.w3.org/2002/07/owl#AnnotationProperty",
    "http://www.w3.org/2002/07/owl#Class",
    "http://www.w3.org/2002/07/owl#DatatypeProperty",
    "http://www.w3.org/2002/07/owl#FunctionalProperty",
    "http://www.w3.org/2002/07/owl#InverseFunctionalProperty",
    "http://www.w3.org/2002/07/owl#NamedIndividual",
    "http://www.w3.org/2002/07/owl#ObjectProperty",
    "http://www.w3.org/2002/07/owl#Ontology",
    "http://www.w3.org/2002/07/owl#Restriction",
    "http://www.w3.org/2002/07/owl#SymmetricProperty",
    "http://www.w3.org/2002/07/owl#TransitiveProperty",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#List",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
    "http://www.w3.org/2000/01/rdf-schema#Class",
}
SCHEMA_PREDICATE_PREFIXES = (
    "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}",
    "{http://www.w3.org/2000/01/rdf-schema#}",
    "{http://www.w3.org/2002/07/owl#}",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "<http://www.w3.org/2000/01/rdf-schema#",
    "<http://www.w3.org/2002/07/owl#",
)


def create_graph(df, all_nodes=None):
    graph = nx.MultiDiGraph()
    node_num = 0
    edge_num = 0
    nodes = dict()
    edges = dict()

    if all_nodes is not None:
        for node in all_nodes:
            if node not in nodes:
                nodes[node] = node_num
                graph.add_node(node_num)
                node_num += 1

    for i, row in df.iterrows():
        if row["s"] not in nodes:
            nodes[row["s"]] = node_num
            node_num += 1
        if row["o"] not in nodes:
            nodes[row["o"]] = node_num
            node_num += 1
        if row["p"] not in edges:
            edges[row["p"]] = edge_num
            edge_num += 1
        graph.add_edge(nodes[row["s"]], nodes[row["o"]], type=edges[row["p"]])

    return graph


def _blank_node(node_id):
    return f"_:{node_id}"


def _rdf_subject(elem):
    subject = elem.attrib.get(RDF_ABOUT)
    if subject is not None:
        return subject

    node_id = elem.attrib.get(RDF_NODE_ID)
    if node_id is not None:
        return _blank_node(node_id)

    return None


def _rdf_object(child):
    obj = child.attrib.get(RDF_RESOURCE)
    if obj is not None:
        return obj

    node_id = child.attrib.get(RDF_NODE_ID)
    if node_id is not None:
        return _blank_node(node_id)

    literal = (child.text or "").strip()
    if literal:
        return literal

    return None


def _load_rdfxml_file(pathfilename):
    entity_dict = {"s": [], "p": [], "o": []}

    for _, elem in ET.iterparse(pathfilename, events=("end",)):
        subject = _rdf_subject(elem)
        if subject is None:
            continue

        for child in elem:
            predicate = child.tag
            obj = _rdf_object(child)
            if obj is None:
                continue

            entity_dict["s"].append(subject)
            entity_dict["p"].append(predicate)
            entity_dict["o"].append(obj)

    return pd.DataFrame(entity_dict)


def _collect_nodes_from_frames(frames):
    if not frames:
        return []

    return sorted(
        pd.unique(
            pd.concat(
                [frame[column] for frame in frames for column in ("s", "o")],
                ignore_index=True,
            )
        )
    )


def load_ore_files(pathfilename):
    with open(pathfilename, "r", encoding="utf-8", newline="\n") as nt_file:
        first_chunk = nt_file.read(256).lstrip()

    if first_chunk.startswith("<?xml") or first_chunk.startswith("<rdf:RDF"):
        return _load_rdfxml_file(pathfilename)

    nt_file = open(pathfilename, "r", encoding="utf-8", newline="\n")
    lines = nt_file.read().split("\r\n")

    entity_dict = dict({"s": [], "p": [], "o": []})

    for line in lines[:-1]:
        split = line.split(" ")
        entity_dict["s"].append(re.findall(r"(?<=\().+", split[0])[0])
        entity_dict["p"].append(re.findall(r".+?(?=\()", split[0])[0])
        entity_dict["o"].append(split[1].replace(")", ""))

    df = pd.DataFrame(entity_dict)

    return df


def _is_subclass_predicate(predicate):
    return predicate in [
        "SubClassOf",
        RDFS_SUBCLASS_URI,
        f"<{RDFS_SUBCLASS_URI}>",
        f"{{{RDFS_SUBCLASS_URI.rsplit('#', 1)[0]}#}}subClassOf",
    ]


def _is_assertion_predicate(predicate):
    return predicate in [
        "ClassAssertion",
        RDF_TYPE_URI,
        f"<{RDF_TYPE_URI}>",
        f"{RDF_NS}type",
    ]


def _is_schema_predicate(predicate):
    return any(predicate.startswith(prefix) for prefix in SCHEMA_PREDICATE_PREFIXES)


def _is_schema_type_object(obj):
    return obj.strip("<>") in SCHEMA_TYPE_URIS


def _is_membership_assertion(row):
    predicate = row["p"]
    if predicate == "ClassAssertion":
        return True
    if not _is_assertion_predicate(predicate):
        return False

    return not row["s"].startswith("_:") and not _is_schema_type_object(row["o"])


def _is_object_property_assertion(row):
    predicate = row["p"]
    return (
        not _is_subclass_predicate(predicate)
        and not _is_assertion_predicate(predicate)
        and not _is_schema_predicate(predicate)
    )


def load_ore_graphs(path, train_file, test_file, all_node_files=None):
    print("Running...", train_file, test_file)

    df_train = load_ore_files(path + train_file)
    df_test = load_ore_files(path + test_file)
    if all_node_files is None:
        all_nodes = _collect_nodes_from_frames([df_train, df_test])
    else:
        node_frames = [load_ore_files(path + filename) for filename in all_node_files]
        all_nodes = _collect_nodes_from_frames(node_frames)

    train_background_mask = df_train.apply(_is_object_property_assertion, axis=1)
    df_train_background = df_train[train_background_mask]
    g_train = create_graph(df_train, all_nodes)

    df_train_filter_subclass = df_train[df_train["p"].map(_is_subclass_predicate)]
    g_train_filter_subclass = create_graph(df_train_filter_subclass, all_nodes)

    df_train_filter_assertion = df_train[
        df_train.apply(_is_membership_assertion, axis=1)
    ]
    g_train_filter_assertion = create_graph(df_train_filter_assertion, all_nodes)

    print(
        "# Train - Triplets: "
        f"{len(df_train)}, # Object property triplets: {len(df_train_background)}, "
        f"# Nodes: {g_train.number_of_nodes()}, # Edges: {g_train.number_of_edges()}"
    )

    test_background_mask = df_test.apply(_is_object_property_assertion, axis=1)
    df_test_background = df_test[test_background_mask]
    g_test = create_graph(df_test_background, all_nodes)

    df_test_filter_subclass = df_test[df_test["p"].map(_is_subclass_predicate)]
    g_test_filter_subclass = create_graph(df_test_filter_subclass, all_nodes)

    df_test_filter_assertion = df_test[df_test.apply(_is_membership_assertion, axis=1)]
    g_test_filter_assertion = create_graph(df_test_filter_assertion, all_nodes)

    print(
        f"# Test - Triplets: {len(df_test_background)}, # Nodes: {g_test.number_of_nodes()}, # Edges: {g_test.number_of_edges()}"
    )

    print()

    return (
        g_train,
        g_train_filter_subclass,
        g_train_filter_assertion,
        g_test,
        g_test_filter_subclass,
        g_test_filter_assertion,
    )


def load_clg_files(pathfilename):

    nt_file = open(pathfilename, "r")
    lines = nt_file.readlines()
    lines = lines[0].split(" .")

    entity_dict = dict({"s": [], "p": [], "o": []})

    for line in lines[:-1]:
        entities = line.split(" ")
        entity_dict["s"].append(entities[0])
        entity_dict["p"].append(entities[1])
        entity_dict["o"].append(entities[2])

    df = pd.DataFrame(entity_dict)

    return df


def load_clg_graphs(path, train_file, test_file, all_node_files=None):
    print("Running...", train_file, test_file)

    df_train = load_clg_files(path + train_file)
    df_test = load_clg_files(path + test_file)
    if all_node_files is None:
        all_nodes = _collect_nodes_from_frames([df_train, df_test])
    else:
        node_frames = [load_clg_files(path + filename) for filename in all_node_files]
        all_nodes = _collect_nodes_from_frames(node_frames)

    train_background_mask = df_train.apply(_is_object_property_assertion, axis=1)
    df_train_background = df_train[train_background_mask]
    g_train = create_graph(df_train, all_nodes)

    df_train_filter_subclass = df_train[df_train["p"].map(_is_subclass_predicate)]
    g_train_filter_subclass = create_graph(df_train_filter_subclass, all_nodes)

    df_train_filter_assertion = df_train[
        df_train.apply(_is_membership_assertion, axis=1)
    ]
    g_train_filter_assertion = create_graph(df_train_filter_assertion, all_nodes)

    print(
        "# Train - Triplets: "
        f"{len(df_train)}, # Object property triplets: {len(df_train_background)}, "
        f"# Nodes: {g_train.number_of_nodes()}, # Edges: {g_train.number_of_edges()}"
    )

    test_background_mask = df_test.apply(_is_object_property_assertion, axis=1)
    df_test_background = df_test[test_background_mask]
    g_test = create_graph(df_test_background, all_nodes)

    df_test_filter_subclass = df_test[df_test["p"].map(_is_subclass_predicate)]
    g_test_filter_subclass = create_graph(df_test_filter_subclass, all_nodes)

    df_test_filter_assertion = df_test[df_test.apply(_is_membership_assertion, axis=1)]
    g_test_filter_assertion = create_graph(df_test_filter_assertion, all_nodes)

    print(
        f"# Test - Triplets: {len(df_test_background)}, # Nodes: {g_test.number_of_nodes()}, # Edges: {g_test.number_of_edges()}"
    )

    print()

    return (
        g_train,
        g_train_filter_subclass,
        g_train_filter_assertion,
        g_test,
        g_test_filter_subclass,
        g_test_filter_assertion,
    )


def build_dataset_configs(path, train_file, file_prefix, test_files, loader="ore"):
    configs = []
    for test_file in test_files:
        suffix = test_file.rsplit(".", 1)[0].replace(train_file.rsplit(".", 1)[0], "")
        suffix = suffix.lstrip("_") or "base"
        configs.append(
            {
                "path": path,
                "train_file": train_file,
                "test_file": test_file,
                "file": f"{file_prefix}_{suffix}",
                "model_file": file_prefix,
                "loader": loader,
                "node_files": [train_file, *test_files],
            }
        )
    return configs


def train_dataset_collection(device, datasets):
    from src.gnn import GNN

    loaders = {
        "ore": load_ore_graphs,
        "clg": load_clg_graphs,
    }
    training_runs = [
        ("GAT", "GAT", False),
        ("2-Hop GAT", "2HopGAT", False),
        ("2-Hop GAT", "Filtered2HopGAT", True),
    ]

    trained_datasets = set()

    for dataset in datasets:
        loader_name = dataset.get("loader", "ore")
        graph_loader = loaders[loader_name]
        model_file = dataset.get("model_file", dataset["file"])
        training_key = (
            loader_name,
            dataset["path"],
            dataset["train_file"],
            model_file,
        )

        if training_key in trained_datasets:
            continue

        (
            g_train,
            g_train_filter_subclass,
            g_train_filter_assertion,
            g_test,
            g_test_filter_subclass,
            g_test_filter_assertion,
        ) = graph_loader(
            dataset["path"],
            dataset["train_file"],
            dataset["test_file"],
            dataset.get("node_files"),
        )

        for variant, checkpoint_suffix, use_filters in training_runs:
            print(variant if not use_filters else f"Filtered {variant}")
            st = time.time()
            model = GNN()
            if use_filters:
                model._train(
                    device,
                    "GAT Reasoner",
                    g_train,
                    g_train_filter_subclass,
                    g_train_filter_assertion,
                )
            else:
                model._train(device, variant, g_train)
            model.save_checkpoint(f"Models/{model_file}_{checkpoint_suffix}")
            et = time.time()
            elapsed_time = et - st
            print(
                f"Run time: {elapsed_time:.0f} seconds, {elapsed_time / 60:.0f} minutes"
            )
            print()

        trained_datasets.add(training_key)
