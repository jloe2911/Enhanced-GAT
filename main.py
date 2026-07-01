import argparse
import random
from pathlib import Path

import torch
import torch_geometric

from src.gnn import GNN, format_metrics
from src.nsorn_protocol import run_nsorn_protocol
from src.utils import build_dataset_configs, load_clg_graphs, load_ore_graphs


SEED = 10
TRAINING_RUNS = [
    ("GAT", "GAT", False),
    ("2-Hop GAT", "2HopGAT", False),
    ("2-Hop GAT", "Filtered2HopGAT", True),
]


def get_dataset_groups():
    return {
        "noise_pizza_100": build_dataset_configs(
            path="./datasets/noise_pizza/",
            train_file="pizza_100_train.owl",
            file_prefix="noise_pizza_100",
            test_files=[
                "pizza_100_test.owl",
                "pizza_100_random_0.25_test.owl",
                "pizza_100_random_0.5_test.owl",
                "pizza_100_random_0.75_test.owl",
                "pizza_100_random_1_test.owl",
                "pizza_100_gnn_0.25_test.owl",
                "pizza_100_gnn_0.5_test.owl",
                "pizza_100_gnn_0.75_test.owl",
                "pizza_100_gnn_1_test.owl",
                "pizza_100_logical_0.25_test.owl",
                "pizza_100_logical_0.5_test.owl",
                "pizza_100_logical_0.75_test.owl",
                "pizza_100_logical_1_test.owl",
            ],
            loader="ore",
        ),
        "noise_pizza_250": build_dataset_configs(
            path="./datasets/noise_pizza/",
            train_file="pizza_250_train.owl",
            file_prefix="noise_pizza_250",
            test_files=[
                "pizza_250_test.owl",
                "pizza_250_random_0.25_test.owl",
                "pizza_250_random_0.5_test.owl",
                "pizza_250_random_0.75_test.owl",
                "pizza_250_random_1_test.owl",
                "pizza_250_gnn_0.25_test.owl",
                "pizza_250_gnn_0.5_test.owl",
                "pizza_250_gnn_0.75_test.owl",
                "pizza_250_gnn_1_test.owl",
                "pizza_250_logical_0.25_test.owl",
                "pizza_250_logical_0.5_test.owl",
                "pizza_250_logical_0.75_test.owl",
                "pizza_250_logical_1_test.owl",
            ],
            loader="ore",
        ),
        "noise_family": build_dataset_configs(
            path="./datasets/noise_family/",
            train_file="family_train.owl",
            file_prefix="noise_family",
            test_files=[
                "family_test.owl",
                "family_random_0.25_test.owl",
                "family_random_0.5_test.owl",
                "family_random_0.75_test.owl",
                "family_random_1_test.owl",
                "family_gnn_0.25_test.owl",
                "family_gnn_0.5_test.owl",
                "family_gnn_0.75_test.owl",
                "family_gnn_1_test.owl",
                "family_logical_0.25_test.owl",
                "family_logical_0.5_test.owl",
                "family_logical_0.75_test.owl",
                "family_logical_1_test.owl",
            ],
            loader="ore",
        ),
        "noise_owl2bench": build_dataset_configs(
            path="./datasets/noise_OWL2Bench/",
            train_file="OWL2DL-1_train.owl",
            file_prefix="noise_OWL2DL-1",
            test_files=[
                "OWL2DL-1_test.owl",
                "OWL2DL-1_random_0.25_test.owl",
                "OWL2DL-1_random_0.5_test.owl",
                "OWL2DL-1_random_0.75_test.owl",
                "OWL2DL-1_random_1_test.owl",
                "OWL2DL-1_gnn_0.25_test.owl",
                "OWL2DL-1_gnn_0.5_test.owl",
                "OWL2DL-1_gnn_0.75_test.owl",
                "OWL2DL-1_gnn_1_test.owl",
                "OWL2DL-1_logical_0.25_test.owl",
                "OWL2DL-1_logical_0.5_test.owl",
                "OWL2DL-1_logical_0.75_test.owl",
                "OWL2DL-1_logical_1_test.owl",
            ],
            loader="ore",
        ),
        "owl2bench": [
            {
                "path": "./datasets/OWL2Bench/OWL2Bench1/",
                "train_file": "_train_OWL2Bench1.owl",
                "test_file": "_test_OWL2Bench1.owl",
                "file": "OWL2Bench1",
                "model_file": "OWL2Bench1",
                "loader": "ore",
            },
            {
                "path": "./datasets/OWL2Bench/OWL2Bench2/",
                "train_file": "_train_OWL2Bench2.owl",
                "test_file": "_test_OWL2Bench2.owl",
                "file": "OWL2Bench2",
                "model_file": "OWL2Bench2",
                "loader": "ore",
            },
        ],
        "ore": [
            {
                "path": "./datasets/ORE/ORE1/",
                "train_file": "_train_ORE1.owl",
                "test_file": "_test_ORE1.owl",
                "file": "ORE1",
                "model_file": "ORE1",
                "loader": "ore",
            },
            {
                "path": "./datasets/ORE/ORE2/",
                "train_file": "_train_ORE2.owl",
                "test_file": "_test_ORE2.owl",
                "file": "ORE2",
                "model_file": "ORE2",
                "loader": "ore",
            },
            {
                "path": "./datasets/ORE/ORE3/",
                "train_file": "_train_ORE3.owl",
                "test_file": "_test_ORE3.owl",
                "file": "ORE3",
                "model_file": "ORE3",
                "loader": "ore",
            },
        ],
        "clg": [
            {
                "path": "./datasets/clg/clg_10e4/",
                "train_file": "clg_10e4-train.nt",
                "test_file": "clg_10e4-test.nt",
                "file": "clg_10e4",
                "model_file": "clg_10e4",
                "loader": "clg",
            },
            {
                "path": "./datasets/clg/clg_10e5/",
                "train_file": "clg_10e5-train.nt",
                "test_file": "clg_10e5-test.nt",
                "file": "clg_10e5",
                "model_file": "clg_10e5",
                "loader": "clg",
            },
        ],
    }


def flatten_selected_datasets(group_names, dataset_names=None):
    groups = get_dataset_groups()
    selected = []

    for group_name in group_names:
        if group_name == "all":
            for datasets in groups.values():
                selected.extend(datasets)
            continue
        if group_name not in groups:
            raise ValueError(f"Unknown group '{group_name}'.")
        selected.extend(groups[group_name])

    if dataset_names:
        wanted = set(dataset_names)
        selected = [dataset for dataset in selected if dataset["file"] in wanted]

    deduped = []
    seen = set()
    for dataset in selected:
        if dataset["file"] in seen:
            continue
        seen.add(dataset["file"])
        deduped.append(dataset)

    return deduped


def get_graph_loader(loader_name):
    loaders = {
        "ore": load_ore_graphs,
        "clg": load_clg_graphs,
    }
    return loaders[loader_name]


def load_graph_bundle(dataset):
    graph_loader = get_graph_loader(dataset["loader"])
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

    return {
        "train": g_train,
        "train_subclass": g_train_filter_subclass,
        "train_assertion": g_train_filter_assertion,
        "test": g_test,
        "test_subclass": g_test_filter_subclass,
        "test_assertion": g_test_filter_assertion,
    }


def train_dataset(device, dataset, graphs, models_dir, epochs=None):
    model_file = dataset["model_file"]

    for variant, checkpoint_suffix, use_filters in TRAINING_RUNS:
        label = variant if not use_filters else f"Filtered {variant}"
        print(label)

        model = GNN()
        if epochs is not None:
            model.epochs = epochs

        if use_filters:
            model._train(
                device,
                "GAT Reasoner",
                graphs["train"],
                graphs["train_subclass"],
                graphs["train_assertion"],
            )
        else:
            model._train(device, variant, graphs["train"])

        checkpoint_path = models_dir / f"{model_file}_{checkpoint_suffix}"
        model.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        print()


def write_header(handle, dataset, device):
    handle.write("Enhanced GAT Experiment Results\n")
    handle.write(f"Device: {device}\n")
    handle.write(f"Dataset file id: {dataset['file']}\n")
    handle.write(f"Path: {dataset['path']}\n")
    handle.write(f"Train file: {dataset['train_file']}\n")
    handle.write(f"Test file: {dataset['test_file']}\n")
    handle.write("\n")


def write_metric_block(handle, dataset_name, model_name, relation_name, metrics):
    handle.write(f"Dataset: {dataset_name}\n")
    handle.write(f"Model: {model_name}\n")
    handle.write(f"Relation set: {relation_name}\n")
    handle.write(f"{format_metrics(metrics)}\n")
    handle.write("-" * 40 + "\n")


def write_error_block(handle, dataset_name, model_name, message):
    handle.write(f"Dataset: {dataset_name}\n")
    handle.write(f"Model: {model_name}\n")
    handle.write(f"ERROR: {message}\n")
    handle.write("-" * 40 + "\n")


def evaluate_relation(
    model,
    variant,
    graph,
    message_graph,
    subclass_graph,
    assertion_graph,
    max_num,
):
    if graph.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    if graph.number_of_edges() == 0:
        raise ValueError("graph has no edges")

    capped_max_num = min(graph.number_of_nodes(), max_num)

    if variant == "Filtered 2-Hop GAT":
        return model._eval(
            capped_max_num,
            "GAT Reasoner",
            graph,
            message_graph,
            subclass_graph,
            assertion_graph,
        )

    base_variant = "GAT" if variant == "GAT" else "2-Hop GAT"
    return model._eval(capped_max_num, base_variant, graph, message_graph)


def evaluate_dataset(device, dataset, graphs, models_dir, results_dir, max_num=100):
    results_path = results_dir / f"{dataset['file']}.txt"
    model_file = dataset["model_file"]

    evaluation_runs = [
        ("GAT", models_dir / f"{model_file}_GAT", "GAT"),
        ("2-Hop GAT", models_dir / f"{model_file}_2HopGAT", "2-Hop GAT"),
        (
            "Filtered 2-Hop GAT",
            models_dir / f"{model_file}_Filtered2HopGAT",
            "GAT Reasoner",
        ),
    ]
    relation_sets = [
        ("SubClass Relations", graphs["test_subclass"]),
        ("Assertion Relations", graphs["test_assertion"]),
        ("Object Property Relations", graphs["test"]),
    ]

    with open(results_path, "w", encoding="utf-8") as handle:
        write_header(handle, dataset, device)

        for model_name, checkpoint_path, fallback_variant in evaluation_runs:
            if not checkpoint_path.exists():
                write_error_block(
                    handle,
                    dataset["file"],
                    model_name,
                    f"Missing checkpoint: {checkpoint_path}",
                )
                continue

            try:
                model = GNN.load_checkpoint(
                    checkpoint_path, device, fallback_variant=fallback_variant
                )
            except Exception as exc:
                write_error_block(handle, dataset["file"], model_name, str(exc))
                continue

            for relation_name, graph in relation_sets:
                try:
                    metrics = evaluate_relation(
                        model,
                        model_name,
                        graph,
                        graphs["train"],
                        graphs["train_subclass"],
                        graphs["train_assertion"],
                        max_num,
                    )
                    write_metric_block(
                        handle, dataset["file"], model_name, relation_name, metrics
                    )
                except Exception as exc:
                    write_error_block(
                        handle,
                        dataset["file"],
                        model_name,
                        f"{relation_name}: {exc}",
                    )

    print(f"Wrote results to {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Enhanced-GAT experiments."
    )
    parser.add_argument(
        "--protocol",
        choices=["current", "nsorn"],
        default="current",
        help=(
            "Experiment protocol. 'current' uses this repo's fixed train/test "
            "workflow; 'nsorn' follows the OWL2Vec-style NSORN fixed split: "
            "clean train graph, clean validation graph, selected clean/noisy "
            "test graph held out, DistMult decoder, typed full ranking, and "
            "repeated seeds."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "all"],
        default="all",
        help="Run training only, evaluation only, or both.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        help=(
            "Dataset groups to run: all, noise_pizza_100, noise_pizza_250, "
            "noise_family, noise_owl2bench, owl2bench, ore, clg."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset file ids to filter to, e.g. OWL2Bench1 or ORE1.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory used for saved checkpoints.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where per-dataset result files are written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Optional override for GNN training epochs.",
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=100,
        help="Maximum number of sampled negative candidates during evaluation.",
    )
    parser.add_argument(
        "--nsorn-runs",
        type=int,
        default=5,
        help="Number of NSORN-protocol repeated runs. Default: 5.",
    )
    parser.add_argument(
        "--nsorn-epochs",
        type=int,
        default=300,
        help="Number of epochs for each NSORN-protocol run. Default: 300.",
    )
    parser.add_argument(
        "--nsorn-rule-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary rule-supervision loss weight for Filtered 2-Hop GAT. "
            "Default: 0.0, which disables the extension."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = flatten_selected_datasets(args.groups, args.datasets)
    if not datasets:
        raise ValueError("No datasets matched the requested filters.")

    print(torch.__version__)
    print(torch_geometric.__version__)
    print(device)
    print()

    if args.protocol == "nsorn":
        run_nsorn_protocol(
            device,
            datasets,
            args.results_dir,
            models_dir,
            epochs=args.nsorn_epochs,
            runs=args.nsorn_runs,
            rule_aux_weight=args.nsorn_rule_aux_weight,
        )
        return

    trained_datasets = set()

    for dataset in datasets:
        print(f"=== {dataset['file']} ===")
        graphs = load_graph_bundle(dataset)

        if args.mode in {"train", "all"}:
            training_key = (
                dataset["loader"],
                dataset["path"],
                dataset["train_file"],
                dataset["model_file"],
            )
            if training_key not in trained_datasets:
                train_dataset(device, dataset, graphs, models_dir, epochs=args.epochs)
                trained_datasets.add(training_key)
            else:
                print(
                    "Skipping training; reusing checkpoints for "
                    f"{dataset['model_file']}."
                )
                print()

        if args.mode in {"eval", "all"}:
            evaluate_dataset(
                device,
                dataset,
                graphs,
                models_dir,
                results_dir,
                max_num=args.max_num,
            )


if __name__ == "__main__":
    main()
