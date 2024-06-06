import argparse
import torch
import util
import jinja2
from torchdrug.data import Graph
from siamdiff.task import SiamDiff, DiffusionProteinGenerator
from siamdiff.model import GVPGNN
from torchdrug import core, utils


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = jinja2.meta.find_undeclared_variables(ast)
    return vars


def load_pretrained_model(checkpoint_path, config_path, vars=None):
    cfg = util.load_config(config_path, context=vars)

    # Load task configuration
    task_config = core.Configurable.load_config_dict(cfg["task"])
    task_config = task_config.config_dict()
    # Extract the model configuration
    model_config = task_config.pop("model")
    model_config.pop("class")
    # Initialize the GVPGNN model
    model = GVPGNN(**model_config)

    # Initialize the SiamDiff model with the task configuration and the model
    task_config["model"] = model

    task_config.pop("class")
    task = SiamDiff(**task_config)

    # Load model weights from the checkpoint
    # task.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")), strict=False)

    return task, cfg


def initialize_graph(num_nodes, num_edges):
    # Create random positions for the nodes
    node_position = torch.randn(num_nodes, 3)

    # Random atom types for the nodes (assuming 20 types)
    atom_type = torch.randint(0, 20, (num_nodes,))

    # Node features (aligned with the node_in_dim specified in the model)
    node_feature = torch.zeros((num_nodes, 100))  # Example feature dimension

    # Random edge list (num_edges x 2 tensor representing connections between nodes)
    edge_list = torch.randint(0, num_nodes, (num_edges, 2))

    # Edge features (aligned with the edge_in_dim specified in the model)
    edge_feature = torch.zeros((num_edges, 16))  # Example feature dimension

    # Initialize the graph with the specified attributes
    graph = Graph(
        node_feature=node_feature,
        edge_list=edge_list,
        edge_feature=edge_feature,
        node_position=node_position,
    )
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of steps for generation"
    )
    args, unparsed = parser.parse_known_args()
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument(f"--{var}", default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}
    checkpoint_path = args.checkpoint
    config_path = args.config
    num_steps = args.num_steps

    model, cfg = load_pretrained_model(checkpoint_path, config_path, vars=vars)

    initial_graph = initialize_graph(num_nodes=150, num_edges=300)
    # Extract additional parameters from the configuration
    num_noise_level = cfg["task"]["num_noise_level"]
    sigma_begin = cfg["task"]["sigma_begin"]
    sigma_end = cfg["task"]["sigma_end"]

    # Initialize the DiffusionProteinGenerator with additional parameters
    generator = DiffusionProteinGenerator(
        model, num_noise_level, sigma_begin, sigma_end
    )

    generated_graphs = generator(initial_graph, num_steps)

    for step, graph in enumerate(generated_graphs):
        print(f"Step {step}:")
        print("Generated Sequence:", generator.get_sequence(graph))
        print("Generated Structure:", generator.get_structure(graph))
