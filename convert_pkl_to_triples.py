import pickle
import json
import os
import argparse
import networkx as nx

def convert_pickle_to_triples_json(input_path, output_path):
    """
    Directly convert a pickle file containing a NetworkX DiGraph
    to the final knowledge graph JSON format: {"triples": [[source, relation, target], ...]}

    Args:
        input_path (str): Path to input .pkl file.
        output_path (str): Path to output .json file OR directory.
                           If directory, output file name is derived from input name.
    """
    # Load the NetworkX graph from pickle
    with open(input_path, 'rb') as f:
        G = pickle.load(f)

    if not isinstance(G, nx.DiGraph):
        raise TypeError("Pickle file must contain a networkx.DiGraph object")

    # Extract triples from edges
    triples = []
    for u, v, attrs in G.edges(data=True):
        # The relation is typically stored in an edge attribute named 'relation'
        relation = attrs.get('relation')
        if relation is None:
            # If no 'relation' attribute, skip or use a default?
            # Here we skip edges without a relation.
            continue
        # Convert nodes and relation to string to ensure JSON serializability
        triples.append([str(u), str(relation), str(v)])

    # Prepare output data
    output_data = {"triples": triples}

    # Determine actual output file path
    if os.path.isdir(output_path):
        base_name = os.path.basename(input_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_path, f"{name_without_ext}.json")
    else:
        output_file = output_path
        # Ensure the output file has .json extension (optional)
        if not output_file.endswith('.json'):
            output_file += '.json'

    # Write JSON output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(triples)} triples from {input_path}")
    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a pickle file (NetworkX DiGraph) to knowledge graph triples JSON."
    )
    parser.add_argument("input", help="Path to input .pkl file")
    parser.add_argument("output", help="Path to output .json file or output directory")
    args = parser.parse_args()

    convert_pickle_to_triples_json(args.input, args.output)

if __name__ == "__main__":
    main()