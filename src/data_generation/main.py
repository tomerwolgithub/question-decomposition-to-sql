from write_grounding import *
import argparse
import os


def main(args):
    examples = load_grounding_examples(args.input_file)
    print(f"Loaded {len(examples)} grounding examples.")
    write_grounding_results(examples, args.output_file, to_json=args.json_steps)
    print("Done grounding all examples.\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="example command: "
                    "python main.py test/grounding_examples.csv "
                    "--json_steps True"
    )
    parser.add_argument('input_file', type=str, help='path to grounding examples csv')
    parser.add_argument('output_file', type=str, help='path to output file, with csv extension')
    parser.add_argument('--json_steps', type=bool, default=None,
                        help='whether to generate grounding steps json')
    args = parser.parse_args()
    assert os.path.exists(args.input_file)

    main(args)

