import argparse
import csv
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Searches for all files mentioned in the input labels file and copies them to a target directory."
)
parser.add_argument("--labels", type=str, help="path to the labels csv file")
parser.add_argument(
    "--search-dir",
    type=str,
    help="path to the directory where the labeled files should be found",
)
parser.add_argument("--out-dir", type=str, help="path to the target directory")


def main(args):
    search_dir = args.search_dir
    out_dir = args.out_dir
    labels_file = args.labels

    filename_to_path = {}
    for path in Path(search_dir).rglob("*.py"):
        filename_to_path[path.name] = str(path.absolute())

    with open(labels_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            fn = row["filename"]
            if fn not in filename_to_path:
                raise Exception(f"{fn} not found")
            shutil.copy(filename_to_path[fn], out_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
