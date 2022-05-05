import argparse
import csv
import os
import dicom2nifti

parser = argparse.ArgumentParser(
    description="Searches for all dcm series mentioned in the input labels file and converts them to .nii.gz files."
)
parser.add_argument("--labels", type=str, help="path to the labels csv file")
parser.add_argument(
    "--search-dir",
    type=str,
    help="path to the directory where the labeled files should be found",
)
parser.add_argument("--out-dir", type=str, help="path to the target directory")


def listdirs(rootdir):
    for it in os.scandir(rootdir):
        if it.is_dir():
            yield (it.name, it.path)
            yield from listdirs(it)
 
rootdir = 'path/to/dir'
listdirs(rootdir)
def main(args):
    search_dir = args.search_dir
    out_dir = args.out_dir
    labels_file = args.labels

    series_to_nii_filename = {}
    with open(labels_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            series = row["series_instance"]
            anon_mrn = row["anon_mrn"]
            anon_accession = row["anon_mrn"]
            series_to_nii_filename[series] = f"{anon_mrn}-{anon_accession}.nii.gz"

    # Scan the search_dir looking for dcm series directories
    for (dirname, path) in listdirs(search_dir):
        print(dirname)
        if dirname in series_to_nii_filename:
            print('converting series:', dirname)
            out_filename = os.path.join(out_dir, series_to_nii_filename[dirname])
            dicom2nifti.dicom_series_to_nifti(path, out_filename , reorient_nifti=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
