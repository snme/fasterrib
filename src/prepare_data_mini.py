import os

from src.prepare_data import prepare_data

dirname = os.path.dirname(__file__)
main_dir = os.path.join(dirname, "../data/ribfrac-challenge/mini/")
images_dir = os.path.join(main_dir, "images/")
labels_dir = os.path.join(main_dir, "labels/")
train_info_path = os.path.join(main_dir, "ribfrac-train-info-all.csv")
out_dir = os.path.join(main_dir, "prepared/")
class_counts_path = os.path.join(main_dir, "class_counts.pt")

n_classes = 6

if __name__ == "__main__":
    prepare_data(
        img_dir=images_dir,
        label_dir=labels_dir,
        info_path=train_info_path,
        out_dir=out_dir,
        class_counts_path=class_counts_path,
    )
