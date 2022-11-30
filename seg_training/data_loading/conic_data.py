import requests
import glob
from zipfile import ZipFile
import cv2
import warnings
from seg_training.data_loading.data_getter_utils.utils import *
from seg_training.data_loading.data_getter_utils.patch_extractor import PatchExtractor
import pandas as pd
import scipy.io as sio
import tqdm
import tifffile as tiff
from torch.utils.data import Dataset, dataset

class ConicData(Dataset):
    classes = ['neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def validation_labels(self):
        warnings.warn("validation_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def validation_data(self):
        warnings.warn("validation_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, download: bool = True, from_ome_tiff: bool = False):
        super(ConicData, self).__init__()
        self.img_dir = "../data/OME-TIFFs/"
        self.np_dir = "../data/patches/"
        self.labels = pd.read_csv("../data/patches/patch_info.csv")
        self.count = pd.read_csv("../data/patches/counts.csv")
        self.from_ome_tiff = from_ome_tiff
        if download:
            self.full_download()
        if not self._check_exists:
            raise RuntimeError("Dataset not found!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> dataset.T_co:
        if self.from_ome_tiff:
            img_path = os.path.join(self.img_dir, self.labels + ".ome.tiff")
            image = np.array(img_path)[:, :, 0:4]
            label = np.array(img_path)[:, :, 4]
            return image, label
        else:
            arr_path = os.path.join(self.np_dir, "images.npy")
            image = np.load(arr_path)[index, :, : 0:4]
            label = np.load(arr_path)[index, :, :, 4]
            return image, label

    def full_download(self, download = True, unzip = True, create_patch = True, ome_tiff = True):
        """Method for downloading, unzipping, patching and creating segmentation masked ome.tiff

            download: Whether files should be downloaded
            unzip: Whether files should be unzipped, necessitates download once before
            create_patch: Whether creates patches from raw images, necessitates unzip once before
            ome_tiff: Whether generates ome.tiff with segmentations masks, necessitates create_patch once before
            """
        if download:
            self._download_files()
        if unzip:
            for zip_file in glob.glob("../data/download/img*"):
                self._unzip_files(zip_file)
                os.remove(zip_file)
            for file in glob.glob("../data/Lizard_I*/*"):
                file_name = file.split("/")[-1]
                shutil.copyfile(file, os.path.join("../data/images/", file_name))
                os.remove(file)
        if create_patch:
            self._create_patches()
        if ome_tiff:
            self._numpy_to_ome_tiff()

    @staticmethod
    def _check_exists():
        if len(glob.glob("../data/patches/*")) > 0:
            return True
        else:
            return False

    @staticmethod
    def _download_files():
        download = "../data/download/"
        """Method to download the raw Lizard Dataset"""
        image_region1 = requests.get("https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_images1.zip")
        print("Image region 1 has been downloaded")
        image_region2 = requests.get("https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_images2.zip")
        print("Image region 2 has been downloaded")
        # overlays = requests.get("https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/overlay.zip")
        # print("Overlays have been downloaded")
        labels = requests.get("https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_labels.zip")
        print("Labels have been downloaded")
        open(download + "img1.zip", "wb").write(image_region1.content)
        open(download + "img2.zip", "wb").write(image_region2.content)
        # open(download + "over.zip", "wb").write(overlays.content)
        open(download + "img_labels.zip", "wb").write(labels.content)

    @staticmethod
    def _unzip_files(zip_file):
        print(zip_file)
        """General Method to unzip folders"""
        with ZipFile(zip_file) as zfile:
            zfile.extractall("../data/")
            zfile.close()

    @staticmethod
    def _create_patches(step_size: int = 256):
        """Patching taken from https://github.com/TissueImageAnalytics/CoNIC/blob/main/extract_patches.py

        step_size:  range of overlaps, decrease this to have a larger overlap between patches
        img_dir: should not be changed, where the finished images are
        """
        win_size = 256  # should keep this the same!
        step_size = step_size  # decrease this to have a larger overlap between patches
        extract_type = "valid"
        img_dir = "../data/images/"
        ann_dir = "../data/Lizard_Labels/Labels/"
        out_dir = "../data/patches/"

        rm_n_mkdir(out_dir)

        xtractor = PatchExtractor(win_size, step_size)

        file_path_list = recur_find_ext(img_dir, ".png")

        pbar_format = (
            "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        )
        pbar = tqdm.tqdm(
            total=len(file_path_list), bar_format=pbar_format, ascii=True, position=0
        )

        img_list = []
        inst_map_list = []
        class_map_list = []
        nuclei_counts_list = []
        patch_names_list = []
        for file_idx, file_path in enumerate(file_path_list):
            basename = pathlib.Path(file_path).stem

            img = cv2.imread(img_dir + basename + ".png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ##
            ann_load = sio.loadmat(ann_dir + basename + ".mat")
            ann_inst = ann_load["inst_map"]
            inst_class = np.squeeze(ann_load["class"]).tolist()
            inst_id = np.squeeze(ann_load["id"]).tolist()
            ann_class = np.full(ann_inst.shape[:2], 0, dtype=np.uint8)
            for val in inst_id:
                ann_inst_tmp = ann_inst == val
                idx_tmp = inst_id.index(val)
                ann_class[ann_inst_tmp] = inst_class[idx_tmp]

            ann = np.dstack([ann_inst, ann_class])

            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            for idx, patch in enumerate(sub_patches):
                patch_img = patch[..., :3]  # RGB image
                patch_inst = patch[..., 3]  # instance map
                patch_class = patch[..., 4]  # class map

                # ensure nuclei range from 0 to N (N is the number of nuclei in the patch)
                patch_inst = remap_label(patch_inst)

                # only consider nuclei for counting if it exists within the central 224x224 region
                patch_inst_crop = cropping_center(patch_inst, [224, 224])
                patch_class_crop = cropping_center(patch_class, [224, 224])
                nuclei_counts_perclass = []
                # get the counts per class
                for nuc_val in range(1, 7):
                    patch_class_crop_tmp = patch_class_crop == nuc_val
                    patch_inst_crop_tmp = patch_inst_crop * patch_class_crop_tmp
                    nr_nuclei = len(np.unique(patch_inst_crop_tmp).tolist()[1:])
                    nuclei_counts_perclass.append(nr_nuclei)

                img_list.append(patch_img)
                inst_map_list.append(patch_inst)
                class_map_list.append(patch_class)
                nuclei_counts_list.append(nuclei_counts_perclass)
                patch_names_list.append("%s-%04d" % (basename, idx))

                assert patch.shape[0] == win_size
                assert patch.shape[1] == win_size

            pbar.update()
        pbar.close()

        # convert to numpy array
        img_array = np.array(img_list).astype("uint8")
        inst_map_array = np.array(inst_map_list).astype("uint16")
        class_map_array = np.array(class_map_list).astype("uint16")
        nuclei_counts_array = np.array(nuclei_counts_list).astype("uint16")

        # combine instance map and classification map to form single array
        inst_map_array = np.expand_dims(inst_map_array, -1)
        class_map_array = np.expand_dims(class_map_array, -1)
        labels_array = np.concatenate((inst_map_array, class_map_array), axis=-1)

        # convert to pandas dataframe
        nuclei_counts_df = pd.DataFrame(
            data={
                "neutrophil": nuclei_counts_array[:, 0],
                "epithelial": nuclei_counts_array[:, 1],
                "lymphocyte": nuclei_counts_array[:, 2],
                "plasma": nuclei_counts_array[:, 3],
                "eosinophil": nuclei_counts_array[:, 4],
                "connective": nuclei_counts_array[:, 5],
            }
        )
        patch_names_df = pd.DataFrame(data={"patch_info": patch_names_list})

        # save output
        np.save(out_dir + "images.npy", img_array)
        np.save(out_dir + "labels.npy", labels_array)
        nuclei_counts_df.to_csv(out_dir + "counts.csv", index=False)
        patch_names_df.to_csv(out_dir + "patch_info.csv", index=False)

    @staticmethod
    def _numpy_to_ome_tiff():
        data_folder = "../data/patches/"
        labels = np.load(data_folder + "labels.npy")
        images = np.load(data_folder + "images.npy")
        segmentations = labels[:, :, :, 0]
        classifications = labels[:, :, :, 1]
        info = pd.read_csv("../data/patches/patch_info.csv")
        for ids in range(len(images)):
            image = images[ids]
            classification = classifications[ids]
            full_image = np.zeros((256, 256, 4))
            full_image[:, :, 0] = image[:, :, 0]
            full_image[:, :, 1] = image[:, :, 1]
            full_image[:, :, 2] = image[:, :, 2]
            full_image[:, :, 3] = classification[:, :]

            full_image = np.transpose(full_image, (2, 0, 1))
            with tiff.TiffWriter(os.path.join("../data/OME-TIFFs/", info.iloc[ids,0] + ".ome.tif"),
                                 bigtiff=True) as tif_file:
                tif_file.write(full_image, photometric="rgb")
