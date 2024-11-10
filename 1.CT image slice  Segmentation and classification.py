import numpy as np
import cv2
import os
import nibabel as nib
import pylidc as pl
from pylidc.utils import consensus

dataset_path = r"C:\Users\jiaoj\Desktop\analysis\VOIs\VOIs\image"
save_dir = r"C:\Users\jiaoj\Desktop\train - image"

def normalize_hu(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return np.clip(image, 0, 1)

def process_scan(dicom_name, vol, patient_id):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    if scan is None:
        print(f"Scan record for patient ID: {patient_id} not found")
        return None, None

    nods = scan.cluster_annotations()
    if not nods:
        print(f"No annotation data found for patient ID: {patient_id}")
        return None, None

    anns = nods[0]
    malignancy = anns[0].Malignancy if anns else None
    return malignancy, nods

def save_slices(image, label1, index):
    train_dirs = ['z', 'x', 'y']
    for directory in train_dirs:
        os.makedirs(os.path.join(save_dir, directory), exist_ok=True)

    with open(os.path.join(save_dir, 'label.txt'), 'a') as txtfile:
        txtfile.write(f"{index}.jpg {label1}\n")

    z_slice = image[:, :, image.shape[2] // 2]
    x_slice = image[image.shape[0] // 2, :, :]
    y_slice = image[:, image.shape[1] // 2, :]

    for direction, slice_data in zip(['z', 'x', 'y'], [z_slice, x_slice, y_slice]):
        resized_slice = cv2.resize(slice_data * 255, (50, 50)).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, direction, f"{index}.jpg"), resized_slice)

def main():
    ii = 0
    all_ids = pl.query(pl.Scan.patient_id).distinct().all()
    print("Patient IDs in the database:", [id[0] for id in all_ids])  # Print all patient IDs

    for dicom_name in os.listdir(dataset_path):
        if dicom_name.endswith('.nii.gz'):
            print(f"Processing file: {dicom_name}")
            path_dicom = os.path.join(dataset_path, dicom_name)
            img = nib.load(path_dicom)
            vol = img.get_fdata()

            patient_id = dicom_name.split('_')[0]  # Extract patient ID
            malignancy, nods = process_scan(dicom_name, vol, patient_id)

            if malignancy is not None:
                label1 = {
                    'Highly Unlikely': 1,
                    'Moderately Unlikely': 2,
                    'Indeterminate': 3,
                    'Moderately Suspicious': 4,
                    'Highly Suspicious': 5
                }.get(malignancy, 0)
                print(f"Malignancy: {malignancy}, Label: {label1}")

                normalized_image = normalize_hu(vol)
                save_slices(normalized_image, label1, ii)
                ii += 1
                print(f"Processed: {ii} files")

if __name__ == "__main__":
    main()
