import deepdrr
from spherical_fib import generate_sf_points, spherical_to_carm_angles
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from pathlib import Path
import numpy as np
from typing import List
import re

dicom_types = ['original', 'bone', 'air', 'soft_tissue']
output_folder = "projection_output"

def normalize_images(images: List[np.ndarray]) -> List[np.ndarray]:
    max_value = max(image.max() for image in images)

    normalized_images = []
    for image in images:
        float_image = image.astype(np.float32)
        normalized = (float_image / max_value) * 255.0
        normalized_uint8 = np.clip(normalized, 0, 255).astype(np.uint8)
        normalized_images.append(normalized_uint8)

    return normalized_images

def get_output_dir(folder_name, dicom_type) -> Path:
    output_dir = Path.cwd() / output_folder / folder_name / dicom_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def project(input_dicoms: List[str], output_filenames: List[str], folder_name, alpha, beta):
    projected_images = []

    for input_dicom, dicom_type in zip(input_dicoms, dicom_types):
        dicom_path = Path(input_dicom).resolve()

        patient = deepdrr.Volume.from_dicom(
            path=dicom_path,
            use_cached=True,
            cache_dir=dicom_path.parent / "cache",
            use_thresholding=True,
        )

        # Project in the AP view for each DICOM type
        center = patient.center_in_world
        carm = deepdrr.MobileCArm(
            isocenter=center,
            degrees=True,
            sensor_width=3000, 
            sensor_height=3000,
            source_to_detector_distance = 510,
        )

        with Projector(patient, carm=carm, add_noise=True, photon_count=1000) as projector:
            carm.move_to(alpha=alpha, beta=beta)
            image = projector()
            projected_images.append(image)

    normalized_images = normalize_images(projected_images)
    # normalized_images = projected_images

    for img, dicom_type, filename in zip(normalized_images, dicom_types, output_filenames):
        output_dir = get_output_dir(folder_name, dicom_type)
        path = output_dir / filename
        image_utils.save(path, img)

if __name__ == "__main__":
    folder_names = ['10_angles', '100_angles', '1000_angles', '10000_angles', '1000000_angles']

    for folder_name in folder_names:
        input_dicoms = [f'./output_dicom_series/multiframe_{dicom_type}.dcm' for dicom_type in dicom_types]

        num_angles = int(re.findall(r'\d+', folder_name)[0])
        points = generate_sf_points(num_angles)
        
        for idx in range(len(points)):
            alpha, beta = spherical_to_carm_angles(points[idx])
            output_filenames = [f'projected_{dicom_type}_{idx}.png' for dicom_type in dicom_types]
            project(input_dicoms, output_filenames, folder_name, alpha, beta)
            print(f"Projected volume with index {idx}\n")