import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import pydicom.uid
import datetime
import re
import os

def img_data_to_np_array(image_data_bytes):
    img_array = np.frombuffer(image_data_bytes, dtype=np.int16).byteswap()

    expected_size = 512 * 512
    if img_array.size != expected_size:
        raise ValueError(f"Unexpected image data size: {img_array.size}, expected: {expected_size}")

    img_array = img_array.reshape((512, 512))
    return img_array

def extract_image_number(filename):
    match = re.search(r'c_vm(\d+)\.fre', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def sort_slices(input_directory):
    fre_files = [f for f in os.listdir(input_directory) if f.endswith('.fre')]
    fre_files = [f for f in fre_files if extract_image_number(f) is not None]
    fre_files.sort(key=lambda x: extract_image_number(x))
    return fre_files

def file_processing(fre_files, input_directory, output_directory, header_directory, study_instance_uid, series_instance_uid):
    bone_output_directory = os.path.join(output_directory, 'bone')
    soft_tissue_output_directory = os.path.join(output_directory, 'soft_tissue')
    air_output_directory = os.path.join(output_directory, 'air')
    original_output_directory = os.path.join(output_directory, 'original')

    os.makedirs(bone_output_directory, exist_ok=True)
    os.makedirs(soft_tissue_output_directory, exist_ok=True)
    os.makedirs(air_output_directory, exist_ok=True)
    os.makedirs(original_output_directory, exist_ok=True)

    for filename in fre_files:
        file_index = filename[4:8]
        # Concatenate to take only the thigh
        if int(file_index) <= 2028:
            continue
        fre_file_path = os.path.join(input_directory, filename)
        bone_output_directory = os.path.join(output_directory, 'bone')
        soft_tissue_output_directory = os.path.join(output_directory, 'soft_tissue')
        air_output_directory = os.path.join(output_directory, 'air')
        original_output_directory = os.path.join(output_directory, 'original')

        # Corresponding header file path
        header_filename = filename + '.txt'
        header_file_path = os.path.join(header_directory, header_filename)

        with open(fre_file_path, 'rb') as f:
            header_bytes = f.read(3416)  # Header size
            image_data_bytes = f.read()

        params = parse_header_file(header_file_path)
        slice_location = params.get('slice_location')
        img_array = img_data_to_np_array(image_data_bytes)
        img_array = img_array[:390, :]
        hu_values = img_array - 1024

        instance_number = extract_image_number(filename)

        # Create mask
        bone_mask = hu_values > 350
        soft_tissue_mask = (hu_values > -800) & (hu_values <= 350)
        air_mask = hu_values <= -800

        img_array_bone = np.where(bone_mask, img_array, -1024).astype(np.int16)
        img_array_soft_tissue = np.where(soft_tissue_mask, img_array, -1024).astype(np.int16)
        img_array_air = np.where(air_mask, img_array, -1024).astype(np.int16)

        # Save bone image
        bone_output_filename = os.path.splitext(filename)[0] + '_bone.dcm'
        bone_output_dcm_path = os.path.join(bone_output_directory, bone_output_filename)
        create_dicom(img_array_bone, slice_location, bone_output_dcm_path, instance_number, study_instance_uid, series_instance_uid, params)

        # Save soft tissue image
        soft_tissue_output_filename = os.path.splitext(filename)[0] + '_soft_tissue.dcm'
        soft_tissue_output_dcm_path = os.path.join(soft_tissue_output_directory, soft_tissue_output_filename)
        create_dicom(img_array_soft_tissue, slice_location, soft_tissue_output_dcm_path, instance_number, study_instance_uid, series_instance_uid, params)

        # Save air image
        air_output_filename = os.path.splitext(filename)[0] + '_air.dcm'
        air_output_dcm_path = os.path.join(air_output_directory, air_output_filename)
        create_dicom(img_array_air, slice_location, air_output_dcm_path, instance_number, study_instance_uid, series_instance_uid, params)

        # Save original image
        output_filename = os.path.splitext(filename)[0] + '.dcm'
        output_dcm_path = os.path.join(original_output_directory, output_filename)
        create_dicom(img_array, slice_location, output_dcm_path, instance_number, study_instance_uid, series_instance_uid, params)

def get_slice_location(header_file_path):
    with open(header_file_path, 'r') as f:
        for line in f:
            match = re.search(r'Image location.*?: ([\-\d\.]+)', line)
            if match:
                slice_location = float(match.group(1))
                return slice_location
            
    print(f"Slice location not found in {header_file_path}")
    return None

def get_normal_s_coord(header_file_path):
    with open(header_file_path, 'r') as f:
        for line in f:
            match = re.search(r'Normal S coord.*?: ([\-\d\.]+)', line)
            if match:
                slice_location = float(match.group(1))
                return slice_location
    
    print(f"Normal S Coord not found in {header_file_path}")
    return None

def create_dicom(img_array, slice_location, output_dcm_path, instance_number, study_instance_uid, series_instance_uid, params):
    # Create a FileDataset instance (initially empty)
    file_meta = pydicom.Dataset()
    ds = FileDataset(output_dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Set the file meta information
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # Set the image data
    ds.PixelData = img_array.tobytes()
    ds.Rows, ds.Columns = img_array.shape

    # Set pixel representation
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = int(params.get('bits_allocated', 16))
    ds.BitsStored = int(params.get('bits_stored', 16))
    ds.HighBit = int(params.get('high_bit', 15))
    ds.PixelRepresentation = 1

    # Set pixel spacing and slice thickness
    ds.PixelSpacing = [
        str(params.get('pixel_spacing_x', '0.898438')),
        str(params.get('pixel_spacing_y', '0.898438'))
    ]
    ds.SliceThickness = str(params.get('slice_thickness', '3'))

    # Set patient information
    ds.PatientName = params.get('patient_name', 'Unknown')
    ds.PatientID = params.get('patient_id', 'Unknown')
    ds.PatientSex = params.get('patient_sex', 'M')
    ds.PatientAge = params.get('patient_age', '000Y')  # Adjust if known

    # Set study date and time
    ds.StudyDate = params.get('study_date', '')
    ds.StudyTime = params.get('study_time', '')

    # Set study information
    ds.StudyInstanceUID = study_instance_uid
    ds.StudyID = params.get('study_id', '1174')
    ds.AccessionNumber = params.get('accession_number', '1174')
    ds.ReferringPhysicianName = params.get('referring_physician_name', '')

    # Set series information
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = params.get('series_number', '2')
    ds.Modality = params.get('modality', 'CT')

    # Set acquisition date and time
    ds.AcquisitionDate = params.get('acquisition_date', '')
    ds.AcquisitionTime = params.get('acquisition_time', '')

    # Set image information
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.InstanceNumber = str(instance_number)

    # Set image position and orientation
    ds.ImagePositionPatient = [
        str(params.get('center_r', '-2.0')),
        str(params.get('center_a', '61.0')),
        str(slice_location)
    ]

    # Handle ImageOrientationPatient
    normal_r = float(params.get('normal_r', '0.0'))
    normal_a = float(params.get('normal_a', '0.0'))
    normal_s = float(params.get('normal_s', '1.0'))

    # Default orientation for axial images
    ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']

    # Adjust orientation based on Normal S Coord
    if normal_s == -1.0:
        # Invert the second vector
        ds.ImageOrientationPatient = ['1', '0', '0', '0', '-1', '0']

    # Set patient position
    ds.PatientPosition = params.get('patient_position', 'FFS')

    # Set the necessary flags
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Save the DICOM file
    pydicom.dcmwrite(output_dcm_path, ds)

def parse_header_file(header_file_path):
    params = {}
    with open(header_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Extract Image Location
        match = re.search(r'Image location.*?: ([\-\d\.]+)', line)
        if match:
            params['slice_location'] = float(match.group(1))
            continue

        # Extract Normal S Coord
        match = re.search(r'Normal S coord.*?: ([\-\d\.]+)', line)
        if match:
            params['normal_s_coord'] = float(match.group(1))
            continue

        # Extract Pixel Spacing X and Y
        match = re.search(r'Image pixel size - X.*?: ([\d\.]+)', line)
        if match:
            params['pixel_spacing_x'] = float(match.group(1))
            continue

        match = re.search(r'Image pixel size - Y.*?: ([\d\.]+)', line)
        if match:
            params['pixel_spacing_y'] = float(match.group(1))
            continue

        # Extract Slice Thickness
        match = re.search(r'Slice Thickness \(mm\).*?: ([\d\.]+)', line)
        if match:
            params['slice_thickness'] = float(match.group(1))
            continue

        # Extract Image Orientation Patient (Normal vectors)
        match = re.search(r'Normal R coord.*?: ([\-\d\.]+)', line)
        if match:
            params['normal_r'] = float(match.group(1))
            continue

        match = re.search(r'Normal A coord.*?: ([\-\d\.]+)', line)
        if match:
            params['normal_a'] = float(match.group(1))
            continue

        match = re.search(r'Normal S coord.*?: ([\-\d\.]+)', line)
        if match:
            params['normal_s'] = float(match.group(1))
            continue

        # Extract Image Position Patient (Center coordinates)
        match = re.search(r'Center R coord of plane image.*?: ([\-\d\.]+)', line)
        if match:
            params['center_r'] = float(match.group(1))
            continue

        match = re.search(r'Center A coord of plane image.*?: ([\-\d\.]+)', line)
        if match:
            params['center_a'] = float(match.group(1))
            continue

        match = re.search(r'Center S coord of Plane image.*?: ([\-\d\.]+)', line)
        if match:
            params['center_s'] = float(match.group(1))
            continue

        # Extract Patient Position
        match = re.search(r'Patient Position.*?: (\d+)', line)
        if match:
            patient_position_code = int(match.group(1))
            if patient_position_code == 1:
                params['patient_position'] = 'FFS'  # Feet First Supine
            elif patient_position_code == 2:
                params['patient_position'] = 'FFP'  # Feet First Prone
            elif patient_position_code == 3:
                params['patient_position'] = 'HFS'  # Head First Supine
            elif patient_position_code == 4:
                params['patient_position'] = 'HFP'  # Head First Prone
            else:
                params['patient_position'] = 'UNKNOWN'
            continue

        # Extract Bits Allocated
        match = re.search(r'Screen Format \(8/16 bit\).*?: (\d+)', line)
        if match:
            bits_allocated = int(match.group(1))
            params['bits_allocated'] = bits_allocated
            params['bits_stored'] = bits_allocated
            params['high_bit'] = bits_allocated - 1
            continue

        # Extract Modality (if available)
        match = re.search(r'Exam Type.*?: (.*)', line)
        if match:
            exam_type = match.group(1).strip()
            params['modality'] = exam_type
            continue

        # Extract Patient ID and Name
        match = re.search(r'Patient ID for this exam.*?: (.*)', line)
        if match:
            params['patient_id'] = match.group(1).strip()
            continue

        match = re.search(r'Patient Name.*?: (.*)', line)
        if match:
            params['patient_name'] = match.group(1).strip()
            continue

        # Extract Acquisition Date/Time
        match = re.search(r'Actual Image Date/Time stamp.*?: (.*)', line)
        if match:
            date_time_str = match.group(1).strip()
            try:
                date_time_obj = datetime.datetime.strptime(date_time_str, '%a %b  %d %H:%M:%S %Y')
                params['acquisition_date'] = date_time_obj.strftime('%Y%m%d')
                params['acquisition_time'] = date_time_obj.strftime('%H%M%S')
            except ValueError:
                pass
            continue

    return params

input_directory = "../datasets/VisibleHuman/Male/Thigh/VisibleHumanThigh"
output_directory = "./output_dicom"
header_directory = "../datasets/VisibleHuman/Male/Thigh/ThighHeader"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Generate UIDs for the study and series (consistent across all images)
study_instance_uid = pydicom.uid.generate_uid()
series_instance_uid = pydicom.uid.generate_uid()

# Get sorted list of fre files
fre_files = sort_slices(input_directory)

# Process the files
file_processing(fre_files, input_directory, output_directory, header_directory, study_instance_uid, series_instance_uid)
