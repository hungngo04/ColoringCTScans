import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import pydicom.uid
import datetime
import re
import os

def img_data_to_np_array(image_data_bytes):
    img_array = np.frombuffer(image_data_bytes, dtype=np.uint16)

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
    for filename in fre_files:
        fre_file_path = os.path.join(input_directory, filename)
        # Corresponding header file path
        header_filename = filename + '.txt'  # Assuming the header files are named 'c_vm1734.fre.txt'
        header_file_path = os.path.join(header_directory, header_filename)

        # Read the .fre file
        with open(fre_file_path, 'rb') as f:
            header_bytes = f.read(3416)  # Header size
            image_data_bytes = f.read()

        # Read the header file to get the slice location
        slice_location = get_slice_location(header_file_path)
        print(slice_location)
        if slice_location is None:
            print(f"Slice location not found for file {header_filename}")
            continue  # Skip this file or handle accordingly

        # Convert image data to numpy array
        img_array = img_data_to_np_array(image_data_bytes).byteswap()

        # Output DICOM file path
        output_filename = os.path.splitext(filename)[0] + '.dcm'
        output_dcm_path = os.path.join(output_directory, output_filename)

        # Extract image number for InstanceNumber
        instance_number = extract_image_number(filename)

        # Create the DICOM file
        create_dicom(img_array, slice_location, output_dcm_path, instance_number, study_instance_uid, series_instance_uid)

def get_slice_location(header_file_path):
    with open(header_file_path, 'r') as f:
        for line in f:
            match = re.search(r'Image location.*?: ([\-\d\.]+)', line)
            if match:
                slice_location = float(match.group(1))
                return slice_location
    # If not found, return None or raise an error
    print(f"Slice location not found in {header_file_path}")
    return None

def create_dicom(img_array, slice_location, output_dcm_path, instance_number, study_instance_uid, series_instance_uid):
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
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned integer

    # Set pixel spacing and slice thickness
    ds.PixelSpacing = [str(0.898438), str(0.898438)]
    ds.SliceThickness = str(3)

    # Set patient information
    ds.PatientName = 'huntsville'
    ds.PatientID = '1109pm 8/5/9'
    ds.PatientSex = 'M'  # 'M' for Male
    ds.PatientAge = '000Y'  # Age not specified; adjust if known

    # Parse and set study date and time
    exam_datetime_str = 'Thu Aug  5 19:10:38 1993'
    exam_datetime = datetime.datetime.strptime(exam_datetime_str, '%a %b %d %H:%M:%S %Y')
    ds.StudyDate = exam_datetime.strftime('%Y%m%d')
    ds.StudyTime = exam_datetime.strftime('%H%M%S')

    # Set study information
    ds.StudyInstanceUID = study_instance_uid
    ds.StudyID = '1174'
    ds.AccessionNumber = '1174'  # Use Exam Number if applicable
    ds.ReferringPhysicianName = ''

    # Set series information
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = '2'
    ds.Modality = 'CT'

    # Parse and set acquisition date and time
    image_datetime_str = 'Thu Aug  5 20:27:13 1993'
    image_datetime = datetime.datetime.strptime(image_datetime_str, '%a %b %d %H:%M:%S %Y')
    ds.AcquisitionDate = image_datetime.strftime('%Y%m%d')
    ds.AcquisitionTime = image_datetime.strftime('%H%M%S')

    # Set image information
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.InstanceNumber = str(instance_number)

    # Set image position and orientation
    ds.ImagePositionPatient = ['-2', '61', str(slice_location)]
    ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']  # Assuming standard orientation

    # Set the necessary flags
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Save the DICOM file
    pydicom.dcmwrite(output_dcm_path, ds)

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
