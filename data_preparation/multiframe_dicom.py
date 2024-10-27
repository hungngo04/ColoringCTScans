import os
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
from pydicom import Sequence
import numpy as np
import re

def read_dicom_files(folder_path):
    dicom_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.dcm'):
            file_path = os.path.join(folder_path, filename)
            try:
                dcm = pydicom.dcmread(file_path)
                # Extract the number from the filename
                match = re.search(r'\d+', filename)
                if match:
                    slice_number = int(match.group())
                else:
                    slice_number = float('inf')  # If no number found, put at the end
                dicom_files.append((slice_number, dcm))
            except pydicom.errors.InvalidDicomError:
                print(f"Skipping invalid DICOM file: {filename}")
    
    # Sort the DICOM files based on the extracted number
    return [dcm for _, dcm in sorted(dicom_files, key=lambda x: x[0])]

def create_multiframe_dicom(dicom_files, output_path):
    # Get the first DICOM file as a reference
    ref_dicom = dicom_files[0]
    
    # Combine pixel data from all files
    pixel_array = np.stack([dcm.pixel_array for dcm in dicom_files])
    
    # Create a new file meta dataset
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2.1'  # Enhanced CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    # Create a new DICOM dataset
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Copy metadata from the reference DICOM
    for elem in ref_dicom.iterall():
        if elem.tag != (0x7FE0, 0x0010):  # Skip pixel data
            ds.add(elem)
    
    # Update necessary attributes for multi-frame
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2.1'
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    
    # Set multi-frame specific attributes
    ds.NumberOfFrames = len(dicom_files)
    ds.PixelData = pixel_array.tobytes()
    
    # Update dimension organization
    ds.DimensionOrganizationType = 'TILED_FULL'
    
    # Create shared functional groups sequence
    shared_func_groups_seq = pydicom.Dataset()
    
    # Pixel Measures Sequence
    pixel_measures_seq = pydicom.Dataset()
    pixel_measures_seq.PixelSpacing = getattr(ref_dicom, 'PixelSpacing', ['1', '1'])
    pixel_measures_seq.SliceThickness = getattr(ref_dicom, 'SliceThickness', '1')
    shared_func_groups_seq.PixelMeasuresSequence = [pixel_measures_seq]
    
    # Plane Orientation Sequence
    plane_orientation_seq = pydicom.Dataset()
    plane_orientation_seq.ImageOrientationPatient = getattr(ref_dicom, 'ImageOrientationPatient', ['1', '0', '0', '0', '1', '0'])
    shared_func_groups_seq.PlaneOrientationSequence = [plane_orientation_seq]
    
    # Pixel Value Transformation Sequence
    pixel_value_transformation_seq = pydicom.Dataset()
    pixel_value_transformation_seq.RescaleIntercept = getattr(ref_dicom, 'RescaleIntercept', '0')
    pixel_value_transformation_seq.RescaleSlope = getattr(ref_dicom, 'RescaleSlope', '1')
    pixel_value_transformation_seq.RescaleType = "HU"
    shared_func_groups_seq.PixelValueTransformationSequence = [pixel_value_transformation_seq]
    
    # CT Image Frame Type Sequence
    ct_image_frame_type_seq = pydicom.Dataset()
    ct_image_frame_type_seq.FrameType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
    shared_func_groups_seq.CTImageFrameTypeSequence = [ct_image_frame_type_seq]
    
    # Image Frame Type Sequence
    image_frame_type_seq = pydicom.Dataset()
    image_frame_type_seq.FrameType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
    shared_func_groups_seq.ImageFrameTypeSequence = [image_frame_type_seq]
    
    ds.SharedFunctionalGroupsSequence = [shared_func_groups_seq]
    
    # Create per-frame functional groups sequence
    ds.PerFrameFunctionalGroupsSequence = []
    for i, dcm in enumerate(dicom_files):
        frame_func_group = pydicom.Dataset()
        
        # Frame Content Sequence
        frame_content_seq = pydicom.Dataset()
        frame_content_seq.FrameAcquisitionNumber = i + 1
        frame_content_seq.FrameReferenceDateTime = getattr(dcm, 'AcquisitionDateTime', '')
        frame_content_seq.FrameAcquisitionDuration = getattr(dcm, 'ExposureTime', 0)
        frame_func_group.FrameContentSequence = [frame_content_seq]
        
        # Plane Position Sequence
        plane_position_seq = pydicom.Dataset()
        plane_position_seq.ImagePositionPatient = getattr(dcm, 'ImagePositionPatient', ['0', '0', str(i)])
        frame_func_group.PlanePositionSequence = [plane_position_seq]
        
        ds.PerFrameFunctionalGroupsSequence.append(frame_func_group)
    
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # Save the multi-frame DICOM
    ds.save_as(output_path)

dicom_types = ['original', 'bone', 'air', 'soft_tissue']
        
for dicom_type in dicom_types:
    input_folder = f'./output_dicom/{dicom_type}'
    output_file = f'./output_dicom_series/multiframe_{dicom_type}.dcm'
    dicom_files = read_dicom_files(input_folder)
    create_multiframe_dicom(dicom_files, output_file)
