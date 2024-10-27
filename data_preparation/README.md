# Order of execution

1. data_extract.py: Extract the data from the visible human dataset
2. After getting all the single dicom files, run multiframe_dicom to merge them into a single dicom series file
3. Run projection to project them