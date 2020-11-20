"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from UNetInferenceAgent import UNetInferenceAgent

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """

    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = dcmlist[0]

  
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """

    # Compute the volume of your hippocampal prediction
    volume_ant = np.sum(pred == 1)
    volume_post = np.sum(pred == 2)
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}

def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype("assets/playfair-display/PlayfairDisplay-Regular.ttf", size=80, encoding = 'unic')
    main_font = ImageFont.truetype("assets/playfair-display/PlayfairDisplay-Regular.ttf", size=40, encoding = 'unic')

    slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4] # is there a better choice?

   
    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text((10, 90),
                        f"Patient ID: {header.PatientID}\n"
                        f"Patient Name: {header.PatientName}\n"
                        f"Series Description: {header.SeriesDescription}\n"
                        f"Series Date: {header.SeriesDate}\n"
                        f"Modality: {header.Modality}\n"
                        f"Anterior Hippocampal Volume: {inference['anterior']}\n"
                        f"Posterior Hippocampal Volume: {inference['posterior']}\n"
                        f"Total Hippocampal Volume: {inference['total']}\n",
                        (255, 255, 255), font=main_font)



    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """


    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

  

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Here we are assuming that path is a directory that contains a full study as a collection
    # of files
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    #dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]
    dicoms = []
    for direc , subdirs, file in os.walk(path):
        for subdir in subdirs:
            dicoms.extend([pydicom.dcmread(os.path.join(path, subdir, f)) for f in os.listdir(os.path.join(path, subdir))])

   

    series_for_inference = [i for i in dicoms if i.SeriesDescription == 'HippoCrop']

    # Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference

def os_command(command):
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], d))]

    # Get the latest directory
    study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

    print(f"Looking for series to run inference on in directory {study_dir}...")

    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=r"/home/workspace/src/model.pth")

    # Run inference
    # single_volume_inference_unpadded takes a volume of arbitrary size 
    # and reshapes y and z dimensions to the patch size used by the model before 
    # running inference. 
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    pred_volumes = get_predicted_volumes(pred_label)

    # Create and save the report
    print("Creating and pushing report...")
    report_save_path = r"/home/workspace/out/report.dcm"
   
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, report_save_path)

    # Send report to our storage archive
    # Write a command line string that will issue a DICOM C-STORE request to send our report
    # to our Orthanc server (that runs on port 4242 of the local machine), using storescu tool
    os_command("storescu localhost 4242 -v -aec HIPPOAI +r +sd /home/workspace/out/report.dcm")

    # This line will remove the study dir if run as root user
    # Sleep to let our StoreSCP server process the report (remember - in our setup
    # the main archive is routing everyting that is sent to it, including our freshly generated
    # report) - we want to give it time to save before cleaning it up
    time.sleep(2)
    shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
