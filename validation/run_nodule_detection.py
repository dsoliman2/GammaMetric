import monai
from monai.bundle import ConfigParser
from pathlib import Path
import SimpleITK as sitk
import numpy as np

BUNDLE_DIR = r"C:\Users\Dan\Desktop\gammametric_output\monai_models\lung_nodule_ct_detection"
DICOM_DIR = r"C:\Users\Dan\Desktop\gammametric_output\Lung DICOM\manifest-1600709154662\LIDC-IDRI\LIDC-IDRI-0009\01-01-2000-NA-NA-07045\3000538.000000-NA-29210"

# Load inference config
parser = ConfigParser()
parser.read_config(str(Path(BUNDLE_DIR) / "configs" / "inference.json"))
parser.read_meta(str(Path(BUNDLE_DIR) / "configs" / "metadata.json"))

infer = parser.get_parsed_content("inferer", instantiate=True)
print(infer)