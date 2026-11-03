"""Synthetic DICOM tests: ordering, pixel decoding, and source UID ownership."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from raptor_data.series_triplets import dicom_reader


class DicomReaderTests(unittest.TestCase):
    def write_slice(self, directory, name, series_uid, depth, value, sop_uid=None):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = sop_uid or generate_uid()
        dataset = FileDataset(str(directory/name), {}, file_meta=meta, preamble=b"\0"*128)
        dataset.SOPClassUID = meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID = series_uid
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [0, 0, depth]
        dataset.InstanceNumber = 100-depth  # deliberately conflicts with physical order
        dataset.PixelSpacing = [0.5, 0.5]
        dataset.Rows = dataset.Columns = 2
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleSlope = 2
        dataset.RescaleIntercept = -1
        dataset.PixelData = np.full((2, 2), value, dtype=np.uint16).tobytes()
        dataset.save_as(directory/name, enforce_file_format=True)

    def test_physically_ordered_series_and_rescaled_pixels(self):
        uid = generate_uid()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/uid
            directory.mkdir()
            self.write_slice(directory, "a.dcm", uid, 2, 3)
            self.write_slice(directory, "b.dcm", uid, 0, 1)
            self.write_slice(directory, "c.dcm", uid, 1, 2)
            order, decode = dicom_reader()
            files, median_spacing = order(directory)
            self.assertEqual([Path(file).name for file, spacing in files], ["b.dcm", "c.dcm", "a.dcm"])
            self.assertEqual(median_spacing, 0.5)
            np.testing.assert_array_equal(decode(files[0][0]), np.ones((2, 2)))

    def test_foreign_series_and_duplicate_acquired_slice_are_rejected(self):
        uid = generate_uid()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/uid
            directory.mkdir()
            self.write_slice(directory, "a.dcm", generate_uid(), 0, 1)
            order, _ = dicom_reader()
            with self.assertRaisesRegex(ValueError, "SeriesInstanceUID"):
                order(directory)
            sop = generate_uid()
            self.write_slice(directory, "a.dcm", uid, 0, 1, sop)
            self.write_slice(directory, "b.dcm", uid, 1, 2, sop)
            with self.assertRaisesRegex(ValueError, "Duplicate acquired"):
                order(directory)


if __name__ == "__main__":
    unittest.main()
