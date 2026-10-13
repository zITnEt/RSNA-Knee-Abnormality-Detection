"""Verify the V5 numerical path and six-slot DICOM output end to end."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/".runtime_preprocess"))
sys.path.insert(0, str(ROOT/"scripts"))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
import preprocess_six_slot_raptor as p
from raptor_data.ready_tensors import ReadyTripletDataset


class PreprocessingTests(unittest.TestCase):
    def test_parent_pixel_math_and_half_precision(self):
        rng = np.random.default_rng(42)
        arrays = [rng.uniform(10, 1500, (200, 220)).astype(np.float32) for _ in range(7)]
        low, high = np.percentile(np.concatenate([a.ravel() for a in arrays]), [2, 98])
        planes = []
        for a in arrays:
            scaled = np.clip((a-low)/(high-low+1e-6), 0, 1)
            crop = scaled[30:170, 40:180]
            reference = (cv2.resize(crop, (336, 336), interpolation=cv2.INTER_AREA)*255).astype(np.uint8)
            np.testing.assert_array_equal(p.crop_plane(a, 1, low, high), reference)
            planes.append(reference)
        wins = np.stack([np.stack(planes[i:i+3]) for i in range(5)]).astype(np.float32)/255
        parent = F.interpolate(torch.from_numpy(wins), size=(384,384), mode="bilinear", align_corners=False)
        parent = (parent-p.MEAN)/p.STD
        actual = p.model_ready(planes)
        np.testing.assert_array_equal(actual, parent.numpy().astype(np.float16))
        self.assertLess(float(np.max(np.abs(actual.astype(np.float32)-parent.numpy()))), .002)

    def test_all_six_slots_saved_without_further_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            uid = generate_uid()
            records = []
            for slot, (_, plane, fluid, fs, _) in enumerate(p.SLOTS):
                series = generate_uid()
                directory = base/uid/series
                directory.mkdir(parents=True)
                records.append({"StudyInstanceUID":uid,"SeriesInstanceUID":series,"Anatomical_Plane":plane,"Fluid_Sensitive":fluid if fs else 0,"Fat_Suppression":fs})
                for index in range(9):
                    meta = FileMetaDataset()
                    meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    meta.MediaStorageSOPClassUID = MRImageStorage
                    meta.MediaStorageSOPInstanceUID = generate_uid()
                    file = directory/f"reverse_{8-index}.dcm"
                    d = FileDataset(str(file), {}, file_meta=meta, preamble=b"\0"*128)
                    d.SOPClassUID = MRImageStorage
                    d.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                    d.SeriesInstanceUID = series
                    d.StudyInstanceUID = uid
                    d.SeriesDescription = ("PD FS" if fs else "PD") if fluid else "T1"
                    d.RepetitionTime = 3000 if fluid else 500
                    d.EchoTime = 30 if fluid else 10
                    d.ImageOrientationPatient = [1,0,0,0,1,0]
                    d.ImagePositionPatient = [0,0,index*3]
                    d.InstanceNumber = index+1
                    d.PixelSpacing = [1,1]
                    d.Rows = d.Columns = 150
                    d.SamplesPerPixel = 1
                    d.PhotometricInterpretation = "MONOCHROME1" if slot == 5 else "MONOCHROME2"
                    d.BitsAllocated = d.BitsStored = 16
                    d.HighBit = 15
                    d.PixelRepresentation = 0
                    d.RescaleSlope = 2
                    d.RescaleIntercept = -20
                    pixels = (np.arange(22500).reshape(150,150) % 1000 + index*10).astype(np.uint16)
                    d.PixelData = pixels.tobytes()
                    d.save_as(file, enforce_file_format=True)
            data, audit = p.prepare(uid, records, base)
            self.assertEqual(data['images'].shape, (46,3,384,384))
            self.assertEqual(data['images'].dtype, np.float16)
            self.assertTrue(data['slot_present'].all())
            self.assertEqual(audit['cross_series_windows'], 10)
            self.assertEqual([r['native_index'] for r in audit['source_files'][:8]], list(range(8)))
            output = base/(uid+'.npz')
            p.save_npz(output, data)
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['images'], data['images'])
                self.assertEqual(json.loads(str(saved['audit_json']))['recipe_sha256'], p.RECIPE_SHA)
            pd.DataFrame({'StudyInstanceUID':[uid],'split':['train']}).to_csv(base/'splits.csv',index=False)
            loaded=ReadyTripletDataset([base],k=None)[0]
            np.testing.assert_array_equal(loaded['images'].numpy(),data['images'])
            self.assertEqual(ReadyTripletDataset([base],k=12)[0]['images'].shape,(12,3,384,384))
            self.assertEqual(sum(p.allocate([25]*6, [s[4] for s in p.SLOTS])), 64)


if __name__ == '__main__':
    unittest.main()
