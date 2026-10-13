"""Exercise physical DICOM sampling, direct resize and series ownership."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime_preprocess'));sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,MRImageStorage,generate_uid
import preprocess_series_safe_384 as p
import pandas as pd
from raptor_data.series_safe_ready import SeriesSafe384Dataset


class Safe384Tests(unittest.TestCase):
    def test_direct_384_with_independent_spacing(self):
        array=(np.arange(240*220).reshape(240,220)%997).astype(np.float32)
        expected=np.clip((array-20)/(970-20+1e-6),0,1)[50:190,75:145]
        expected=(cv2.resize(expected,(384,384),interpolation=cv2.INTER_AREA)*255).astype(np.uint8)
        np.testing.assert_array_equal(p.crop_plane(array,[1.,2.],20,970),expected)
        self.assertEqual([s[4] for s in p.SLOTS],[18,14,12,8,12,8])
        np.testing.assert_array_equal(p.sample_indices(100,18),np.linspace(2,97,18).round().astype(int))
        self.assertEqual(len(p.sample_indices(3,18)),2)

    def test_six_series_72_slices_60_triplets_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);uid=generate_uid();rows=[]
            raw=[]
            for slot,(_,plane,fluid,fs,allocation) in enumerate(p.SLOTS):
                series=generate_uid();folder=root/uid/series;folder.mkdir(parents=True)
                rows.append(dict(StudyInstanceUID=uid,SeriesInstanceUID=series,Anatomical_Plane=plane,
                                 Fluid_Sensitive=fluid if fs else 0,Fat_Suppression=fs))
                slot_raw=[]
                for i in range(24):
                    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
                    meta.MediaStorageSOPClassUID=MRImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
                    file=folder/f'reverse_{23-i:03d}.dcm'
                    d=FileDataset(str(file),{},file_meta=meta,preamble=b'\0'*128)
                    d.SOPClassUID=MRImageStorage;d.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
                    d.SeriesInstanceUID=series;d.StudyInstanceUID=uid
                    d.SeriesDescription=('PD FS' if fs else 'PD') if fluid else 'T1'
                    d.RepetitionTime=3000 if fluid else 500;d.EchoTime=30 if fluid else 10
                    d.ImageOrientationPatient=[1,0,0,0,1,0];d.ImagePositionPatient=[0,0,i*3]
                    d.InstanceNumber=24-i;d.PixelSpacing=[1,2];d.Rows=180;d.Columns=180
                    d.SamplesPerPixel=1;d.PhotometricInterpretation='MONOCHROME1' if slot==5 else 'MONOCHROME2'
                    d.BitsAllocated=d.BitsStored=16;d.HighBit=15;d.PixelRepresentation=0
                    d.RescaleSlope=2;d.RescaleIntercept=-20
                    pixels=(np.arange(180*180).reshape(180,180)%1000+i*10).astype(np.uint16)
                    d.PixelData=pixels.tobytes();d.save_as(file,enforce_file_format=True)
                    a=pixels.astype(np.float32)*2-20
                    if slot==5:a=a.max()-a
                    slot_raw.append(a)
                raw.append(slot_raw)
            data,audit=p.prepare(uid,rows,root)
            self.assertEqual(data['slices_uint8'].shape,(72,384,384))
            self.assertEqual(data['images'].shape,(60,3,384,384))
            self.assertTrue(data['slot_present'].all());self.assertTrue(data['same_series'].all())
            self.assertEqual(audit['cross_series_windows'],0)
            for i,ref in enumerate(audit['intensity']):
                chosen=p.sample_indices(24,p.SLOTS[i][4]);low,high=np.percentile(np.concatenate([raw[i][j].ravel() for j in chosen]),[2,98])
                self.assertEqual(ref['reference_indices'],chosen.tolist())
                self.assertAlmostEqual(ref['p2'],low);self.assertAlmostEqual(ref['p98'],high)
            self.assertEqual([s['native_index'] for s in audit['source_files'][:18]],p.sample_indices(24,18).tolist())
            indices=data['triplet_indices'];sources=data['slice_series_uids'][indices]
            self.assertTrue(np.all(sources==sources[:,:1]))
            expected=(data['slices_uint8'][indices].astype(np.float32)/255-p.MEAN)/p.STD
            np.testing.assert_array_equal(data['images'],expected.astype(np.float16))
            file=root/'saved.npz';p.save_npz(file,data)
            with np.load(file,allow_pickle=False) as saved:p.validate_arrays(saved)
            # A boundary-crossing edit must be rejected, not merely marked.
            bad=dict(data);bad['triplet_indices']=indices.copy();bad['triplet_indices'][0,2]=18
            with self.assertRaisesRegex(ValueError,'CROSS-SERIES'):p.validate_arrays(bad)
            # The ready-input loader must preserve pixel values and exclude Gold.
            file.replace(root/(uid+'.npz'));gold_uid=generate_uid()
            p.save_npz(root/(gold_uid+'.npz'),data)
            (root/'recipe.json').write_text(json.dumps({**p.RECIPE,'sha256':p.RECIPE_SHA}))
            pd.DataFrame({'StudyInstanceUID':[uid,gold_uid],'split':['train','held_out_gold']}).to_csv(root/'splits.csv',index=False)
            labels=pd.DataFrame([dict(StudyInstanceUID=uid,**{label:.25 for label in p.LABELS})])
            labels.to_csv(root/'labels_hybrid_train.csv',index=False)
            train=SeriesSafe384Dataset([root],k=None)
            self.assertEqual(len(train),1);item=train[0]
            self.assertEqual(item['study_uid'],uid)
            np.testing.assert_array_equal(item['images'].numpy(),data['images'])
            np.testing.assert_array_equal(item['targets'].numpy(),np.full(12,.25,np.float32))
            gold=SeriesSafe384Dataset([root],split='held_out_gold',labels=False)
            self.assertEqual(len(gold),1);self.assertEqual(gold[0]['study_uid'],gold_uid)
            with self.assertRaisesRegex(ValueError,'Gold labels'):SeriesSafe384Dataset([root],split='held_out_gold')
            leaked=pd.concat([labels,labels.assign(StudyInstanceUID=gold_uid)],ignore_index=True)
            leaked.to_csv(root/'labels_hybrid_train.csv',index=False)
            with self.assertRaisesRegex(ValueError,'HYBRID'):SeriesSafe384Dataset([root])


if __name__=='__main__':unittest.main()
