"""GPU integration for complete variable bags; uses random weights, no downloads."""
import unittest

import timm
import torch

from models.raptor_v5_radimagenet import COATNET_ARCH, FusedCoAtNet, FusionConfig, RadImageNetEncoder, RaptorClassifier
from scripts.train_fusion_ready import FullBagClassifier


@unittest.skipUnless(torch.cuda.is_available(), 'GPU integration check')
class FullBagBackboneTests(unittest.TestCase):
    def test_all_triplets_backward_and_one_bn_update_per_chunk(self):
        torch.set_num_threads(4)
        torch.manual_seed(9)
        coat = timm.create_model(COATNET_ARCH, pretrained=False, num_classes=0, global_pool='avg')
        backbone = FusedCoAtNet(coat, RadImageNetEncoder(), FusionConfig(gradient_checkpointing=False))
        model = RaptorClassifier(backbone,1024,drop=0.).cuda().train()
        network = FullBagClassifier(model,4,torch.float16)
        images = torch.randn(9,3,384,384,device='cuda',requires_grad=True)
        # Odd K exercises a smaller final chunk; nine slices all receive gradients.
        with torch.autocast('cuda',dtype=torch.float16):
            logits=network(images)
        self.assertEqual(logits.shape,(1,12))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),torch.full_like(logits.float(),.25))
        (loss*128).backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertTrue((images.grad.flatten(1).abs().sum(1)>0).all())
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                self.assertEqual(module.num_batches_tracked.item(),3)
        print('Real full-bag AMP check:',{'loss':float(loss.detach()),
              'triplets_with_gradients':9,'peak_gpu_gib':round(torch.cuda.max_memory_allocated()/2**30,2)},flush=True)


if __name__=='__main__':
    unittest.main()
