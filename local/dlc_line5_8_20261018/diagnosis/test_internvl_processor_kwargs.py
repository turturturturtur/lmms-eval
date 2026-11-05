"""Exercise installed vLLM's actual processor construction, without a GPU."""
import unittest
from types import SimpleNamespace

from transformers import AutoTokenizer
from vllm.model_executor.models.internvl import InternVLProcessingInfo


class ProcessorKwargsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            '/mnt/nasA/tianleniu/Innovator-Tune/lmms-eval/eval_model_views/internvl35_row07_nothink_20261018',
            trust_remote_code=True, local_files_only=True)

    def info(self, merged=None):
        info = InternVLProcessingInfo.__new__(InternVLProcessingInfo)
        config = SimpleNamespace(vision_config=SimpleNamespace(image_size=448,patch_size=14),
            min_dynamic_patch=1,max_dynamic_patch=12,dynamic_image_size=True,use_thumbnail=True,
            downsample_ratio=0.5)
        info.ctx = SimpleNamespace(get_merged_mm_kwargs=lambda kwargs: {**(merged or {}),**kwargs})
        info.get_hf_config = lambda: config
        info.get_tokenizer = lambda: self.tokenizer
        info.__dict__['ctx_video_token'] = '<|video_pad|>'
        return info

    def test_default_processor_unchanged(self):
        p = self.info().get_hf_processor()
        self.assertEqual(p.image_processor.max_dynamic_patch,12)
        self.assertEqual(p.video_processor.image_size,448)
        self.assertEqual(p.image_seq_length,256)

    def test_image_override_does_not_leak_into_video_constructor(self):
        p = self.info().get_hf_processor(max_dynamic_patch=11)
        self.assertEqual(p.image_processor.max_dynamic_patch,11)
        self.assertEqual(p.video_processor.image_size,448)
        self.assertLess(p.get_num_image_tokens(image_width=1792,image_height=1344),
            self.info().get_hf_processor().get_num_image_tokens(image_width=1792,image_height=1344))

    def test_merged_image_override(self):
        p = self.info({'max_dynamic_patch':10,'min_dynamic_patch':1,
                       'dynamic_image_size':True,'use_thumbnail':True}).get_hf_processor()
        self.assertEqual(p.image_processor.max_dynamic_patch,10)
        self.assertEqual(p.video_processor.image_size,448)

    def test_unknown_keyword_still_rejected(self):
        with self.assertRaises(TypeError):
            self.info().get_video_processor(unknown_parameter=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
