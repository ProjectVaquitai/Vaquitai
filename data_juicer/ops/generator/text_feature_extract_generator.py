import numpy as np
from PIL import ImageOps, Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys, EmbKeys
from data_juicer.utils.mm_utils import (SpecialTokens, load_image,
                                        remove_special_tokens)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Generator
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'text_feature_extract_generator'


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFeatureExtractGenerator(Generator):
    """Extracting feature vectors from text"""

    def __init__(self,
                 hf_blip='Salesforce/blip-itm-base-coco',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_blip: blip model name on huggingface to compute
            the matching score between image and text.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.model_key = prepare_model(model_type='hf_blip', model_key=hf_blip)
        self.model, self.processor = get_model(self.model_key)
        self.transform = transforms.Compose([
                transforms.Resize([336, 336]),
                transforms.ToTensor()
            ])

    def compute_embedding(self, sample):
        # check if it's computed already
        if EmbKeys.image_embedding in sample:
            return sample

        # there is no image in this sample
        sample[EmbKeys.image_embedding] = []
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # compute image embeddings
        text = self.model.tokenizer(text, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(self.device)
        text_output = self.model.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')     
        text_features = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)  

        sample[EmbKeys.text_embedding] = text_features.cpu().tolist()[0]

        return sample

    def process(self, dataset):
        """
        For doc-level, dataset --> dataset.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is
            open.
        :return: deduplicated dataset and the sampled duplicate pairs.
        """
        # no need to deduplicate because too few samples
        if len(dataset) <= 1:
            return dataset, {}        

        return dataset
        