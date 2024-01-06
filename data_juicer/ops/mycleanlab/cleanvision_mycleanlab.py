import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys, CleaningKeys
from data_juicer.utils.mm_utils import load_image
from datasets import load_dataset,Dataset

from ..base_op import OPERATORS, Mycleanlab
from ..op_fusion import LOADED_IMAGES

import numpy as np
from cleanvision import Imagelab
from PIL import Image

@OPERATORS.register_module('cleanvision_mycleanlab')
@LOADED_IMAGES.register_module('cleanvision_mycleanlab')
class CleanvisionMycleanlab(Mycleanlab):
    """Filter to keep samples within normal blurriness
    """

    def __init__(self,
                 *args,
                 **kwargs):
        """
        Initialization method.
        
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)


    
    def process(self, dataset):
        # def _open_map_helper(sample):
        #     sample[self.image_key] = Image.open(sample[self.image_key])
        #     # print(Image.open(sample[self.image_key]))
        #     return sample
        
        # dataset_hf = dataset.map(_open_map_helper)
        my_dict = {"image": [Image.open(_) for _ in dataset[self.image_key]], "image_path": dataset[self.image_key]}
        hf_dataset = Dataset.from_dict(my_dict)
        imagelab = Imagelab(hf_dataset=hf_dataset, image_key=self.image_key)
        imagelab.find_issues()
        return dataset
            
