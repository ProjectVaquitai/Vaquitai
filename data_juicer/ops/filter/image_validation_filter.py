import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys, CleaningKeys
from data_juicer.utils.mm_utils import load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

import cv2
import numpy as np

@OPERATORS.register_module('image_validation_filter')
@LOADED_IMAGES.register_module('image_validation_filter')
class ImageValidationFilter(Filter):
    """Filter to keep samples within normal validation
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
        

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if CleaningKeys.validation in sample:
            return sample

        # there is no image in this sample
        sample[CleaningKeys.validation] = True
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        loaded_image_key = sample[self.image_key]
        sample[CleaningKeys.validation] = False

        try:
            image = load_image(loaded_image_key)
        except:
            sample[CleaningKeys.validation] = True
            
        return sample

    def process(self, sample):
        validation = sample[CleaningKeys.validation]
        return not validation
            
