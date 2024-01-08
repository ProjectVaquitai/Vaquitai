import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys, CleaningKeys
from data_juicer.utils.mm_utils import load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

import cv2
import numpy as np

@OPERATORS.register_module('image_blurriness_filter')
@LOADED_IMAGES.register_module('image_blurriness_filter')
class ImageBlurrinessFilter(Filter):
    """Filter to keep samples within normal blurriness
    """

    def __init__(self,
                threshold: int = 500,
                 *args,
                 **kwargs):
        """
        Initialization method.
        
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if CleaningKeys.blurriness in sample:
            return sample

        # there is no image in this sample
        sample[CleaningKeys.blurriness] = ''
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        loaded_image_key = sample[self.image_key]
        images = {}
        
        if context and loaded_image_key in sample[Fields.context]:
            # load from context
            images[loaded_image_key] = sample[
                Fields.context][loaded_image_key]
        else:
            if loaded_image_key not in images:
                # avoid load the same images
                image = load_image(loaded_image_key)
                images[loaded_image_key] = image
                if context:
                    # store the image data into context
                    sample[Fields.context][loaded_image_key] = image
        
        def _variance_of_laplacian(image):
            return cv2.Laplacian(image, cv2.CV_64F).var()
        
        sample[CleaningKeys.blurriness] = _variance_of_laplacian(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY))
        
        if sample[CleaningKeys.blurriness] < self.threshold:
            sample[CleaningKeys.blurriness_label] = True
        else:
            sample[CleaningKeys.blurriness_label] = False
            
        return sample

    def process(self, sample):
        # blurriness = sample[CleaningKeys.blurriness]
        return True
            
