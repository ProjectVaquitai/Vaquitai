import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys, CleaningKeys
from data_juicer.utils.mm_utils import load_image

from ...base_op import OPERATORS, Filter
from ...op_fusion import LOADED_IMAGES

import cv2
import numpy as np

@OPERATORS.register_module('image_brightness_filter')
@LOADED_IMAGES.register_module('image_brightness_filter')
class ImageBrightnessFilter(Filter):
    """Filter to keep samples within normal brightness
    """

    def __init__(self,
                max_brightness: int = 200,
                min_brightness: int = 20, 
                 *args,
                 **kwargs):
        """
        Initialization method.
        
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.max_brightness = max_brightness
        self.min_brightness = min_brightness

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if CleaningKeys.brightness in sample:
            return sample

        # there is no image in this sample
        sample[CleaningKeys.brightness] = ''
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
        
        sample[CleaningKeys.brightness] = np.mean(cv2.split(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV))[2])
        
        if sample[CleaningKeys.brightness] > self.max_brightness:
            sample[CleaningKeys.brightness_label] = "bright"
        elif sample[CleaningKeys.brightness] < self.min_brightness:
            sample[CleaningKeys.brightness_label] = "dark"
        else:
            sample[CleaningKeys.brightness_label] = "normal"
            
        return sample

    def process(self, sample):
        # brightness = sample[CleaningKeys.brightness]
        return True
            
