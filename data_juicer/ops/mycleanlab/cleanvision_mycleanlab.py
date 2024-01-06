from data_juicer.utils.constant import DEFAULT_PREFIX
from datasets import Dataset

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
                 issues: list = ["is_odd_size_issue",
                                "is_odd_aspect_ratio_issue", 
                                "is_low_information_issue", "is_light_issue", 
                                "is_grayscale_issue", "is_dark_issue", "is_blurry_issue"], 
                                # "is_exact_duplicates_issue", "is_near_duplicates_issue"],
                 *args,
                 **kwargs):
        """
        Initialization method.
        
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.issues = issues

    def save_results(self, sample):
        for issue in self.issues:
            index = self.hf_dataset[self.image_key + "_path"].index(sample.get(self.image_key))
            sample[DEFAULT_PREFIX + issue] = self.res_df.iloc[[index]].get(issue).to_list()[0]
        return sample
        
    def process(self, dataset):
        my_dict = {self.image_key: [Image.open(_) for _ in dataset[self.image_key]], self.image_key + "_path": dataset[self.image_key]}
        self.hf_dataset = Dataset.from_dict(my_dict)
        imagelab = Imagelab(hf_dataset=self.hf_dataset, image_key=self.image_key)
        imagelab.find_issues()
        self.res_df = imagelab.issues
        dataset = dataset.map(self.save_results)        
        return dataset
            
