from data_juicer.utils.constant import DEFAULT_PREFIX
from datasets import Dataset, load_dataset, concatenate_datasets

from ..base_op import OPERATORS, Mycleanlab
from ..op_fusion import LOADED_IMAGES

import numpy as np
import pandas as pd
from cleanvision import Imagelab
from PIL import Image

from tqdm import tqdm
from multiprocessing.pool import ThreadPool


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

    # def save_results(self, sample):
    #     for issue in self.issues:
    #         index = self.hf_dataset[self.image_key + "_path"].index(sample.get(self.image_key))
    #         sample[DEFAULT_PREFIX + issue] = self.res_df.iloc[[index]].get(issue).to_list()[0]
    #     return sample

    def save_results(self, sample):
        for issue in self.issues:
            index = self.index_lookup.get(sample.get(self.image_key))
            if index is not None:
                sample[DEFAULT_PREFIX + issue] = self.res_df.iloc[[index]].get(issue).to_list()[0]
        return sample
    
    def process(self, dataset, num_proc):
        image_paths = dataset[self.image_key]
        hf_dataset_lst, res_df_lst = [], []
        chunk_size = 100000
        for j, image_pathxs in enumerate([image_paths[i : i + chunk_size] for i in range(0, len(image_paths), chunk_size)]):
            def worker(_):
                return Image.open(_)
            with ThreadPool(processes = num_proc) as pool:
                image_keys = list(tqdm(pool.imap(worker, image_pathxs), total=len(image_pathxs), desc='Images Loading'))
                pool.terminate()
                
            my_dict = {self.image_key: image_keys, self.image_key + "_path": dataset[self.image_key][j * chunk_size : (j + 1) * chunk_size]}
            tmp_dataset = Dataset.from_dict(my_dict)
            imagelab = Imagelab(hf_dataset=tmp_dataset, image_key=self.image_key)
            imagelab.find_issues()
            hf_dataset_lst.append(tmp_dataset.remove_columns([self.image_key]))
            res_df_lst.append(imagelab.issues)
            
        self.hf_dataset = concatenate_datasets(hf_dataset_lst)
        self.res_df = pd.concat(res_df_lst)
        
        self.index_lookup = {v: i for i, v in enumerate(self.hf_dataset[self.image_key + "_path"])}
        dataset = dataset.map(self.save_results)        
        return dataset
            
