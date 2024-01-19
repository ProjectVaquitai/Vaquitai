import numpy as np

from data_juicer.utils.constant import CleaningKeys

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

import numpy as np
import os

@OPERATORS.register_module('point_cloud_validation_filter')
@LOADED_IMAGES.register_module('point_cloud_validation_filter')
class PointCloudValidationFilter(Filter):
    """Filter to keep samples within normal validation
    """

    def __init__(self,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        """
        Initialization method.
        
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
        

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if CleaningKeys.validation_point_cloud in sample:
            return sample

        sample[CleaningKeys.validation_point_cloud] = []
        # there is no image in this sample
        if self.point_cloud_key not in sample or not sample[self.point_cloud_key]:
            sample[CleaningKeys.validation_point_cloud] = np.array(
                [], dtype=np.int64)
            return sample

        # load images
        loaded_point_cloud_keys = sample[self.point_cloud_key]
        for loaded_point_cloud_key in loaded_point_cloud_keys:
            sample[CleaningKeys.validation_point_cloud].append(int(not os.path.isfile(loaded_point_cloud_key)))

        return sample
        

    def process(self, sample):
        validation = np.array(sample[CleaningKeys.validation_point_cloud])
        
        if self.any:
            return not validation.any()
        else:
            return not validation.all()
