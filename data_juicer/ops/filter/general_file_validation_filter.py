import numpy as np

from data_juicer.utils.constant import CleaningKeys

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

import numpy as np
import os

@OPERATORS.register_module('general_file_validation_filter')
@LOADED_IMAGES.register_module('general_file_validation_filter')
class GeneralFileValidationFilter(Filter):
    """Filter to keep samples within normal validation
    """

    def __init__(self,
                 any_or_all: str = 'any',
                 general_file_key: str = 'general_file',
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
        self.general_file_key = general_file_key
        

    def compute_stats(self, sample, context=False):
        file_path = sample[self.general_file_key]
        if file_path is None:
            sample[CleaningKeys.validation_general_file] = [1]
            return sample
        else:
            sample[CleaningKeys.validation_general_file] = [int(not os.path.isfile(file_path))]

        return sample
        

    def process(self, sample):
        validation = np.array(sample[CleaningKeys.validation_general_file])
        
        if self.any:
            return not validation.any()
        else:
            return not validation.all()
