# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from .parse_config import get_config
from .models_mgt import ai_runner_invoke, load_model_for_training
from .gen_h_file import gen_h_user_file_n6
from .connections import skeleton_connections_dict, stm32_to_opencv_colors_dict