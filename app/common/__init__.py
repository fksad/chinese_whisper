# -*- coding: utf-8 -*-
import os
from app.common.log import logger

cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.abspath(os.path.join(cur_dir, '../..'))
print(project_root_path)