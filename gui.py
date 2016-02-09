# -*- coding: utf-8 -*-

import logging

import numpy as np

from phy.utils._misc import _read_python
from phy.gui import create_app, run_app
from phycontrib.template import create_template_gui, TemplateController


logger = logging.getLogger(__name__)

params = _read_python('params.py')
params['dtype'] = getattr(np, params['dtype'], 'int16')

# tc = TemplateController(**params)
# d = tc.get_features([10])
# print(d)
# exit()

create_app()
plugins = ['SaveGeometryStatePlugin',
           ]
gui = create_template_gui(plugins=plugins, **params)

gui.show()
run_app()
gui.close()
del gui
