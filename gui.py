# -*- coding: utf-8 -*-

import logging

import numpy as np

from phy import add_default_handler
from phy.utils._misc import _read_python
from phy.gui import create_app, run_app
from phycontrib.template import create_template_gui

logger = logging.getLogger(__name__)

add_default_handler('DEBUG')

params = _read_python('params.py')
params['dtype'] = getattr(np, params['dtype'], 'int16')

create_app()
plugins = ['SaveGeometryStatePlugin',
           ]
gui = create_template_gui(plugins=plugins, **params)

gui.show()
run_app()
gui.close()
del gui
