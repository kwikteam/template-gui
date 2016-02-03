# -*- coding: utf-8 -*-

import logging

import numpy as np

from phy.gui import create_app, run_app
from phycontrib.template import create_template_gui


logger = logging.getLogger(__name__)

kwargs = dict(
    dat_path='20151102_1.dat',
    n_channels_dat=129,
    sample_rate=25000.,
    n_samples_waveforms=30,
    dtype=np.int16,
)

create_app()
plugins = ['SaveGeometryStatePlugin',
           ]
gui = create_template_gui(plugins=plugins, **kwargs)

gui.show()
run_app()
gui.close()
del gui
