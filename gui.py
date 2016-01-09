# -*- coding: utf-8 -*-

"""Template matching GUI."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

logging.getLogger(__name__).setLevel('DEBUG')

from phy.gui import create_app, create_gui, run_app

from model import get_model

model = get_model()

create_app()

# List of plugins activated by default.
plugins = ['ContextPlugin',
           'ClusterStorePlugin',
           'ManualClusteringPlugin',
           'WaveformViewPlugin',
           # 'FeatureViewPlugin',
           'CorrelogramViewPlugin',
           'TraceViewPlugin',
           'SaveGeometryStatePlugin',
           ]

# Create the GUI.
gui = create_gui(name='TemplateGUI',
                 # subtitle=model.kwik_path,
                 model=model,
                 plugins=plugins,
                 )

# # Save.
# @gui.connect_
# def on_request_save(spike_clusters, groups):
#     groups = {c: g.title() for c, g in groups.items()}
#     model.save(spike_clusters, groups)

# Show the GUI.
gui.show()

# Start the Qt event loop.
run_app()

# Close the GUI.
gui.close()
del gui
