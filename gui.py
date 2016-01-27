# -*- coding: utf-8 -*-

"""Template matching GUI."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from phy.cluster.manual.gui_component import ManualClustering
from phy.cluster.manual.views import ScatterView
from phy.gui import create_app, create_gui, run_app

from phycontrib.kwik_gui.gui import (add_waveform_view,
                                     add_trace_view,
                                     add_correlogram_view,
                                     )

from model import get_model

logging.getLogger(__name__).setLevel('DEBUG')


# -----------------------------------------------------------------------------
# Launch the template matching GUI
# -----------------------------------------------------------------------------

model = get_model()
create_app()

plugins = ['SaveGeometryStatePlugin',
           ]

# Create the GUI.
gui = create_gui(name='TemplateGUI',
                 plugins=plugins,
                 )
gui.model = model


# Create the manual clustering.
mc = ManualClustering(model.spike_clusters,
                      model.spikes_per_cluster,
                      similarity=model.probe_distance,
                      cluster_groups=model.cluster_groups,
                      )
mc.attach(gui)


def add_amplitude_view(gui):
    view = ScatterView(coords=model.amplitudes,
                       data_bounds=[0, 0, model.duration,
                                    model.amplitudes_lim],
                       )
    view.attach(gui)


# Save.
@gui.connect_
def on_request_save(spike_clusters, groups):
    groups = {c: g.title() for c, g in groups.items()}
    # TODO: save the spike clusters and groups on disk.

# Views
add_waveform_view(gui)
# add_feature_view(gui)
add_correlogram_view(gui)
add_amplitude_view(gui)
tv = add_trace_view(gui)

gui.show()
run_app()
gui.close()
del gui
