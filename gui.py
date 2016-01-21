# -*- coding: utf-8 -*-

"""Template matching GUI."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils import Bunch
from phy.io.array import _spikes_in_clusters, _concat
from phy.cluster.manual.views import (ScatterView,)
from phy.gui import create_app, create_gui, run_app
from phycontrib.kwik_gui.gui import (add_waveform_view,
                                     add_trace_view,
                                     add_correlogram_view,
                                     )

from model import get_model


import os.path as op

from phy.io import Context, Selector
from phy.cluster.manual.gui_component import ManualClustering

from phycontrib.kwik import create_cluster_store


logging.getLogger(__name__).setLevel('DEBUG')


# -----------------------------------------------------------------------------
# Launch the template matching GUI
# -----------------------------------------------------------------------------

model = get_model()
create_app()


# List of plugins activated by default.
plugins = ['SaveGeometryStatePlugin',
           ]

# Create the GUI.
gui = create_gui(name='TemplateGUI',
                 model=model,
                 plugins=plugins,
                 state={'ClusterView': {
                        'quality': 'n_spikes',
                        # 'similarity': 'sim_templates',
                        }
                        },
                 )

# Create the manual clustering.
mc = ManualClustering(model.spike_clusters,
                      cluster_groups=model.cluster_groups,)
mc.attach(gui)

# Create the context.
path = '.'
context = Context(op.join(op.dirname(path), '.phy'))


# Create the store.
def spikes_per_cluster(cluster_id):
    # HACK: we get the spikes_per_cluster from the Clustering instance.
    # We need to access it from a function to avoid circular dependencies
    # between the cluster store and manual clustering plugins.
    mc = gui.request('manual_clustering')
    return mc.clustering.spikes_per_cluster[cluster_id]


selector = Selector(spike_clusters=model.spike_clusters,
                    spikes_per_cluster=spikes_per_cluster,
                    )
create_cluster_store(model, selector=selector, context=context)


def add_amplitude_view(gui):

    @_concat
    @context.cache
    def amplitudes(cluster_id):
        spike_ids = _spikes_in_clusters(model.spike_clusters, [cluster_id])
        d = Bunch()
        d.spike_ids = spike_ids
        d.x = model.spike_times[spike_ids]
        d.spike_clusters = cluster_id * np.ones(len(spike_ids),
                                                dtype=np.int32)
        d.y = model.all_amplitudes[spike_ids]
        return d
    model.amplitudes = amplitudes

    view = ScatterView(coords=model.amplitudes,
                       data_bounds=[0, 0, model.duration,
                                    model.amplitudes_lim],
                       )
    view.attach(gui)


add_waveform_view(gui)
add_trace_view(gui)
# add_feature_view(gui)
add_correlogram_view(gui)
add_amplitude_view(gui)


def select(cluster_id, n=None):
    assert isinstance(cluster_id, int)
    assert cluster_id >= 0
    return selector.select_spikes([cluster_id], max_n_spikes_per_cluster=n)


def _get_data(**kwargs):
    kwargs['spike_clusters'] = model.spike_clusters[kwargs['spike_ids']]
    return Bunch(**kwargs)


@_concat
@context.cache
def waveforms(cluster_id):
    spike_ids = select(cluster_id, 100)
    waveforms = np.atleast_2d(model.waveforms[spike_ids])
    assert waveforms.ndim == 3
    masks = np.atleast_2d(model.masks[spike_ids])
    assert masks.ndim == 2
    # Ensure that both arrays have the same number of channels.
    assert masks.shape[1] == waveforms.shape[2]

    # Templates
    mw = model.templates[:, :, [cluster_id]].transpose((2, 1, 0))
    mm = model.template_masks[[cluster_id], :]

    return _get_data(spike_ids=spike_ids,
                     waveforms=waveforms,
                     masks=masks,
                     mean_waveforms=mw,
                     mean_masks=mm,
                     )
model.waveforms = waveforms


# Save.
@gui.connect_
def on_request_save(spike_clusters, groups):
    groups = {c: g.title() for c, g in groups.items()}
    # TODO: save the spike clusters and groups on disk.


# Show the GUI.
gui.show()

# Start the Qt event loop.
run_app()

# Close the GUI.
gui.close()
del gui
