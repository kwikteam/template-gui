# -*- coding: utf-8 -*-

"""Template matching GUI."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging
import os.path as op

import numpy as np

from phy.cluster.manual.gui_component import ManualClustering
from phy.cluster.manual.views import (ScatterView,
                                      select_traces,
                                      extract_spikes,
                                      )
from phy.gui import create_app, create_gui, run_app
from phy.io import Context, Selector
from phy.io.array import _spikes_in_clusters, concat_per_cluster
from phy.utils import Bunch

from phycontrib.kwik.store import create_cluster_store
from phycontrib.kwik_gui.gui import (add_waveform_view,
                                     add_trace_view,
                                     add_correlogram_view,
                                     )

from model import get_model, subtract_templates

logging.getLogger(__name__).setLevel('DEBUG')


# -----------------------------------------------------------------------------
# Launch the template matching GUI
# -----------------------------------------------------------------------------

model = get_model()

# Create the context.
path = '.'
context = Context(op.join(op.dirname(path), '.phy'))


# Define and cache the cluster -> spikes function.
@context.cache(memcache=True)
def spikes_per_cluster(cluster_id):
    return np.nonzero(model.spike_clusters == cluster_id)[0]
model.spikes_per_cluster = spikes_per_cluster

selector = Selector(model.spikes_per_cluster)
create_cluster_store(model, selector=selector, context=context)


create_app()


# List of plugins activated by default.
plugins = ['SaveGeometryStatePlugin',
           ]

# Create the GUI.
gui = create_gui(name='TemplateGUI',
                 plugins=plugins,
                 state={'ClusterView': {
                        'quality': 'n_spikes',
                        # 'similarity': 'sim_templates',
                        }
                        },
                 )
gui.model = model

# Create the manual clustering.
mc = ManualClustering(model.spike_clusters,
                      model.spikes_per_cluster,
                      cluster_groups=model.cluster_groups,
                      )
mc.attach(gui)


def add_amplitude_view(gui):

    @concat_per_cluster
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


def select(cluster_id, n=None):
    assert isinstance(cluster_id, int)
    assert cluster_id >= 0
    return selector.select_spikes([cluster_id], max_n_spikes_per_cluster=n)


def _get_data(**kwargs):
    kwargs['spike_clusters'] = model.spike_clusters[kwargs['spike_ids']]
    return Bunch(**kwargs)


def traces(interval):
    """Load traces and spikes in an interval."""
    tr = select_traces(model.all_traces, interval,
                       sample_rate=model.sample_rate,
                       )
    tr = tr - np.mean(tr, axis=0)

    a, b = model.spike_times.searchsorted(interval)
    sc = model.spike_clusters[a:b]

    # Remove templates.
    tr_sub = subtract_templates(tr,
                                start=interval[0],
                                spike_times=model.spike_times[a:b],
                                amplitudes=model.all_amplitudes[a:b],
                                spike_templates=model.templates[sc],
                                whitening_matrix=model.whitening_matrix,
                                sample_rate=model.sample_rate,
                                scaling_factor=1. / 200)

    return [tr, tr_sub]
model.traces = traces


def spikes_traces(interval):
    traces1, traces2 = model.traces(interval)
    out1 = extract_spikes(traces1, interval,
                          sample_rate=model.sample_rate,
                          spike_times=model.spike_times,
                          spike_clusters=model.spike_clusters,
                          all_masks=model.all_masks,
                          n_samples_waveforms=model.n_samples_waveforms,
                          )
    # out2 = extract_spikes(traces2, interval,
    #                       sample_rate=model.sample_rate,
    #                       spike_times=model.spike_times,
    #                       spike_clusters=model.spike_clusters,
    #                       all_masks=model.all_masks,
    #                       n_samples_waveforms=model.n_samples_waveforms,
    #                       )
    return out1
model.spikes_traces = spikes_traces


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


# @tv.actions.add(shortcut='alt+r')
# def toggle_trace_residuals():
#     global do_show_residuals
#     do_show_residuals = not do_show_residuals
#     print("show residuals:", do_show_residuals)
#     tv.on_select()


# Show the GUI.
gui.show()

# Start the Qt event loop.
run_app()

# Close the GUI.
gui.close()
del gui
