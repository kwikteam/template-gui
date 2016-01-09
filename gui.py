# -*- coding: utf-8 -*-

"""Template matching GUI."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy import IPlugin
from phy.utils import Bunch
from phy.io.array import _index_of, _spikes_in_clusters
from phy.cluster.manual.views import (ManualClusteringView,
                                      WaveformView,
                                      _get_color,
                                      )
from phy.gui import create_app, create_gui, run_app

from model import get_model

logging.getLogger(__name__).setLevel('DEBUG')


# -----------------------------------------------------------------------------
# Amplitude view
# -----------------------------------------------------------------------------

class AmplitudeView(ManualClusteringView):
    _default_marker_size = 3.

    def __init__(self,
                 amplitudes=None,  # function clusters: Bunch(times, amplis)
                 amplitudes_lim=None,
                 duration=None,
                 **kwargs):

        assert amplitudes
        self.amplitudes = amplitudes

        assert duration > 0

        # Initialize the view.
        super(AmplitudeView, self).__init__(**kwargs)

        # Feature normalization.
        self.data_bounds = [0, 0, duration, amplitudes_lim]

    def on_select(self, cluster_ids=None):
        super(AmplitudeView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Get the spike times and amplitudes
        data = self.amplitudes(cluster_ids)
        spike_ids = data.spike_ids
        spike_times = data.spike_times
        spike_clusters = data.spike_clusters
        amplitudes = data.amplitudes
        n_spikes = len(spike_ids)
        assert n_spikes > 0
        assert len(spike_clusters) == n_spikes
        assert len(amplitudes) == n_spikes

        # Get the spike clusters.
        sc = _index_of(spike_clusters, cluster_ids)

        # Plot the amplitudes.
        with self.building():

            m = np.ones(n_spikes)
            # Get the color of the markers.
            color = _get_color(m, spike_clusters_rel=sc, n_clusters=n_clusters)
            assert color.shape == (n_spikes, 4)
            ms = (self._default_marker_size if sc is not None else 1.)

            self.scatter(x=spike_times,
                         y=amplitudes,
                         color=color,
                         data_bounds=self.data_bounds,
                         size=ms * np.ones(n_spikes),
                         )


class AmplitudeViewPlugin(IPlugin):
    def attach_to_gui(self, gui):
        cs = gui.request('cluster_store')
        assert cs

        # state = gui.state
        model = gui.request('model')

        @cs.add(concat=True)
        def amplitudes(cluster_id):
            spike_ids = _spikes_in_clusters(model.spike_clusters, [cluster_id])
            d = Bunch()
            d.spike_ids = spike_ids
            d.spike_times = model.spike_times[spike_ids]
            d.spike_clusters = cluster_id * np.ones(len(spike_ids),
                                                    dtype=np.int32)
            d.amplitudes = model.amplitudes[spike_ids]
            return d

        view = AmplitudeView(amplitudes=cs.amplitudes,
                             amplitudes_lim=model.amplitudes_lim,
                             duration=model.duration,
                             )
        view.attach(gui)


# -----------------------------------------------------------------------------
# Template view
# -----------------------------------------------------------------------------

class TemplateView(WaveformView):
    default_shortcuts = {
        # 'toggle_waveform_overlap': 'o',

        # Box scaling.
        'widen': 'ctrl+shift+right',
        'narrow': 'ctrl+shift+left',
        'increase': 'ctrl+shift+up',
        'decrease': 'ctrl+shift+down',

        # Probe scaling.
        'extend_horizontally': 'shift+alt+right',
        'shrink_horizontally': 'shift+alt+left',
        'extend_vertically': 'shift+alt+up',
        'shrink_vertically': 'shift+alt+down',
    }


class TemplateViewPlugin(IPlugin):
    def attach_to_gui(self, gui):
        state = gui.state
        model = gui.request('model')
        bs, ps, ov = state.get_view_params('TemplateView',
                                           'box_scaling',
                                           'probe_scaling',
                                           'overlap',
                                           )
        cs = gui.request('cluster_store')

        @cs.add(concat=True)
        def templates(cluster_id):
            d = Bunch()
            d.spike_ids = np.array([0])
            d.spike_clusters = np.array([cluster_id])
            d.waveforms = model.templates[:, :, [cluster_id]]. \
                transpose((2, 1, 0))
            d.masks = model.template_masks[[cluster_id], :]
            assert (d.spike_ids.shape[0] ==
                    d.waveforms.shape[0] ==
                    d.masks.shape[0] == 1)
            return d

        assert cs  # We need the cluster store to retrieve the data.
        view = TemplateView(waveforms_masks=cs.templates,
                            channel_positions=model.channel_positions,
                            n_samples=model.n_samples_templates,
                            box_scaling=bs,
                            probe_scaling=ps,
                            waveform_lim=model.template_lim,
                            )
        view.attach(gui)

        if ov is not None:
            view.overlap = ov

        @gui.connect_
        def on_close():
            # Save the box bounds.
            state.set_view_params(view,
                                  box_scaling=tuple(view.box_scaling),
                                  probe_scaling=tuple(view.probe_scaling),
                                  overlap=view.overlap,
                                  )


# -----------------------------------------------------------------------------
# Launch the template matching GUI
# -----------------------------------------------------------------------------

model = get_model()
create_app()

# List of plugins activated by default.
plugins = ['ContextPlugin',
           'ClusterStorePlugin',
           'ManualClusteringPlugin',
           'WaveformViewPlugin',
           # 'TemplateViewPlugin',  # not very useful, better to put templates
                                    # the waveform view  # noaq
           'AmplitudeViewPlugin',
           'CorrelogramViewPlugin',
           'TraceViewPlugin',
           'SaveGeometryStatePlugin',
           ]

# Create the GUI.
gui = create_gui(name='TemplateGUI',
                 model=model,
                 plugins=plugins,
                 state={'ClusterView': {
                        'quality': 'n_spikes',
                        'similarity': 'sim_templates',
                        }
                        },
                 )

cs = gui.request('cluster_store')
selector = gui.request('selector')


def select(cluster_id, n=None):
    assert isinstance(cluster_id, int)
    assert cluster_id >= 0
    return selector.select_spikes([cluster_id], max_n_spikes_per_cluster=n)


def _get_data(**kwargs):
    kwargs['spike_clusters'] = model.spike_clusters[kwargs['spike_ids']]
    return Bunch(**kwargs)


@cs.add(concat=True)
def waveforms_masks(cluster_id):
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
