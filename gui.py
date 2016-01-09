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
from phy.cluster.manual.views import ManualClusteringView, _get_color
from phy.gui import create_app, create_gui, run_app

from model import get_model

logging.getLogger(__name__).setLevel('DEBUG')


# -----------------------------------------------------------------------------
# Amplitude view
# -----------------------------------------------------------------------------

class AmplitudeView(ManualClusteringView):
    _default_marker_size = 3.
    # _amplitude_scaling = 1.

    # default_shortcuts = {
    #     'increase': 'ctrl++',
    #     'decrease': 'ctrl+-',
    # }

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

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(AmplitudeView, self).attach(gui)
        # self.actions.add(self.increase)
        # self.actions.add(self.decrease)

    # def increase(self):
    #     """Increase the scaling of the features."""
    #     self.feature_scaling *= 1.2
    #     self.on_select()

    # def decrease(self):
    #     """Decrease the scaling of the features."""
    #     self.feature_scaling /= 1.2
    #     self.on_select()

    # @property
    # def feature_scaling(self):
    #     return self._feature_scaling

    # @feature_scaling.setter
    # def feature_scaling(self, value):
    #     self._feature_scaling = value


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

        alim = np.percentile(model.amplitudes, 95)

        view = AmplitudeView(amplitudes=cs.amplitudes,
                             amplitudes_lim=alim,
                             duration=model.duration,
                             )
        view.attach(gui)

        # fs, = state.get_view_params('AmplitudeView', 'amplitude_scaling')
        # if fs:
        #     view.amplitude_scaling = fs

        # @gui.connect_
        # def on_close():
        #     # Save the box bounds.
        #     state.set_view_params(view,
        #                           amplitude_scaling=view.amplitude_scaling)


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
           'AmplitudeViewPlugin',
           'CorrelogramViewPlugin',
           'TraceViewPlugin',
           'SaveGeometryStatePlugin',
           ]

# Create the GUI.
gui = create_gui(name='TemplateGUI',
                 # subtitle=model.kwik_path,
                 model=model,
                 plugins=plugins,
                 state={'ClusterView': {
                        'quality': 'n_spikes',
                        # 'similarity': 'n_spikes',
                        }
                        },
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
