import os.path as op

import numpy as np
import scipy.io as sio

from phy.traces import SpikeLoader, WaveformLoader
from phy.traces.filter import apply_filter, bandpass_filter
from phy.io.array import _spikes_per_cluster, _spikes_in_clusters
from phy.utils import Bunch
from phycontrib.kwik.model import _concatenate_virtual_arrays
from phycontrib.csicsvari.traces import read_dat


filenames = {
    'traces': '20151102_1.dat',
    'amplitudes': 'amplitudes.npy',
    'spike_clusters': 'clusterIDs.npy',
    'templates': 'templates.npy',
    'spike_samples': 'spikeTimes.npy',
    'channel_mapping': 'chanMap0ind.npy',
    'channel_positions_x': 'xcoords.npy',
    'channel_positions_y': 'ycoords.npy',
    # '': 'Fs.mat',
    # '': 'connected.mat',
}


def read_array(name):
    fn = filenames[name]
    arr_name, ext = op.splitext(fn)
    if ext == '.mat':
        return sio.loadmat(fn)[arr_name]
    elif ext == '.npy':
        return np.load(fn)


def get_masks(templates):
    n_channels, n_samples_templates, n_templates = templates.shape
    templates = np.abs(templates)
    m = templates.max(axis=1).T  # n_templates, n_channels
    mm = m.max(axis=1)
    masks = m / mm[:, np.newaxis]
    masks[mm == 0, :] = 0
    return masks


class MaskLoader(object):
    def __init__(self, cluster_masks, spike_clusters):
        self._spike_clusters = spike_clusters
        self._cluster_masks = cluster_masks
        self.shape = (len(spike_clusters), cluster_masks.shape[1])

    def __getitem__(self, item):
        # item contains spike ids
        clu = self._spike_clusters[item]
        return self._cluster_masks[clu]


def get_model():

    # TODO: params
    n_channels_dat = 129
    sample_rate = 25000.
    n_samples_waveforms = 30

    traces = read_dat(filenames['traces'],
                      n_channels=n_channels_dat,
                      dtype=np.int16,
                      )

    n_samples_t, _ = traces.shape
    assert _ == n_channels_dat

    amplitudes = read_array('amplitudes').squeeze()
    n_spikes, = amplitudes.shape

    spike_clusters = read_array('spike_clusters').squeeze()
    spike_clusters -= 1  # 1 -> n_templates
    assert spike_clusters.shape == (n_spikes,)

    spike_samples = read_array('spike_samples').squeeze()
    assert spike_samples.shape == (n_spikes,)

    templates = read_array('templates')
    templates[np.isnan(templates)] = 0
    n_channels, n_samples_templates, n_templates = templates.shape

    channel_mapping = read_array('channel_mapping').squeeze().astype(np.int32)
    assert channel_mapping.shape == (n_channels,)

    channel_positions = np.c_[read_array('channel_positions_x'),
                              read_array('channel_positions_y')]
    assert channel_positions.shape == (n_channels, 2)

    model = Bunch()
    model.n_channels = n_channels
    # Take dead channels into account.
    traces = _concatenate_virtual_arrays([traces], channel_mapping)
    model.n_spikes = n_spikes

    # Amplitudes
    model.all_amplitudes = amplitudes
    model.amplitudes_lim = np.percentile(model.all_amplitudes, 95)

    # Templates
    model.templates = templates
    model.n_samples_templates = n_samples_templates
    model.template_lim = np.max(np.abs(model.templates))

    model.sample_rate = sample_rate
    model.duration = n_samples_t / float(model.sample_rate)
    model.spike_times = spike_samples / float(model.sample_rate)
    model.spike_clusters = spike_clusters
    model.cluster_ids = np.unique(model.spike_clusters)
    n_clusters = len(model.cluster_ids)
    model.channel_positions = channel_positions
    model.all_traces = traces

    # Filter the waveforms.
    order = 3
    filter_margin = order * 3
    b_filter = bandpass_filter(rate=sample_rate,
                               low=500.,
                               high=sample_rate * .475,
                               order=order)

    def the_filter(x, axis=0):
        return apply_filter(x, b_filter, axis=axis)

    # Fetch waveforms from traces.
    waveforms = WaveformLoader(traces=traces,
                               n_samples_waveforms=n_samples_waveforms,
                               filter=the_filter,
                               filter_margin=filter_margin,
                               )
    waveforms = SpikeLoader(waveforms, spike_samples)
    model.all_waveforms = waveforms

    model.template_masks = get_masks(templates)
    model.all_masks = MaskLoader(model.template_masks, spike_clusters)
    # model.features = None
    # model.features_masks = None

    model.spikes_per_cluster = _spikes_per_cluster(model.spike_clusters)
    model.n_features_per_channel = 1
    model.n_samples_waveforms = n_samples_waveforms
    model.cluster_groups = {c: None for c in range(n_clusters)}

    return model


if __name__ == '__main__':
    model = get_model()
