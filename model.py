import os.path as op

import numpy as np
import scipy.io as sio

from phy.cluster.manual.views import select_traces
from phy.io import Context, Selector
from phy.io.array import (_spikes_in_clusters,
                          concat_per_cluster,
                          _get_data_lim,
                          )
from phy.utils import Bunch
from phy.traces import SpikeLoader, WaveformLoader
from phy.traces.filter import apply_filter, bandpass_filter

from phycontrib.kwik.model import _concatenate_virtual_arrays
from phycontrib.kwik.store import create_cluster_store
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
    'whitening_matrix': 'whiteningMatrix.npy',

    'features': 'pcFeatures.npy',
    'features_ind': 'pcFeatureInds.npy',
    'template_features': 'templateFeatures.npy',
    'template_features_ind': 'templateFeatureInds.npy',
}


def read_array(name):
    fn = filenames[name]
    arr_name, ext = op.splitext(fn)
    if ext == '.mat':
        return sio.loadmat(fn)[arr_name]
    elif ext == '.npy':
        return np.load(fn)


def get_masks(templates):
    n_templates, n_samples_templates, n_channels = templates.shape
    templates = np.abs(templates)
    m = templates.max(axis=1)  # (n_templates, n_channels)
    mm = m.max(axis=1)  # (n_templates,
    masks = m / mm[:, np.newaxis]  # (n_templates, n_channels)
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


def subtract_templates(traces,
                       start=None,
                       spike_times=None,
                       spike_clusters=None,
                       amplitudes=None,
                       spike_templates=None,
                       sample_rate=None,
                       ):
    traces = traces.copy()
    st = spike_times
    w = spike_templates * amplitudes[:, np.newaxis, np.newaxis]
    n = traces.shape[0]
    for index in range(w.shape[0]):
        t = int(round((st[index] - start) * sample_rate))
        i, j = 20, 41
        x = w[index]  # (n_samples, n_channels)
        sa, sb = t - i, t + j
        if sa < 0:
            x = x[-sa:, :]
            sa = 0
        elif sb > n:
            x = x[:-(sb - n), :]
            sb = n
        traces[sa:sb, :] -= x
    return traces


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
    templates = np.transpose(templates, (2, 1, 0))
    n_templates, n_samples_templates, n_channels = templates.shape

    channel_mapping = read_array('channel_mapping').squeeze().astype(np.int32)
    assert channel_mapping.shape == (n_channels,)

    channel_positions = np.c_[read_array('channel_positions_x'),
                              read_array('channel_positions_y')]
    assert channel_positions.shape == (n_channels, 2)

    all_features = np.load(filenames['features'], mmap_mode='r')
    features_ind = np.load(filenames['features_ind'], mmap_mode='r')

    template_features = np.load(filenames['template_features'],
                                mmap_mode='r')
    template_features_ind = np.load(filenames['template_features_ind'],
                                    mmap_mode='r')
    template_features_ind = template_features_ind.T.copy()

    model = Bunch()
    model.n_channels = n_channels
    # Take dead channels into account.
    traces = _concatenate_virtual_arrays([traces], channel_mapping)
    model.n_spikes = n_spikes

    # Amplitudes
    model.all_amplitudes = amplitudes
    model.amplitudes_lim = np.percentile(model.all_amplitudes, 100)

    # Templates
    model.templates = templates
    model.n_samples_templates = n_samples_templates
    model.template_lim = np.max(np.abs(model.templates))
    model.n_templates = len(model.templates)

    def similarity(cluster_id):
        n = model.template_features_ind.shape[1]
        sim = model.template_features_ind[cluster_id - 1]
        sim = [(int(c), -n - i) for i, c in enumerate(sim)]
        sim.extend(model.probe_distance(cluster_id))
        return sim
    model.similarity = similarity

    model.sample_rate = sample_rate
    model.duration = n_samples_t / float(model.sample_rate)
    model.spike_times = spike_samples / float(model.sample_rate)
    model.spike_clusters = spike_clusters
    model.cluster_ids = np.unique(model.spike_clusters)
    n_clusters = len(model.cluster_ids)
    model.channel_positions = channel_positions
    model.all_traces = traces

    model.whitening_matrix = read_array('whitening_matrix')

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

    model.n_features_per_channel = 3
    model.n_samples_waveforms = n_samples_waveforms
    model.cluster_groups = {c: None for c in range(n_clusters)}

    # Create the context.
    path = '.'
    context = Context(op.join(op.dirname(path), '.phy'))

    # Define and cache the cluster -> spikes function.
    @context.memcache
    def spikes_per_cluster(cluster_id):
        return np.nonzero(model.spike_clusters == cluster_id)[0]
    model.spikes_per_cluster = spikes_per_cluster

    selector = Selector(model.spikes_per_cluster)
    create_cluster_store(model, selector=selector, context=context)

    def select(cluster_id, n=None):
        assert isinstance(cluster_id, int)
        assert cluster_id >= 0
        return selector.select_spikes([cluster_id], max_n_spikes_per_cluster=n)

    def _get_data(**kwargs):
        kwargs['spike_clusters'] = model.spike_clusters[kwargs['spike_ids']]
        return Bunch(**kwargs)

    # Check sparse features arrays shapes.
    assert all_features.ndim == 3
    n_loc_chan = all_features.shape[2]
    assert all_features.shape == (model.n_spikes,
                                  model.n_features_per_channel,
                                  n_loc_chan,
                                  )
    assert features_ind.shape == (n_loc_chan, model.n_templates)

    n_sim_tem = template_features.shape[1]
    assert template_features.shape == (n_spikes, n_sim_tem)
    assert template_features_ind.shape == (n_templates, n_sim_tem)
    model.template_features_ind = template_features_ind

    @concat_per_cluster
    @context.cache
    def features(cluster_id):
        spike_ids = select(cluster_id, 1000)
        nc = model.n_channels
        nfpc = model.n_features_per_channel
        ns = len(spike_ids)
        shape = (ns, nc, nfpc)
        f = np.zeros(shape)
        # Sparse channels.
        ch = features_ind[:, cluster_id] - 1
        # Populate the dense features array.
        f[:, ch, :] = np.transpose(all_features[spike_ids, :, :], (0, 2, 1))
        m = model.masks(cluster_id).masks
        return _get_data(spike_ids=spike_ids,
                         features=f,
                         masks=m,
                         )
    model.features = features

    model.background_features = lambda: None

    @context.memcache
    @context.cache
    def feature_lim():
        """Return the max of a subset of the feature amplitudes."""
        return _get_data_lim(all_features, 1000)
    model.feature_lim = feature_lim

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

    @context.cache
    def get_template_features(cluster_ids):
        d = Bunch()
        if len(cluster_ids) < 2:
            return None
        assert template_features.shape == (n_spikes, n_sim_tem)
        assert template_features_ind.shape == (n_templates, n_sim_tem)
        cx, cy = map(int, cluster_ids[:2])
        sim_x = template_features_ind[cx - 1].tolist()
        sim_y = template_features_ind[cy - 1].tolist()
        if cx not in sim_y or cy not in sim_x:
            return None
        sxy = sim_x.index(cy)
        syx = sim_y.index(cx)
        spikes_x = _spikes_in_clusters(model.spike_clusters, [cx])
        spikes_y = _spikes_in_clusters(model.spike_clusters, [cy])
        spike_ids = np.hstack([spikes_x, spikes_y])
        d.x = np.hstack([template_features[spikes_x, 0],
                         template_features[spikes_y, syx]])
        d.y = np.hstack([template_features[spikes_x, sxy],
                         template_features[spikes_y, 0]])
        d.spike_ids = spike_ids
        d.spike_clusters = model.spike_clusters[spike_ids]
        return d
    model.template_features = get_template_features
    tf = template_features[:1000, :]
    m, M = tf.min(), tf.max()
    model.template_features_bounds = [m, m, M, M]

    wmi = np.linalg.inv(model.whitening_matrix / 200.)
    # Unwhiten the templates.
    model.templates_unw = np.dot(model.templates, wmi)

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
                                    spike_clusters=sc,
                                    amplitudes=model.all_amplitudes[a:b],
                                    spike_templates=model.templates_unw[sc],
                                    sample_rate=model.sample_rate,
                                    )

        return [Bunch(traces=tr),
                Bunch(traces=tr_sub, color=(.25, .25, .25, .75))]
    model.traces = traces

    return model


if __name__ == '__main__':
    model = get_model()
