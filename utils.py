import os


def canonicalize_audio_options(_audio_options, mmspec):
    audio_options = dict(_audio_options)
    whitelisted_keys = set([
        'sample_rate',
        'frame_size',
        'fft_size',
        'hop_size',
        'num_channels',
        'spectrogram_type',
        'filterbank',
        'num_bands',
        'fmin',
        'fmax',
        'fref',
        'norm',
        'norm_filters',
        'unique_filters',
        'circular_shift'
    ])

    spectype = getattr(mmspec, audio_options['spectrogram_type'])
    del audio_options['spectrogram_type']

    if 'filterbank' in audio_options:
        audio_options['filterbank'] = getattr(mmspec, audio_options['filterbank'])

    # delete everything that is not in whitelist
    keys = list(audio_options.keys())
    for key in keys:
        if key not in whitelisted_keys:
            del audio_options[key]

    return spectype, audio_options


def ensure_directory_exists(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
