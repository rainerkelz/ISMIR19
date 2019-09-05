# ISMIR19
This is code to reproduce the results in [this paper](http://arxiv.org/abs/1909.01622).

## Installation

it is recommended you first create a python 3 virtual environment

```
$ python3 -m venv ISMIR19
$ cd ISMIR19
$ source bin/activate
$ git clone https://github.com/rainerkelz/ISMIR19
```

until a new madmom version is released on pip, you'll have to build madmom from source:

```
$ pip install -r ISMIR19/requirements_00.txt
$ git clone https://github.com/CPJKU/madmom.git
$ cd madmom
$ git submodule update --init --remote
$ python setup.py develop
$ cd ..
```

you should have madmom version 0.17.dev0 or higher now (you can check with `pip list` what is installed where, and if it's indeed a `develop` install that points to your virtualenv)

now we'll install the second set of requirements

```
$ pip install -r ICASSP19/requirements_01.txt
```

## Data
obtain the [MAPS](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) dataset

create datadirectory, and symlink to MAPS data
```
$ mkdir data
$ cd data
$ ln -s <path-to-where-MAPS-was-extracted-to> .
$ cd ..
```

create metadata-file for non-overlapping MAPS MUS subset (or use the ones checked in ...)
```
$ python prepare-maps-non-overlapping-splits.py data/maps_piano/data
```

create metadata-file for MAPS ISOL subset (or use the ones checked in ...)
```
$ python prepare-maps-single-note-splits.py data/maps_piano/data
```

create metadata-file for MAPS MUS subset as isolated tracks (or use the ones checked in ...)
```
$ python prepare-maps-individual-tracks.py data/maps_piano/data
```

## Training
train a model on MAPS (the script automatically uses CUDA, if pytorch knows about it)
```
$ python train.py
```

## Generating all the plots
(you will need a (trained) model file for this)

##### Figures 1 and 5
- this generates figure 1 (among other, similar figures) by going from [x, x_pad] -> [y, yz_pad, z] and back again
- filename for figure 1 is `z_hat_pad_y_hat_input_output_MAPS_MUS-chpn-p19_ENSTDkCl.pdf`
- this also generates figure 5 (among other, similar figures) by replacing the inferred [y, yz_pad, z] vector by something produced by an oracle (a perfectly working denoising algorithm, for example, or the groundtruth)
- this is to see what happens when we use the network as a conditional GAN only
- filename for figure 5 is `z_samp_zero_y_true_input_output_MAPS_MUS-chpn-p19_ENSTDkCl.pdf`

```
$ python plot_maps_spec2labels_xyz.py runs/<run-name>/model_state_final.pkl plots/xyz
```

##### Figure 4
- this generates figure 4 (among other, similar figures) by going from [x, x_pad] -> [y, yz_pad, z] -> [y_denoised, 0, z ~ N(O,I)] -> [x_sampled, x_pad_sampled]
- the 'editing' of the inferred variables in y_denoised is done in a **very ad-hoc** fashion, nowhere near a proper denoising algorithm ...
- filename `z_zero_y_edit_input_output_MAPS_MUS-chpn-p19_ENSTDkCl.pdf`

```
$ python plot_maps_spec2labels_edit_xyz.py runs/<run-name>/model_state_final.pkl plots/edit_xyz
```

##### Figure 6
- generate each individual note multiple times with different z ~ N(O, I), and record the interquartile range of differences between generated notes and actual individual note provided in the dataset
- filename is `notes_ENSTDkCl_F.pdf`
```
$ python plot_maps_spec2labels_single_notes_figure.py runs/<run-name>/model_state_final.pkl plots/single_notes
```

##### Figure 7
- sample single notes ("concepts") which were understood, and those which were not
- filename is `good_bad.pdf`
```
$ python plot_maps_spec2labels_single_notes_good_bad.py runs/maps_spec2labels_swd/model_state_final.pkl  plots/single_notes
```
