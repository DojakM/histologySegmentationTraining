============
histologySegmentationTraining
============

.. image:: https://github.com/asd/seg_training/workflows/Train%20seg_training%20using%20CPU/badge.svg
        :target: https://github.com/asd/seg_training/workflows/Train%20seg_training%20using%20CPU/badge.svg
        :alt: Github Workflow CPU Training seg_training Status

.. image:: https://github.com/asd/seg_training/workflows/Publish%20Container%20to%20Docker%20Packages/badge.svg
        :target: https://github.com/asd/seg_training/workflows/Publish%20Container%20to%20Docker%20Packages/badge.svg
        :alt: Publish Container to Docker Packages

.. image:: https://github.com/asd/seg_training/workflows/mlf-core%20lint/badge.svg
        :target: https://github.com/asd/seg_training/workflows/mlf-core%20lint/badge.svg
        :alt: mlf-core lint


.. image:: https://readthedocs.org/projects/seg_training/badge/?version=latest
        :target: https://seg_training.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Deep Learning training module for semantic segmentation in histological images. The training dataset is the Lizard
(https://zenodo.org/record/7508237) dataset. The dataset comprises 4981 patched images from multiple colon tissue H&E
stained histological images. Each image contains a segmentation mask with six nuclei classes. The classes are neutrophil
epithelial, lymphocyte, plasma, eosinophil, connective tissue. The training can be done both deterministically and non
deterministically on three different architectures: A basic U-Net, a context U-Net and a spatial transformer U-Net.

.. image:: _images//basic_image.png

.. image:: _images//cu_image.png
    :scale: 70%


* Free software: MIT
* Documentation: https://seg-training.readthedocs.io.


Features
--------

* Fully reproducible mlf-core Pytorch model
* Allows training of pytorch models with the following structures:
    * U-Net
    * Context U-Net
    * Spatial Transformer U-Net

Credits
-------

This package was created with `mlf-core`_ using Cookiecutter_.

.. _mlf-core: https://mlf-core.readthedocs.io/en/latest/
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
