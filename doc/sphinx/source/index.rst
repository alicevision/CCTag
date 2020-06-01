CCTag Library
=============

This library provides the code for the detection of CCTag markers made up of concentric circles :cite:`calvet2016Detection`.
CCTag markers are a robust, highly accurate fiducial system that can be robustly localized in the image even under challenging conditions.
The library can efficiently detect the position of the image of the (common) circle center and identify the marker based on the different ratio of their crown sizes.

.. image:: img/cctags-example.png

An example of three different CCTag markers with three crowns.
Each marker can be uniquely identified thanks to the thickness of each crown, which encodes the information of the marker, typically a unique ID.gi

The implementation is done in both CPU and GPU (Cuda-enabled cards).
The GPU implementation can reach real-time performances on full HD images.


Example of detection in challenging conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: img/challenging.png

Examples of synthetic images of circular fiducials under very challenging shooting conditions i.e., perturbed, in particular, by a (unidirectional) motion blur of magnitude 15px.
The markers are correctly detected and identified (b,d)  with an accuracy of 0.54px and 0.36px resp. in (a) and (c) for the estimated imaged center of the outer ellipse whose semi-major axis (in green) is equal to 31.9px and 34.5px resp.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Install

   install/install

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Documentation

   api/usage
   api/api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Markers

   markers/markers

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: About

   about/about

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: References

   bibliography
