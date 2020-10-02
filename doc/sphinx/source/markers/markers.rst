Markers usage
=============

You can find the pdf of the marker to use in the :code:`markersToPrint` of the project root directory.


Print the markers
~~~~~~~~~~~~~~~~~

We recommend to print the markers on a hard, flat and matt surface.

The size of the marker can be chosen considering the minimum size of the marker image that can be detected.
The image of the marker should be roughly no less than 30 pixel of radius for the external ring.
The size of the actual marker to print can be computed considering the distance of the camera w.r.t the marker, the focal length and the resolution of the image.

To **roughly** estimate the (minimum) radius :math:`R` of the marker to print you can use the formula:

.. math::

   R = \frac{m \, u}{f} d

where:

* :math:`m` is minimum size in pixel for the radius (e.g. 30 pixel)

* :math:`u` is the pixel size in mm (that can be found on the specs of the camera)

* :math:`f` is the focal length in mm

* :math:`d` is the distance between the camera and the marker.

For example, for the marker to have a :math:`m=75` pixels radius using a camera with a pixel size of :math:`u=0.00434` mm and a focal length of :math:`f=24` mm and seeing the marker from a distance of :math:`d=5` m, the estimated radius of the actual marker to print is  :math:`R = \frac{75 * 0.00434}{24} \; 5000 = 67.81` mm.



Generate the markers
~~~~~~~~~~~~~~~~~~~~~~

In the :code:`markersToPrint` directory you can also find a python program :code:`generate.py` to generate the svg file of the markers.
You can customize the size and print the id of the marker on the corner.

Here is the usage and options:

.. code::

    usage: generate.py [-h] [--rings N] [--outdir dir] [--margin N] [--radius N]
                       [--addId] [--addCross] [--generatePng] [--generatePdf]
                       [--whiteBackground]

    Generate the svg file for the markers.

    optional arguments:
      -h, --help         show this help message and exit
      --rings N          the number of rings (possible values {3, 4}, default: 3)
      --outdir dir       the directory where to save the files (default: ./)
      --margin N         the margin to add around the external ring (default: 400)
      --radius N         the radius of the outer circle (default: 500)
      --addId            add the marker id on the top left corner
      --addCross         add a small cross in the center of the marker
      --generatePng      also generate a png file
      --generatePdf      also generate a pdf file
      --whiteBackground  set the background (outside the marker) to white instead
                         of transparent


For example, calling:

.. code:: shell

   ./generate.py --outdir markers3 --margin 100 --addId

it will create a directory :code:`markers3` where it saves an svg file for each marker with a margin around the marker of 100 and with the ID of the marker printed on the top left corner.

To generate pdf and/or png file, use the flags :code:`--generatePdf` and :code:`--generatePng` .