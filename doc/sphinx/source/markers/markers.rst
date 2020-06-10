Markers usage
=============



Print the markers
~~~~~~~~~~~~~~~~~

You can find the pdf of the marker to use in the :code:`markersToPrint` of the project root directory.
We recommend to print the markers on a hard, flat and non reflective surface.


Generate the markers
~~~~~~~~~~~~~~~~~~~~~~

In the :code:`markersToPrint` directory you can also find a python program :code:`generate.py` to generate the svg file of the markers.
You can customize the size and print the id of the marker on the corner.

Here is the usage and options:

.. code::

    usage: generate.py [-h] [--rings N] [--outdir dir] [--margin N] [--radius N]
                       [--addId] [--whiteBackground]

    Generate the svg file for the markers.

    optional arguments:
      -h, --help         show this help message and exit
      --rings N          the number of rings (possible values {3, 4}, default: 3)
      --outdir dir       the directory where to save the files
      --margin N         the margin to add around the external ring (default: 400)
      --radius N         the radius of the outer circle (default: 500)
      --addId            add the marker id on the top left corner
      --whiteBackground  set the background (outside the marker) to white instead
                         of transparent

For example, calling:

.. code:: shell

   ./generate.py --outdir markers3 --margin 100 --addId

it will create a directory :code:`markers3` where it saves an svg file for each marker with a margin around the marker of 100 and with the ID of the marker printed on the top left corner.
