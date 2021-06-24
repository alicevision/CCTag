CCTag library
===================
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3989/badge)](https://bestpractices.coreinfrastructure.org/projects/3989) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/a99505cea4bd4cd195b9b65b6114ee90)](https://www.codacy.com/manual/alicevision/CCTag?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=alicevision/CCTag&amp;utm_campaign=Badge_Grade)

Detection of CCTag markers made up of concentric circles. Implementations in both CPU and GPU.

The library is the implementation of the paper: 

* Lilian Calvet, Pierre Gurdjos, Carsten Griwodz, Simone Gasparini. **Detection and Accurate Localization of Circular Fiducials Under Highly Challenging Conditions.** In: *Proceedings of the International Conference on Computer Vision and Pattern Recognition (CVPR 2016)*, Las Vegas, E.-U., IEEE Computer Society, p. 562-570, June 2016.  https://doi.org/10.1109/CVPR.2016.67 

If you want to cite this work in your publication, please use the following 

```latex
@inproceedings{calvet2016Detection,
  TITLE = {{Detection and Accurate Localization of Circular Fiducials under Highly Challenging Conditions}},
  AUTHOR = {Calvet, Lilian and Gurdjos, Pierre and Griwodz, Carsten and Gasparini, Simone},
  BOOKTITLE = {{Proceedings of the 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
  ADDRESS = {Las Vegas, United States},
  PAGES = {562 - 570},
  YEAR = {2016},
  MONTH = Jun,
  DOI = {10.1109/CVPR.2016.67}
}
```

Marker library
---------

Markers to print are located [here](markersToPrint).

**WARNING**
Please respect the provided margins. The reported detection rate and localization accuracy are valid with completely planar support: be careful not to use bent support (e.g. corrugated sheet of paper).

The four rings CCTags will be available soon.

CCTags requires either CUDA 8.0 and newer or CUDA 7.0 (CUDA 7.5 builds are known to have runtime errors on some devices including the GTX980Ti). The device must have at least compute capability 3.5.

Check your graphic card CUDA compatibility [here](https://github.com/tpruvot/ccminer/wiki/Compatibility).

Building
--------

See [INSTALL](INSTALL.md) text file.

Continuous integration:

|         	| windows 	| linux 	|
|---------	|:-------:	|:-----:	|
| master  	|  [![Build status](https://ci.appveyor.com/api/projects/status/909b01qlixwslvmf?svg=true)](https://ci.appveyor.com/project/AliceVision/cctag)       	| ![Continuous Integration](https://github.com/alicevision/CCTag/workflows/Continuous%20Integration/badge.svg?branch=master)      	|
| develop 	|  [![Build status](https://ci.appveyor.com/api/projects/status/909b01qlixwslvmf/branch/develop?svg=true)](https://ci.appveyor.com/project/AliceVision/cctag/branch/develop)        	| ![Continuous Integration](https://github.com/alicevision/CCTag/workflows/Continuous%20Integration/badge.svg?branch=develop)       	|

Running
-------

Once compiled, you might want to run the CCTag detection on a sample image:

```bash
$ build/src/detection -n 3 -i sample/01.png
```
For the library interface, see [ICCTag.hpp](src/cctag/ICCTag.hpp).

Documentation
-------------

The documentation can be found on the [Read the Docs page](https://cctag.readthedocs.io/en/doc-first/index.html)


License
-------

CCTag is licensed under [MPL v2 license](COPYING.md).

Authors
-------

Lilian Calvet (CPU, lilian.calvet@gmail.com)  
Carsten Griwodz (GPU, griff@simula.no)  
Stian Vrba (CPU, vrba@mixedrealities.no)  
Cyril Pichard (pih@mikrosimage.eu)  
Simone Gasparini (simone.gasparini@gmail.com)

Acknowledgments
---------

This has been developed in the context of the European project [POPART](https://alicevision.org/popart/) founded by European Union’s Horizon 2020 research and innovation programme under grant agreement No 644874.

Additional contributions for performance optimizations have been funded by the Norwegian RCN FORNY2020 project FLEXCAM.
