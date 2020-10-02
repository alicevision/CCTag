API References
==============


Main Classes
~~~~~~~~~~~~

.. doxygenstruct:: cctag::Parameters
   :members: Parameters, setDebugDir, serialize, setUseCuda

.. doxygenclass:: cctag::CCTag
   :members:


Functions
~~~~~~~~~

.. doxygenfunction:: cctagDetection(boost::ptr_list<ICCTag> &markers, int pipeId, std::size_t frame, const cv::Mat &graySrc, std::size_t nRings = 3, logtime::Mgmt *durations = nullptr, const std::string &parameterFile = "", const std::string &cctagBankFilename = "")

.. doxygenfunction:: cctag::cctagDetection(boost::ptr_list<ICCTag> &markers, int pipeId, std::size_t frame, const cv::Mat &graySrc, const cctag::Parameters &params, logtime::Mgmt *durations = nullptr, const CCTagMarkersBank *pBank = nullptr)


Utility Classes
~~~~~~~~~~~~~~~

.. doxygenclass:: cctag::numerical::geometry::Ellipse
   :members:

.. doxygenstruct:: cctag::logtime::Mgmt
   :members: