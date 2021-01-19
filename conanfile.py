from conans import ConanFile, CMake, tools
from conans.model.version import Version


class CCTagConan(ConanFile):
    name = "cctag"
    version = "1.0.1"
    license = "MPL v2"
    url = "http://alicevision.github.io"
    description = "Detection of CCTag markers made up of concentric circles."
    settings = "os", "compiler", "build_type", "arch"
    requires = "boost/1.69.0", "opencv/4.1.2@oppen/testing", "tbb/2020.1"
    generators = "cmake_find_package"
    exports_sources = "src/*", "cmake/*", "CMakeLists.txt"

    def _configure(self):
        cmake = CMake(self)
        cmake.definitions['CCTAG_WITH_CUDA'] = False
        cmake.definitions['BUILD_APPS'] = False
        cmake.definitions['BUILD_TESTS'] = False
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure()
        cmake.build()

    def package(self):
        cmake = self._configure()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ['CCTag']

