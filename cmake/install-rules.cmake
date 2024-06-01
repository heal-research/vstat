if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR include/vstat CACHE PATH "")
endif()

# Project is configured with no languages, so tell GNUInstallDirs the lib dir
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package vstat)

install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT vstat_Development
)

install(
    TARGETS vstat_vstat
    EXPORT vstatTargets
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

if (vstat_BUILD_PYTHON AND Python_FOUND AND nanobind_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import sysconfig as sc; print(sc.get_path('platlib', 'posix_user', {'userbase': ''})[1:])"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE VSTAT_PYTHON_SITELIB)
    install(
        TARGETS vstat_python
        EXPORT vstatTargets
        LIBRARY DESTINATION "${VSTAT_PYTHON_SITELIB}/${package}"
    )
endif()

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

# Allow package maintainers to freely override the path for the configs
set(
    vstat_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/${package}"
    CACHE PATH "CMake package config location relative to the install prefix"
)
mark_as_advanced(vstat_INSTALL_CMAKEDIR)

install(
    FILES cmake/install-config.cmake
    DESTINATION "${vstat_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT vstat_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${vstat_INSTALL_CMAKEDIR}"
    COMPONENT vstat_Development
)

install(
    EXPORT vstatTargets
    NAMESPACE vstat::
    DESTINATION "${vstat_INSTALL_CMAKEDIR}"
    COMPONENT vstat_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()

