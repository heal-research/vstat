cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    vstat
    VERSION 1.0.0
    DESCRIPTION "C++17 library of computationally efficient methods for calculating sample statistics (mean, variance, covariance, correlation)."
    HOMEPAGE_URL "https://github.com/heal-research/vstat"
    LANGUAGES CXX
)

docs_early_return()

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Declare library ----
add_library(vstat_vstat INTERFACE)
add_library(vstat::vstat ALIAS vstat_vstat)

set_property(
    TARGET vstat_vstat PROPERTY
    EXPORT_NAME vstat
)

target_include_directories(
    vstat_vstat ${vstat_warning_guard}
    INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

# ---- Dependencies ----
find_package(vectorclass)
if (vectorclass_FOUND)
    target_link_libraries(vstat_vstat INTERFACE vectorclass::vectorclass)
else()
    include(FindPkgConfig)
    pkg_search_module(vectorclass vectorclass)
    target_link_libraries(vstat_vstat INTERFACE vectorclass)
endif()
target_compile_features(vstat_vstat INTERFACE cxx_std_17)

if (NOT VSTAT_NAMESPACE)
    set(VSTAT_NAMESPACE vstat)
endif()
message(STATUS "vstat namespace defined as '${VSTAT_NAMESPACE}'")

if(MSVC)
    target_compile_options(vstat_vstat INTERFACE "$<$<CONFIG:Release>:/O2;/std:c++latest>")
else()
    target_compile_options(vstat_vstat INTERFACE "$<$<CONFIG:Release>:-march=x86-64;-mavx2;-mfma;-fno-math-errno>")
endif()

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT vstat_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of vstat"
  )
endif()

include(cmake/dev-mode.cmake)
