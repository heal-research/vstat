cmake_minimum_required(VERSION 3.20)

include(cmake/prelude.cmake)

project(
    vstat
    VERSION 1.0.0
    DESCRIPTION "C++20 library of computationally efficient methods for calculating sample statistics (mean, variance, covariance, correlation)."
    HOMEPAGE_URL "https://github.com/heal-research/vstat"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

docs_early_return()

include(cmake/CPM.cmake)
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
target_compile_features(vstat_vstat INTERFACE cxx_std_20)

# ---- Dependencies ----
CPMAddPackage(
    NAME eve
    VERSION 2023.02.15
    GITHUB_REPOSITORY jfalcou/eve
    OPTIONS "EVE_BUILD_TEST OFF"
)
target_link_libraries(vstat_vstat INTERFACE eve::eve)

if (NOT VSTAT_NAMESPACE)
    set(VSTAT_NAMESPACE vstat)
endif()
message(STATUS "vstat namespace defined as '${VSTAT_NAMESPACE}'")

# ---- Python module ----
if(vstat_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(vstat_python ${PROJECT_SOURCE_DIR}/src/vstat.cpp)
    target_link_libraries(vstat_python PRIVATE vstat::vstat)
    set_target_properties(vstat_python PROPERTIES OUTPUT_NAME "vstat")
endif()

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----
if(NOT vstat_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(AUTHOR_WARNING "Developer mode is intended for developers of vstat")
endif()

include(cmake/dev-mode.cmake)

