cmake_minimum_required(VERSION 3.20)

include(cmake/prelude.cmake)

project(
    vstat
    VERSION 1.0.0
    DESCRIPTION "C++17 library of computationally efficient methods for calculating sample statistics (mean, variance, covariance, correlation)."
    HOMEPAGE_URL "https://github.com/heal-research/vstat"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
target_compile_features(vstat_vstat INTERFACE cxx_std_20)

# ---- Dependencies ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(eve REQUIRED IMPORTED_TARGET eve)
target_link_libraries(vstat_vstat INTERFACE PkgConfig::eve)

if (NOT VSTAT_NAMESPACE)
    set(VSTAT_NAMESPACE vstat)
endif()
message(STATUS "vstat namespace defined as '${VSTAT_NAMESPACE}'")

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
