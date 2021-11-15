cmake_minimum_required(VERSION 3.14)

project(vstatTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)

# ---- dependencies for testing/benchmarks ----
find_package(PkgConfig REQUIRED)

# ---- GNU Scientific Library ----
find_package(GSL REQUIRED)

# ---- Boost Accumulators ----
find_package(Boost REQUIRED)

# ---- Linasm ----
pkg_search_module(LINASM linasm REQUIRED)

if(PROJECT_IS_TOP_LEVEL)
  find_package(vstat REQUIRED)
  enable_testing()
endif()

add_executable(vstat_test source/vstat_test.cpp)
target_link_libraries(vstat_test PRIVATE vstat::vstat GSL::gsl ${LINASM_LIBRARIES})
target_compile_features(vstat_test PRIVATE cxx_std_17)

add_test(NAME vstat_test COMMAND vstat_test)
