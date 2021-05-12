# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2020-2021 Heal Research

cmake_minimum_required(VERSION 3.12)

project(
    vstat
    VERSION 1.0.0.0
    DESCRIPTION "Fast, vectorized statistics."
    LANGUAGES CXX
)

include(CTest)
include(FetchContent)
include(FeatureSummary)

# optional packages only required for testing
find_package(PkgConfig)
set_package_properties(FindPkgConfig PROPERTIES TYPE OPTIONAL)

find_package(GSL)
set_package_properties(GSL PROPERTIES TYPE OPTIONAL)

find_package(Boost)
set_package_properties(Boost PROPERTIES TYPE OPTIONAL)

# required dependencies
FetchContent_Declare(
    doctest
    URL https://raw.githubusercontent.com/onqtam/doctest/master/doctest/doctest.h
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/tests/
    DOWNLOAD_NO_EXTRACT 1
)

FetchContent_Declare(
    vectorclass
    GIT_REPOSITORY https://github.com/vectorclass/version2
    GIT_TAG master
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/vcl
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/vcl
)

FetchContent_Declare(
    nanobench
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/nanobench/
    URL https://raw.githubusercontent.com/martinus/nanobench/master/src/include/nanobench.h
    DOWNLOAD_NO_EXTRACT 1
)

FetchContent_MakeAvailable(doctest vectorclass nanobench)

set(THIRDPARTY_INCLUDE_DIRS "${vectorclass_SOURCE_DIR}" "${nanobench_SOURCE_DIR}")

if(PKG_CONFIG_FOUND)
    pkg_search_module(LINASM linasm)
endif()

if(NOT ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(VSTAT_INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/include"
        "${THIRDPARTY_INCLUDE_DIRS}" PARENT_SCOPE)
endif()

# Export compile commands
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

add_library(vstat INTERFACE)
target_include_directories(vstat INTERFACE ./include ${THIRDPARTY_INCLUDE_DIRS})
target_compile_features(vstat INTERFACE cxx_std_17)

if(LINASM_FOUND AND GSL_FOUND AND Boost_FOUND)
    add_executable(vstat-test tests/vstat.cpp)
    target_include_directories(vstat-test PRIVATE src/ ${THIRDPARTY_INCLUDE_DIRS} "${LINASM_INCLUDE_DIRS}"
        "${GSL_INCLUDE_DIRS}" "${Boost_INCLUDE_DIRS}"
    )
    target_link_directories(vstat-test PRIVATE "${LINASM_LIBRARY_DIRS}")
    target_link_libraries(vstat-test vstat GSL::gsl ${LINASM_LIBRARIES})
    target_compile_options(vstat-test PRIVATE "$<$<CONFIG:Debug>:-g>$<$<CONFIG:Release>:-O3;-g;-march=native>")

    add_test(NAME vstat-test COMMAND vstat-test)
    enable_testing()
endif()
