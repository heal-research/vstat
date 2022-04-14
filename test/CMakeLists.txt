cmake_minimum_required(VERSION 3.14)

project(vstatTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)

if(PROJECT_IS_TOP_LEVEL)
    find_package(vstat REQUIRED)
    enable_testing()
endif()

add_executable(vstat_test source/vstat_test.cpp)


# ---- Doctest ----
find_package(doctest REQUIRED)

# ---- GNU Scientific Library ----
find_package(GSL REQUIRED)

# ---- Boost Accumulators ----
find_package(Boost REQUIRED)

# ---- Linasm ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(linasm IMPORTED_TARGET linasm)
if(linasm_FOUND)
    target_link_libraries(vstat_test PRIVATE PkgConfig::linasm)
else()
        message(FATAL_ERROR "LinAsm dependency could not be found.")
endif()



target_link_libraries(vstat_test PRIVATE vstat::vstat GSL::gsl doctest::doctest)
target_compile_features(vstat_test PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(vstat_test PUBLIC "$<$<CONFIG:Release>:/O2;/std:c++latest>")
else()
    target_compile_options(vstat_test PUBLIC "$<$<CONFIG:Release>:-fno-math-errno>")
endif()

add_test(NAME vstat_test COMMAND vstat_test)
