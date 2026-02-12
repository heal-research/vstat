include(CMakeFindDependencyMacro)

# INTERFACE dependencies
find_dependency(eve CONFIG REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/vstatTargets.cmake")
check_required_components(vstat)

