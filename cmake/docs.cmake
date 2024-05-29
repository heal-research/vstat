# ---- Redefine docs_early_return ----

# This function must be a macro, so the return() takes effect in the calling
# scope. This prevents other targets from being available and potentially
# requiring dependencies. This cuts down on the time it takes to generate
# documentation in CI.
macro(docs_early_return)
  return()
endmacro()

# ---- Dependencies ----

find_package(Doxygen)

if (DOXYGEN_FOUND)
    message( STATUS "[vstat] Doxygen available")
    set(DOXYGEN_CONFIG ${PROJECT_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs"
        CACHE PATH "Path for the generated Doxygen documentation"
    )
else (DOXYGEN_FOUND)
    message( STATUS "[vstat] Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

# ---- Declare documentation target ----
set(working_dir "${PROJECT_BINARY_DIR}/docs")
add_custom_target(
    docs
    COMMAND "${CMAKE_COMMAND}" -E remove_directory
    "${DOXYGEN_OUTPUT_DIRECTORY}/html"
    "${DOXYGEN_OUTPUT_DIRECTORY}/xml"
    COMMAND DOXYGEN_OUTPUT=${DOXYGEN_OUTPUT_DIRECTORY} ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG}
    COMMENT "Building documentation using Doxygen and doxygen-awesome.css"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
    VERBATIM
)
