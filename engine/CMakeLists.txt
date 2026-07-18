cmake_minimum_required(VERSION 3.20)
project(opusedge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 3.4 REQUIRED)

# ── Header-only library ──────────────────────────────────────────────
add_library(opusedge INTERFACE)
target_include_directories(opusedge INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/opusedge-${PROJECT_VERSION}>
)
target_link_libraries(opusedge INTERFACE Eigen3::Eigen)

# ── Install rules ────────────────────────────────────────────────────
include(GNUInstallDirs)
install(DIRECTORY include/opusedge
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opusedge-${PROJECT_VERSION})

install(TARGETS opusedge EXPORT opusedgeTargets)
install(EXPORT opusedgeTargets
        FILE opusedgeTargets.cmake
        NAMESPACE opusedge::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opusedge-${PROJECT_VERSION})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/opusedgeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/opusedgeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/opusedgeConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opusedge-${PROJECT_VERSION}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/opusedgeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/opusedgeConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opusedge-${PROJECT_VERSION}
)

# ── pkg-config metadata ─────────────────────────────────────────────
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/opusedge.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/opusedge.pc"
    @ONLY
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/opusedge.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# ── Subdirectories (opt-in) ──────────────────────────────────────────
if(OPUSEDGE_BUILD_TESTS)
    enable_testing()
    # Prefer a system GTest, fall back to FetchContent so CI works with
    # nothing more than a compiler and network access.
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    add_subdirectory(tests)
endif()

if(OPUSEDGE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(OPUSEDGE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ── Uninstall target ─────────────────────────────────────────────────
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
        IMMEDIATE @ONLY
    )
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
    )
endif()
