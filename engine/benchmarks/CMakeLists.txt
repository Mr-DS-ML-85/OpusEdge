add_executable(bench_delta_ar bench_delta_ar.cpp)
target_link_libraries(bench_delta_ar PRIVATE opusedge)

add_executable(bench_scaling bench_scaling.cpp)
target_link_libraries(bench_scaling PRIVATE opusedge)
set_target_properties(bench_scaling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")

add_executable(bench_pipeline bench_pipeline.cpp)
target_link_libraries(bench_pipeline PRIVATE opusedge)
set_target_properties(bench_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")
