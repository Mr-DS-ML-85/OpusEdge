# GTest already resolved (system or FetchContent) by the parent CMakeLists.

function(add_opusedge_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE opusedge gtest gtest_main)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_opusedge_test(test_delta_ar)
add_opusedge_test(test_selkv)
add_opusedge_test(test_smsa)
add_opusedge_test(test_delta_rank)
add_opusedge_test(test_head_gate)
add_opusedge_test(test_state_compress)
add_opusedge_test(test_signal)
add_opusedge_test(test_cal)
add_opusedge_test(test_pareto)
