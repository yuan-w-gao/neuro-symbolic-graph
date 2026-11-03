cmake_minimum_required(VERSION 3.16)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# Default to Release. Upstream hard-set Debug here, which silently overrode
# -DCMAKE_BUILD_TYPE=Release from the build script (~16x slower parsing).
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(shrg_graph_parser)

add_definitions(
    -D_GLIBCXX_USE_CXX11_ABI=0
    -DBOOST_NO_CXX98_FUNCTION_BASE
    -DBOOST_ALLOW_DEPRECATED_HEADERS
    -DBOOST_NO_CXX11_TEMPLATE_ALIASES
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -include functional")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ENABLE_PROFILING "Enable gprof" OFF)
option(ENABLE_PARSER_CHECK "Enable parser check" OFF)
option(USE_SYSTEM_BOOST "Use system FindBoost.cmake" OFF)
option(USE_PYTHON "Compile python interface with pybind11" OFF)
option(BUILD_UTILITIES "Build optional utility executables" ON)

if(ENABLE_PROFILING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

if(ENABLE_PARSER_CHECK)
  add_definitions( -DSHRG_PARSER_CHECK )
endif()

# Boost configuration
if(USE_SYSTEM_BOOST)
  find_package(Boost REQUIRED)
  message(STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(STATUS "Boost_INCLUDE_DIRS = ${PROJECT_SOURCE_DIR}/third-party/")
  include_directories(${PROJECT_SOURCE_DIR}/third-party/)
endif()

# Include directories
include_directories(
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/third-party/sparsehash-c11
    ${PROJECT_SOURCE_DIR}/third-party/sparsehash-c11/sparsehash
    ${PROJECT_SOURCE_DIR}/src
)

# ===== LIBRARIES =====

# Core SHRG graph parser library
file(GLOB GRAPH_PARSER_SOURCES "src/graph_parser/*.cpp")
file(GLOB AMBIGUITY_METRICS_SOURCES "src/ambiguity_metrics/*.cpp")
add_library(shrg STATIC
    "src/include/bleu.cpp"
    ${GRAPH_PARSER_SOURCES}
    ${AMBIGUITY_METRICS_SOURCES}
)
set_target_properties(shrg PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# Forest caching library (for skipping re-parsing across runs)
add_library(forest_cache STATIC
    "src/forest_cache.cpp"
)
target_link_libraries(forest_cache PRIVATE shrg)
target_include_directories(forest_cache PUBLIC ${PROJECT_SOURCE_DIR}/src)
set_target_properties(forest_cache PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# EM framework
add_library(em_legacy STATIC
    "src/em_framework/find_derivations.cpp"
    "src/em_framework/em_base.cpp"
    "src/em_framework/em.cpp"
    "src/em_framework/em_utils.cpp"
    "src/em_framework/em_batch.cpp"
    "src/em_framework/em_viterbi.cpp"
    "src/em_framework/em_online.cpp"
    "src/manager.cpp"
)
target_link_libraries(em_legacy PRIVATE shrg forest_cache)

# Ambiguity metrics (derivation entropy DP)
add_library(ambiguity_metrics STATIC ${AMBIGUITY_METRICS_SOURCES})
target_include_directories(ambiguity_metrics PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ambiguity_metrics PRIVATE shrg)
set_target_properties(ambiguity_metrics PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# ===== EVALUATION EXECUTABLES =====

# Entropy / BLEU / F1 for a weight file (expects SHRG-indexed weights)
add_executable(comprehensive_eval src/comprehensive_eval.cpp)
target_link_libraries(comprehensive_eval PRIVATE ambiguity_metrics em_legacy forest_cache shrg)

# Gold derivation edge sets (LoadGoldDerivationIndices, ExtractGoldDerivationTree)
add_executable(get_deriv src/get_derivation_edges.cpp)
target_link_libraries(get_deriv PRIVATE em_legacy shrg)

add_executable(compute_entropy_dp src/compute_entropy_dp.cpp)
target_link_libraries(compute_entropy_dp PRIVATE ambiguity_metrics em_legacy shrg)

if(BUILD_UTILITIES)
    add_executable(generate src/generate.cpp)
    target_link_libraries(generate PRIVATE em_legacy shrg)

    add_executable(compute_baseline src/compute_baseline.cpp)
    target_link_libraries(compute_baseline PRIVATE em_legacy shrg)

    add_executable(count_derivations src/count_derivations.cpp)
    target_link_libraries(count_derivations PRIVATE ambiguity_metrics em_legacy shrg)
endif()

# ===== PYTHON MODULE =====

if(USE_PYTHON)
    find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
    file(GLOB PYTHON_SOURCES "src/python/*.cpp")
    add_library(pyshrg SHARED ${PYTHON_SOURCES})
    target_link_libraries(pyshrg PRIVATE em_legacy forest_cache shrg ${Python_LIBRARIES})
    target_include_directories(pyshrg PRIVATE ${Python_INCLUDE_DIRS})
    target_include_directories(pyshrg PRIVATE ${PROJECT_SOURCE_DIR}/third-party/pybind11/include)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python interface: ${USE_PYTHON}")
message(STATUS "Build utilities: ${BUILD_UTILITIES}")
