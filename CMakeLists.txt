cmake_minimum_required(VERSION 3.20)
project(ttoy)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard to conform to")

set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})
set(CMAKE_PREFIX_PATH "/usr/local/lib/cmake") # change to the path where you install llvm & mlir
set(TABLEGEN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/generated")

message(STATUS "tablegen_output_dir set at ${TABLEGEN_OUTPUT_DIR}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    find_package(MLIR REQUIRED CONFIG)
    find_package(LLVM REQUIRED CONFIG)
    
    if(MLIR_FOUND)
        message(STATUS "MLIR found successfully: ${MLIR_DIR}")
    else()
        message(FATAL_ERROR "MLIR not found")
    endif()

    if(LLVM_FOUND)
        message(STATUS "LLVM found subccessfully: ${LLVM_DIR}")
    else()
        message(FATAL_ERROR "LLVM not found")
    endif()

    message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
    
    # set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
    # set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
    # set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
else()
    # Build via external projects mechanism
    set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
    set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
    set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
    set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()  

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})


# include_directories(${PROJECT_BINARY_DIR}/include)
# link_directories(${LLVM_BUILD_LIBRARY_DIR})
# add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(generated)

message(STATUS "TTOY_SOURCE_FILES: ${TTOY_SOURCE_FILES}")

add_executable(ttoy "${CMAKE_SOURCE_DIR}/src/main.cpp" ${TTOY_SOURCE_FILES})

add_dependencies(ttoy 
    TToy_Dialect_Operations 
    TToy_PatternMatch
    TToy_ShapeInferenceOpInterface
)

target_link_libraries(ttoy PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFunctionInterfaces
    MLIRPass
    MLIRTransforms
    MLIRCastInterfaces
    LLVMSupport
    LLVMCore
)

target_compile_definitions(ttoy PRIVATE
    CLANG_ENABLE_ARCSEH=OFF
    CLANG_DEFAULT_CXX_STDLIB=libc++
)