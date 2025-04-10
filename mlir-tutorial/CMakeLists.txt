cmake_minimum_required(VERSION 3.13.4)

project(mlir-toy VERSION 0.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 生成 compile_commands.json 便于代码高亮
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_subdirectory(ex1-io)
add_subdirectory(ex2-build)

# NOTE: 下面的子目录中都定义了 toy-opt
# 如果同时存在会报错

# add_subdirectory(ex3-dialect)
add_subdirectory(ex5-pass)
