# mlir-hello
Learning MLIR ...

1. Build MLIR
```shell
git submodule update --init
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=$(realpath ../../install) \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --target install
```

2. Setup Environment
```shell
source setup.sh
```

3. Build Tutorial Project
```shell
cd mlir-tutorial
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_INSTALL_PREFIX=$(realpath ../../install)
cmake --build .
```

**Reference:**
- [KEKE046/mlir-tutorial](https://github.com/KEKE046/mlir-tutorial)
