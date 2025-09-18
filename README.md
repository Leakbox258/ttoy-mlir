# env
在编译之前，需要安装LLVM和MLIR的支持
```bash
git submodule update --init --recursive --progress
cd ./third_party/llvm-project/
mkdir build
cd build
# you may change the install path you self
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DCMAKE_INSTALL_PREFIX=/usr/local/lib/cmake
ninja
ninja install
```
此时的相关的头文件库文件被安装在 prefix 指定的路径下
# cmake
- 关闭标准 RTTI：llvm-project 本身禁用了 std RTTI，并提供一套轻量级的 LLVM style RTTI。由于链接器禁止链接 no RTTI 和 RTTI 文件，所以建议将本项目的 RTTI 关闭以配合 LLVM 库文件。在定义类时，也需要 提供 LLVM style 的 RTTI 接口
- 找到 LLVM 和 MLIR 的库文件，在上面定义的路径下