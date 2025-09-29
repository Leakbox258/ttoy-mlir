# 前言
基于MLIR tutorials。具体来说，可以通过`llvm-project/mlir/docs/Tutorials/Toy/` 和 `llvm-project/mlir/example/toy/`分别找到文档和源文件
本项目的`ttoy`方言，基于`toy dialect`暴改而来，因为`toy dialect`本身和语言BNF设计之下（以及本身的教学目的），整个语言至少有以下不足:
- 1. 只有PrintOp, 没有ScanOp，所以实际上整个程序都是编译期能决定的
- 2. 不强制（以及部分地方不支持）类型声明
- 3. 没控制流，所以也没递归
- 4. 所有非Builtin的Function全都强制内联
- 5. ShapeInfer在目标制品等于或者低级于Affine Dailect时才会启用，也就是toy dialect上类型信息可能不完整
- 6. 只有JIT，但是也没有REPL（源于对ReturnOp的粗暴处理）
- 7. 同上，强制main函数Return Void，导致返回的错误码基本都是随机的
- 8. 运算符太少，而且都是element-wise
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
- 找到 LLVM 和 MLIR 的库文件，在上面定义的路径下，如果需要找某个符号，可以使用那个shell script，不过可能需要修改其中的查找路径