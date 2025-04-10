#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"

#include "toy/ToyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "toy/Toy.cpp.inc"

using namespace mlir;
using namespace toy;

void ToyDialect::initialize() {
  // 下面的代码会生成 Op 的列表，专门用来初始化
  addOperations<
#define GET_OP_LIST
#include "toy/Toy.cpp.inc"
  >();
}