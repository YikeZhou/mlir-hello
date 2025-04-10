#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "toy/ToyOps.h"
#include <memory>

namespace toy {
// 先生成定义
#define GEN_PASS_DECL
#include "toy/ToyPasses.h.inc"

// 再写 create 函数表
std::unique_ptr<mlir::Pass> createConvertToyToArithPass(
    ConvertToyToArithOptions options={}
);

std::unique_ptr<mlir::Pass> createDCEPass();

// 最后注册
#define GEN_PASS_REGISTRATION
#include "toy/ToyPasses.h.inc"

}