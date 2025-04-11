#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/ToyDialect.h"
#include "toy/ToyOps.h"
#include "toy/ToyTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// #include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define GEN_PASS_DEF_CONVERTTOYTOARITH
#include "toy/ToyPasses.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace llvm;
using namespace toy;

struct AddOpPattern : OpConversionPattern<AddOp> {
    using OpConversionPattern<AddOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(AddOp op, AddOpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto inputs = to_vector(adaptor.getInputs());
        auto result = inputs[0];
        for (size_t i = 1; i < inputs.size(); ++i) {
            assert(inputs[i]);
            result = rewriter.create<arith::AddIOp>(op->getLoc(), result, inputs[i]);
        }
        rewriter.replaceOp(op, ValueRange{result});
        return success();
    }
};

struct SubOpPattern : OpConversionPattern<SubOp> {
    using OpConversionPattern<SubOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(SubOp op, SubOpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::SubIOp>(op, adaptor.getLhs(), adaptor.getRhs());
        return success();
    }
};

struct ConstantOpPattern : OpConversionPattern<ConstantOp> {
    using OpConversionPattern<ConstantOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(ConstantOp op, ConstantOpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValueAttr());
        return success();
    }
};

struct ReturnOpPattern : OpConversionPattern<ReturnOp> {
    using OpConversionPattern<ReturnOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(ReturnOp op, ReturnOpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto data = adaptor.getData();
        rewriter.startRootUpdate(op);
        op.getDataMutable().assign(data);
        rewriter.finalizeRootUpdate(op);
        return success();
    }
};

struct CallOpPattern : OpConversionPattern<CallOp> {
    using OpConversionPattern<CallOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(CallOp op, CallOpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        SmallVector<Type> resultTypes;
        assert(succeeded(getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)));
        rewriter.replaceOpWithNewOp<CallOp>(op, resultTypes, op.getCallee(), adaptor.getOperands());
        return success();
    }
};

struct ConvertToyToArithPass :
    toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>
{
  // 使用父类的构造函数
  using toy::impl::ConvertToyToArithBase<ConvertToyToArithPass>::ConvertToyToArithBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    // 如果需要转换为 LLVM IR，则需要添加 LLVM IR 的 Dialect 和对应的 TypeConverter

    auto checkValid = [](Operation *f) {
      return llvm::all_of(f->getOperandTypes(), [](Type t) {
        return !isa<ToyIntegerType>(t);
      });
    };
    target.addDynamicallyLegalOp<AddOp, SubOp, ConstantOp, ReturnOp, CallOp>(checkValid);

    // 添加 ToyInteger 类型转换
    TypeConverter converter;
    converter.addConversion([&](ToyIntegerType type) -> std::optional<IntegerType> {
        return IntegerType::get(&getContext(), type.getWidth());
    });
    converter.addTargetMaterialization([](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> std::optional<Value> {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
    });

    // 重写规则
    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpPattern, SubOpPattern, ConstantOpPattern, ReturnOpPattern, CallOpPattern>(converter, &getContext());

    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);

    // partialConversion：如果 Pattern 转换结果是 Legal，则保留转换结果。如果输入存在 IllegalOp 或 IllegalDialect，立刻报错。
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> toy::createConvertToyToArithPass(
    ConvertToyToArithOptions options) {
  return std::make_unique<ConvertToyToArithPass>(options);
}