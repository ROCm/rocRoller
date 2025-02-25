
#include <memory>

#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace InstructionGenerators
    {
        RegisterComponent(MatrixMultiplyGenerator);

        const std::string MatrixMultiply::Basename = "MatrixMultiply";

        std::string typeStr(auto dtype)
        {
            switch(dtype)
            {
            case DataType::Float:
                return "f32";
            case DataType::Halfx2:
                return "f16";
            case DataType::BFloat16x2:
                return "bf16";
            case DataType::FP8x4:
                return "_fp8_fp8";
            case DataType::BF8x4:
                return "_bf8_bf8";
            case DataType::FP6x16:
            case DataType::BF6x16:
            case DataType::FP4x8:
                return "_f8f6f4";
            default:
                Throw<FatalError>("Unable to determine MFMA type: unhandled data type.",
                                  ShowValue(dtype));
            }
        }

        Generator<Instruction> MatrixMultiplyGenerator::mul(Register::ValuePtr D,
                                                            Register::ValuePtr A,
                                                            Register::ValuePtr B,
                                                            Register::ValuePtr C,
                                                            int                M,
                                                            int                N,
                                                            int                K,
                                                            int                BATCH)
        {
            AssertFatal(A != nullptr);
            AssertFatal(B != nullptr);
            AssertFatal(C != nullptr);

            auto typeA = A->variableType().dataType;
            auto typeB = B->variableType().dataType;
            auto typeC = C->variableType().dataType;
            auto typeD = C->variableType().dataType;
            if(D != nullptr)
                typeD = D->variableType().dataType;

            auto const lanesPerWavefront = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultWavefrontSize);
            auto const packingA = DataTypeInfo::Get(typeA).packing;
            auto const packingB = DataTypeInfo::Get(typeB).packing;
            AssertFatal(M > 0 && N > 0 && K > 0 && BATCH > 0 && lanesPerWavefront > 0,
                        "Invalid inputs",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(K),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront));
            AssertFatal(A->valueCount() * packingA == (size_t)M * K * BATCH / lanesPerWavefront,
                        "A matrix size mismatch",
                        ShowValue(M),
                        ShowValue(K),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * K * BATCH / lanesPerWavefront),
                        ShowValue(A->valueCount()),
                        ShowValue(packingA));
            AssertFatal(B->valueCount() * packingB == (size_t)K * N * BATCH / lanesPerWavefront,
                        "B matrix size mismatch",
                        ShowValue(K),
                        ShowValue(N),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront),
                        ShowValue(K * N * BATCH / lanesPerWavefront),
                        ShowValue(B->valueCount()),
                        ShowValue(packingA));
            AssertFatal(C->valueCount() == (size_t)M * N * BATCH / lanesPerWavefront,
                        "C matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N * BATCH / lanesPerWavefront),
                        ShowValue(C->valueCount()));
            AssertFatal(D->valueCount() == (size_t)M * N * BATCH / lanesPerWavefront,
                        "D matrix size mismatch",
                        ShowValue(M),
                        ShowValue(N),
                        ShowValue(BATCH),
                        ShowValue(lanesPerWavefront),
                        ShowValue(M * N * BATCH / lanesPerWavefront),
                        ShowValue(D->valueCount()));
            AssertFatal(A->regType() == Register::Type::Vector,
                        "Invalid LHS (A) register type",
                        ShowValue(A->regType()));
            AssertFatal(B->regType() == Register::Type::Vector,
                        "Invalid B (B) register type",
                        ShowValue(B->regType()));
            AssertFatal(C->variableType() == D->variableType(),
                        "Invalid D/R2HS (D/C) data types",
                        ShowValue(C->variableType()));
            AssertFatal(D->regType() == Register::Type::Accumulator,
                        "Invalid D (D) register type",
                        ShowValue(A->regType()));

            std::string inputType;
            std::string modifier;

            if(typeA == typeB)
            {
                // Uniform type for A and B.  Result will be similar
                // to "f16", and may be "_f8f6f4".
                inputType = typeStr(typeA);

                // For F8 types, result will be "_fp8_fp8" (or "bf8").
                // Change this to "_f8f6f4" for 32x32x64 and 16x16x128
                // tile sizes.
                if(typeA == DataType::FP8x4 || typeA == DataType::BF8x4)
                {
                    if((M == 32 && N == 32 && K == 64) || (M == 16 && N == 16 && K == 128))
                        inputType = "_f8f6f4";
                }
            }
            else
            {
                // Mixed types for A and B.  Only works for lower
                // precisions.
                auto segA = DataTypeInfo::Get(typeA).segmentVariableType;
                auto segB = DataTypeInfo::Get(typeB).segmentVariableType;
                AssertFatal(DataTypeInfo::Get(segA).elementBits <= 8,
                            "Mixed MFMA inputs (A) must be low precision.",
                            ShowValue(typeA));
                AssertFatal(DataTypeInfo::Get(segB).elementBits <= 8,
                            "Mixed MFMA inputs (B) must be low precision.",
                            ShowValue(typeB));
                inputType = "_f8f6f4";
            }

            // TODO: _fp8_bf8 not handled

            if(inputType == "_f8f6f4")
            {
                if(!((M == 32 && N == 32 && K == 64) || (M == 16 && N == 16 && K == 128)))
                {
                    Throw<FatalError>(
                        "Invalid F8F6F4 MFMA size.", ShowValue(M), ShowValue(N), ShowValue(K));
                }

                modifier = concatenate("cbsz:",
                                       Arithmetic::getModifier(typeA),
                                       " blgp:",
                                       Arithmetic::getModifier(typeB));
            }

            auto mfma = concatenate("v_mfma_", typeStr(typeD), "_", M, "x", N, "x", K, inputType);

            co_yield_(Instruction(mfma, {D}, {A, B, C}, {modifier}, ""));
        }
    }
}
