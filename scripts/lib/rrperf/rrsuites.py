import pathlib
from rrperf.problems import GEMMRun, CodeGenRun, TensileRun

repo_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent

fp4fp4_fp32 = dict(
    type_A="fp4",
    type_B="fp4",
    type_C="float",
    type_D="float",
)

fp6fp6_fp32 = dict(
    type_A="fp6",
    type_B="fp6",
    type_C="float",
    type_D="float",
)

bf6bf6_fp32 = dict(
    type_A="bf6",
    type_B="bf6",
    type_C="float",
    type_D="float",
)

fp8fp8_fp32 = dict(
    type_A="fp8",
    type_B="fp8",
    type_C="float",
    type_D="float",
)

bf8bf8_fp32 = dict(
    type_A="bf8",
    type_B="bf8",
    type_C="float",
    type_D="float",
)

fp16 = dict(
    type_A="half",
    type_B="half",
    type_C="half",
    type_D="half",
)

fp32 = dict(
    type_A="float",
    type_B="float",
    type_C="float",
    type_D="float",
)

SGEMM_3072x4096x4096 = dict(
    M=3072, N=4096, K=4096, mac_m=64, mac_n=64, mac_k=64, **fp32
)

HGEMM_7680x8448x8192 = dict(
    M=7680, N=8448, K=8192, mac_m=64, mac_n=64, mac_k=64, **fp16
)

HGEMM_7680x8448x8448 = dict(
    M=7680,
    N=8448,
    K=8448,  # matches the guidepost unit test
    mac_m=64,
    mac_n=64,
    mac_k=64,
    workgroup_size_x=128,
    workgroup_size_y=2,
    trans_A="N",
    trans_B="T",
    **fp16,
)


def update_parameters(*args, **kwargs):
    rv = {}
    for d in args:
        rv.update(d)
    rv.update(kwargs)
    return rv


def mkGEMM(*args, **kwargs):
    return GEMMRun(**update_parameters(*args, **kwargs))


def unit():
    default = dict(
        M=1024,
        N=1024,
        K=128,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        numWarmUp=1,
        numOuter=1,
        numInner=1,
    )
    yield mkGEMM(default, fp32)
    yield mkGEMM(default, fp16)


def sgemm():
    yield mkGEMM(SGEMM_3072x4096x4096)
    yield mkGEMM(SGEMM_3072x4096x4096, mac_m=128, mac_n=64, mac_k=16)


def hgemm_tensile_guidepost():
    yield mkGEMM(HGEMM_7680x8448x8448)


def hgemm():
    yield mkGEMM(HGEMM_7680x8448x8192)
    yield mkGEMM(
        HGEMM_7680x8448x8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8192,
        trans_A="N",
        trans_B="T",
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=4,
    )

    yield mkGEMM(HGEMM_7680x8448x8192, trans_A="T", trans_B="N")
    yield mkGEMM(HGEMM_7680x8448x8192, trans_A="T", trans_B="T")
    yield mkGEMM(HGEMM_7680x8448x8192, trans_A="N", trans_B="T")

    yield mkGEMM(HGEMM_7680x8448x8448)
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=4,
    )
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
    )

    for sched in ["Priority", "Cooperative", "Sequential"]:
        yield mkGEMM(
            M=3840,
            N=4224,
            K=4224,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield mkGEMM(
            M=1024,
            N=50304,
            K=8192,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield mkGEMM(
            M=3840,
            N=4224,
            K=4224,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            betaInFma=False,
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield mkGEMM(
            M=1024,
            N=50304,
            K=8192,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            betaInFma=False,
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield mkGEMM(
            M=8448,
            N=8448,
            K=128,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield mkGEMM(
            M=7680,
            N=8448,
            K=8192,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            workgroup_size_x=256,
            workgroup_size_y=1,
            trans_A="N",
            trans_B="T",
            visualize=False,
            scheduler=sched,
            **fp16,
        )

    yield from visualizer()


def visualizer():
    yield mkGEMM(
        M=512,
        N=768,
        K=512,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        alpha=2.0,
        beta=0.5,
        workgroup_size_x=256,
        workgroup_size_y=1,
        trans_A="N",
        trans_B="T",
        storeLDS_D=False,
        visualize=True,
        prefetch=False,
        **fp16,
    )


def guidepost_1():
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        scheduler="Priority",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
    )


def guidepost_2():
    yield mkGEMM(
        HGEMM_7680x8448x8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        scheduler="Priority",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
    )


def hgemm_no_store_LDS():
    for K in [8448, 8192]:
        yield mkGEMM(
            HGEMM_7680x8448x8448,
            K=K,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            storeLDS_D=False,
            scheduler="Priority",
            prefetch=True,
            prefetchInFlight=2,
            prefetchLDSFactor=0,
        )


def tensile_asm_guidepost_1():
    yield mkGEMM(
        M=7680,
        N=8448,
        K=8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        trans_A="N",
        trans_B="T",
        visualize=False,
        scheduler="TENSILE_ASM",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        **fp16,
    )


def tensile_asm_guidepost_2():
    yield mkGEMM(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        trans_A="N",
        trans_B="T",
        visualize=False,
        scheduler="TENSILE_ASM",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        **fp16,
    )


def tensile_guidepost():
    yield TensileRun(
        config=str(
            repo_dir
            / "test"
            / "unit"
            / "GemmGuidePost"
            / "HGemmGuidePost_Optimized.yaml"
        ),
    )


def tensile_sgemm_guidepost():
    yield TensileRun(
        config=str(
            repo_dir
            / "test"
            / "unit"
            / "GemmGuidePost"
            / "GemmGuidePost_Optimized.yaml"
        ),
    )


def streamk():
    for twoTile in {True, False}:
        # SGEMM
        yield mkGEMM(
            SGEMM_3072x4096x4096,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            # prefetchInFlight=2,
            # prefetchLDSFactor=2,
            streamK=True,
            streamKTwoTile=twoTile,
        )
        # HGEMM
        yield mkGEMM(
            HGEMM_7680x8448x8448,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            # prefetchInFlight=2,
            # prefetchLDSFactor=2,
            streamK=True,
            streamKTwoTile=twoTile,
        )
        yield mkGEMM(
            HGEMM_7680x8448x8192,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            trans_A="N",
            trans_B="T",
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            streamK=True,
            streamKTwoTile=twoTile,
        )
        yield mkGEMM(
            HGEMM_7680x8448x8192,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            trans_A="N",
            trans_B="T",
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            streamK=True,
            streamKTwoTile=twoTile,
            numWGs=220,
        )


def scalar_is_zero():
    # TODO: Make streamK and ConstantPropagation transformation can be both applied
    sgemm = update_parameters(
        SGEMM_3072x4096x4096,
        beta=0.0,
        streamK=False,
    )
    yield mkGEMM(sgemm)
    yield mkGEMM(sgemm, mac_m=128, mac_n=64, mac_k=16)

    hgemm = update_parameters(
        HGEMM_7680x8448x8192,
        beta=0.0,
        streamK=False,
    )
    yield mkGEMM(hgemm)
    yield mkGEMM(
        hgemm,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
    )
    yield mkGEMM(
        hgemm,
        trans_A="N",
        trans_B="T",
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
    )
    yield mkGEMM(
        hgemm,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        beta=0.0,
        workgroup_size_x=64,
        workgroup_size_y=4,
    )


def tensile_benchmarks():
    yield from tensile_guidepost()
    yield from tensile_sgemm_guidepost()
    yield from tensile_asm_guidepost_1()
    yield from tensile_asm_guidepost_2()


def codegen():
    yield CodeGenRun(instCount=40000, instructions="comments")
    yield CodeGenRun(instCount=40000, instructions="simple_mfma")
    yield CodeGenRun(instCount=40000, instructions="complex_mfma_with_coop")


def f8gemm():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=1024,
        mac_m=64,
        mac_n=16,
        mac_k=64,
        workgroup_size_x=256,
        workgroup_size_y=1,
        **fp8fp8_fp32,
    )


def f8gemm_16x16x128_f8f6f4():
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=64,
        mac_n=64,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        wave_k=128,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    yield GEMMRun(
        **params,
        **fp8fp8_fp32,
    )
    yield GEMMRun(
        **params,
        **bf8bf8_fp32,
    )


def f8gemm_32x32x64_f8f6f4():
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=128,
        mac_n=128,
        mac_k=64,
        wave_m=32,
        wave_n=32,
        wave_k=64,
        workgroup_size_x=256,
        workgroup_size_y=1,
    )
    yield GEMMRun(
        **params,
        **fp8fp8_fp32,
    )
    yield GEMMRun(
        **params,
        **bf8bf8_fp32,
    )


def f8gemm_f8f6f4():
    yield from f8gemm_32x32x64_f8f6f4()
    yield from f8gemm_16x16x128_f8f6f4()


def f6gemm_16x16x128_f8f6f4():
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=64,
        mac_n=64,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        wave_k=128,
        workgroup_size_x=256,
        workgroup_size_y=1,
        trans_A="T",
        trans_B="N",
    )
    yield GEMMRun(
        **params,
        **fp6fp6_fp32,
    )
    yield GEMMRun(
        **params,
        **bf6bf6_fp32,
    )


def f6gemm_32x32x64_f8f6f4():
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=128,
        mac_n=128,
        mac_k=64,
        wave_m=32,
        wave_n=32,
        wave_k=64,
        workgroup_size_x=256,
        workgroup_size_y=1,
        trans_A="T",
        trans_B="N",
    )
    yield GEMMRun(
        **params,
        **fp6fp6_fp32,
    )
    yield GEMMRun(
        **params,
        **bf6bf6_fp32,
    )


def f6gemm_f8f6f4():
    yield from f6gemm_32x32x64_f8f6f4()
    yield from f6gemm_16x16x128_f8f6f4()


def f4gemm_16x16x128_f8f6f4():
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=64,
        mac_n=64,
        mac_k=128,
        wave_m=16,
        wave_n=16,
        wave_k=128,
        workgroup_size_x=256,
        workgroup_size_y=1,
        trans_A="T",
        trans_B="N",
    )
    yield GEMMRun(
        **params,
        **fp4fp4_fp32,
    )


def f4gemm_32x32x64_f8f6f4():
    params = dict(
        M=256,
        N=256,
        K=512,
        mac_m=128,
        mac_n=128,
        mac_k=64,
        wave_m=32,
        wave_n=32,
        wave_k=64,
        workgroup_size_x=256,
        workgroup_size_y=1,
        trans_A="T",
        trans_B="N",
    )
    yield GEMMRun(
        **params,
        **fp4fp4_fp32,
    )


def f4gemm_f8f6f4():
    yield from f4gemm_32x32x64_f8f6f4()
    yield from f4gemm_16x16x128_f8f6f4()


def all():
    yield from sgemm()
    yield from hgemm()
    yield from hgemm_no_store_LDS()
    yield from tensile_benchmarks()
    yield from streamk()
    yield from scalar_is_zero()
    yield from codegen()


def hgemm_guideposts():
    yield from guidepost_1()
    yield from guidepost_2()
    yield from tensile_asm_guidepost_1()
    yield from tensile_asm_guidepost_2()


def priority_problems():
    return {
        "1. HGEMM Guidepost": {
            "M": 7680,
            "N": 8448,
            "trans_A": "N",
            "trans_B": "T",
        },
        "2. Halfs": {"type_A": "half"},
        "3. Floats": {"type_A": "float"},
    }
