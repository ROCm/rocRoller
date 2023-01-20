from rrperf import GEMMRun, CodeGenRun

fp16 = {
    "type_A": "half",
    "type_B": "half",
    "type_C": "half",
    "type_D": "half",
}

fp32 = {
    "type_A": "float",
    "type_B": "float",
    "type_C": "float",
    "type_D": "float",
}

tensile_guidepost_HGEMM = {
    "M": 7680,
    "N": 8448,
    "K": 8448,  # matches the guidepost unit test
    "trans_B": "T",
}


def unit():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=128,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        **fp32,
        numWarmUp=1,
        numOuter=1,
        numInner=1,
    )
    yield GEMMRun(
        M=1024,
        N=1024,
        K=128,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        **fp16,
        numWarmUp=1,
        numOuter=1,
        numInner=1,
    )


def sgemm():
    yield GEMMRun(
        M=3072,
        N=4096,
        K=4096,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        storeLDS_D=True,
        **fp32,
    )
    yield GEMMRun(
        M=3072,
        N=4096,
        K=4096,
        mac_m=128,
        mac_n=64,
        mac_k=16,
        storeLDS_D=True,
        **fp32,
    )


def hgemm_tensile_guidepost():
    yield GEMMRun(
        **tensile_guidepost_HGEMM,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        workgroup_size_x=128,
        workgroup_size_y=2,
        storeLDS_D=True,
        **fp16,
    )


def hgemm():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        storeLDS_D=True,
        **fp16,
    )
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=8,
        **fp16,
    )
    yield from hgemm_tensile_guidepost()
    yield GEMMRun(
        **tensile_guidepost_HGEMM,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=8,
        **fp16,
    )


def codegen():
    yield CodeGenRun(instCount=40000, instructions="comments")
    yield CodeGenRun(instCount=40000, instructions="simple_mfma")
    yield CodeGenRun(instCount=40000, instructions="complex_mfma_with_coop")


def all():
    yield from sgemm()
    yield from hgemm()
    yield from codegen()
