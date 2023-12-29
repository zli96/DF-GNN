import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dgNN",
    version="0.1",
    author="HenryChang fishming",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "dgNN"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    ext_modules=[
        CUDAExtension(
            "package.fused_gatconv",
            [
                "dgNN/src/fused_gatconv/fused_gatconv.cpp",
                "dgNN/src/fused_gatconv/fused_gatconv_kernel.cu",
                "dgNN/src/fused_gatconv/fused_gatconv_hyper.cu",
                "dgNN/src/fused_gatconv/fused_gatconv_softmax.cu",
                "dgNN/src/fused_gatconv/fused_gatconv_hyper_ablation.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_75", "-lineinfo"]},
            extra_link_args=["-lcurand"],
        ),
        CUDAExtension(
            "package.fused_gtconv",
            [
                "dgNN/src/fused_gtconv/fused_gtconv.cpp",
                "dgNN/src/fused_gtconv/fused_gtconv_csr.cu",
                # "dgNN/src/fused_gtconv/fused_gtconv_indegree.cu",
                "dgNN/src/fused_gtconv/fused_gtconv_hyper.cu",
                "dgNN/src/fused_gtconv/fused_gtconv_tiling.cu",
                "dgNN/src/fused_gtconv/fused_gtconv_hyper_ablation.cu",
                "dgNN/src/fused_gtconv/fused_gtconv_softmax.cu",
                # "dgNN/src/fused_gtconv/fused_gtconv_subgraph.cu",
                "dgNN/src/fused_gtconv/fused_gtconv_backward.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_75", "-lineinfo"]},
            extra_link_args=["-lcurand", "-lcusparse"],
        ),
        CUDAExtension(
            "package.fused_dotgatconv",
            [
                "dgNN/src/fused_dotgatconv/fused_dotgatconv.cpp",
                "dgNN/src/fused_dotgatconv/fused_dotgatconv_tile.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_75", "-lineinfo"]},
            extra_link_args=["-lcurand", "-lcusparse"],
        ),
        CUDAExtension(
            "package.mhsddmm",
            ["dgNN/src/sddmm/mhsddmm.cc", "dgNN/src/sddmm/mhsddmm_kernel.cu"],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_75"]},
            extra_link_args=["-lcurand"],
        ),
        CUDAExtension(
            "package.mhtranspose",
            [
                "dgNN/src/csr2csc/mhtranspose.cc",
                "dgNN/src/csr2csc/mhtranspose_kernel.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_75"]},
            extra_link_args=["-lcurand"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
