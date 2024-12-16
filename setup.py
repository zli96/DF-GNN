import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DFGNN",
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
    # package_dir={"": "DFGNN"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    ext_modules=[
        CUDAExtension(
            "fused_gatconv",
            [
                "DFGNN/src/fused_gatconv/fused_gatconv.cpp",
                "DFGNN/src/fused_gatconv/fused_gatconv_kernel.cu",
                "DFGNN/src/fused_gatconv/fused_gatconv_hyper.cu",
                "DFGNN/src/fused_gatconv/fused_gatconv_hyper_recompute.cu",
                "DFGNN/src/fused_gatconv/fused_gatconv_hyper_v2.cu",
                "DFGNN/src/fused_gatconv/fused_gatconv_softmax.cu",
                "DFGNN/src/fused_gatconv/fused_gatconv_hyper_ablation.cu",
                "DFGNN/src/fused_gatconv/fused_gatconv_tiling.cu",
                "DFGNN/src/fused_gatconv/fused_gatconv_softmax_gm.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_80", "-lineinfo"]},
            extra_link_args=["-lcurand"],
        ),
        CUDAExtension(
            "fused_gtconv",
            [
                "DFGNN/src/fused_gtconv/fused_gtconv.cpp",
                "DFGNN/src/fused_gtconv/fused_gtconv_csr.cu",
                # "DFGNN/src/fused_gtconv/fused_gtconv_indegree.cu",
                "DFGNN/src/fused_gtconv/fused_gtconv_hyper.cu",
                "DFGNN/src/fused_gtconv/fused_gtconv_tiling.cu",
                "DFGNN/src/fused_gtconv/fused_gtconv_hyper_ablation.cu",
                "DFGNN/src/fused_gtconv/fused_gtconv_softmax.cu",
                "DFGNN/src/fused_gtconv/fused_gtconv_softmax_gm.cu",
                # "DFGNN/src/fused_gtconv/fused_gtconv_subgraph.cu",
                "DFGNN/src/fused_gtconv/fused_gtconv_backward.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_80", "-lineinfo"]},
            extra_link_args=["-lcurand", "-lcusparse"],
            include_dirs=["DFGNN/src/spmm/spmm.cuh", "DFGNN/src/sddmm/sddmm.cuh"],
        ),
        CUDAExtension(
            "fused_dotgatconv",
            [
                "DFGNN/src/fused_dotgatconv/fused_dotgatconv.cpp",
                "DFGNN/src/fused_dotgatconv/fused_dotgatconv_tile.cu",
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_80", "-lineinfo"]},
            extra_link_args=["-lcurand", "-lcusparse"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
