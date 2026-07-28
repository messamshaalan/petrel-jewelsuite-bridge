from setuptools import setup, find_packages

setup(
    name="petrel-jewel-bridge",
    version="0.1.0",
    description=(
        "Bi-directional 3D grid format converter between Petrel and JewelSuite.  "
        "Solves the three critical interoperability gaps: JewelGrid export, pillar-grid "
        "bi-directional support, and tetrahedral mesh → Petrel voxel conversion."
    ),
    author="G&G Team",
    long_description=open("README.md", encoding="utf-8").read() if __import__("pathlib").Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "resfo>=4.0.0",
        "resqpy>=4.0.0",
        "pyvista>=0.42.0",
        "trimesh>=4.0.0",
        "meshio>=5.3.0",
        "h5py>=3.9.0",
        "shapely>=2.0.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "loguru>=0.7.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.9",
    ],
    extras_require={
        "xtgeo":  ["xtgeo>=3.4.0"],
        "opm":    ["opm-common>=2023.10"],
        "dev":    ["pytest>=7.4.0", "pytest-cov>=4.1.0", "httpx>=0.27.0"],
        "nb":     ["jupyterlab>=4.0.0", "matplotlib>=3.7.0", "plotly>=5.17.0"],
    },
    entry_points={
        "console_scripts": [
            "petrel-jewel=cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
