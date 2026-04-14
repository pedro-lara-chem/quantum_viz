from setuptools import setup, find_packages

# Read the README file to use as the long description on PyPI/GitHub
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum_viz",
    version="2.1.0", # Matched to the version in your main.py
    author="Quantum Chemistry Visualization Team",
    description="A high-performance toolkit for parsing quantum chemistry outputs and generating 3D molecular visualizations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Assuming you used the src/ directory structure we discussed
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Ensure users have a compatible Python version
    python_requires=">=3.8",
    
    # These are critical! This tells pip to install these when someone installs your package
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "pyvista",
        "tqdm",
        "matplotlib"
    ],
    
    # This creates a terminal command for your users!
    entry_points={
        "console_scripts": [
            "quantum-viz=quantum_viz.main:main",
        ],
    },
)
