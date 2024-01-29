import os
import sys
import subprocess
from setuptools import setup, Extension
from Cython.Build import cythonize

# Get a list of all files in src
cwd = os.getcwd()
cwd = cwd+"/src"
src_files = os.popen("ls " + cwd).read()
src_files = src_files.split("\n")

# Pull the .cpp and .h files
cppfiles = []
for f in src_files:
    if len(f.split("."))>1:
        end = f.split(".")[1]
        if end == "h" or end == "cpp":
            cppfiles.append("src/"+f)


# Copy object files to openbtpt/bin
exec_list = []            
for f in src_files:
    if len(f.split("."))>1:
        end = f.split(".")[1]
        #if end in ["o","lo","la"]:
        #    os.system("cp " + cwd +"/" + f + " " + os.getcwd() + "/openbtpy/"+f)
    else:
        if "openbt" in f:
            os.system("cp " + cwd +"/" + f + " " + os.getcwd() + "/openbtpy/"+f)
            #exec_list.append(cwd + "/" + f)
            exec_list.append(f)         


# Clone Eigen repository
eigen_repo_url = "https://gitlab.com/libeigen/eigen.git"
eigen_clone_path = "src/Eigen"
eigen_branch = "3.4.0"

subprocess.run(["git", "clone", "--depth=1","-b", eigen_branch, eigen_repo_url, eigen_clone_path])

#os.chdir("src/Eigen")
#subprocess.run(["cd", "src/Eigen", "&& git checkout 3.4.0"])
#os.chdir("..")

os.chdir("src")

# Run the makefile
subprocess.run(["make",cwd])

# Create extensions from list
cppext = [
    Extension('openbtcpp',sources = cppfiles)
]

# Setup
setup(
    name='openbtpy',
    version='0.1',
    packages=["openbtpy"],
    package_data={'openbtpy': ['*.o',"*.lo","*.la",".libs/*"]+exec_list},  # Include compiled shared libraries
    zip_safe=False,
)

# Setup step
# setup(
#     name="openbtpy",
#     version="1.0",
#     packages=["openbtpy"],
#     ext_modules=cythonize(cppext),
#     include_dirs=["/src"],
#     zip_safe = False
# )


# THIS RUNS BUT THERE IS A COMPILATION ERROR ON PIPS END

# def get_cpp_sources(folder, include_headers=False):
#     """Find all C/C++ source files in the `folder` directory."""
#     allowed_extensions = [".c", ".C", ".cc", ".cpp", ".cxx", ".c++"]
#     if include_headers:
#         allowed_extensions.extend([".h", ".hpp"])
#     sources = []
#     for root, dirs, files in os.walk(folder):
#         for name in files:
#             ext = os.path.splitext(name)[1]
#             if ext in allowed_extensions:
#                 sources.append(os.path.join(root, name))
#     return sources


# setup(
#     name="openbtpy",
#     version="1.0",
#     packages=["openbtpy"],
#     ext_modules=[
#         Extension(
#             "openbtpy",
#             include_dirs=[cwd],
#             sources=get_cpp_sources(cwd, include_headers=False),
#         ),
#     ],
#     zip_safe = False
# )

