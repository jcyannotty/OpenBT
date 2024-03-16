import os
import sys
import subprocess
from setuptools import setup, Extension
import platform
import distutils
#from Cython.Build import cythonize

# Get a list of all files in src
startwd = os.getcwd()
cwd = startwd+"/src"
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
        if end in ["o","lo","la"]:
            os.system("cp " + cwd +"/" + f + " " + os.getcwd() + "/openbtmixing/"+f)
            exec_list.append(f)
    else:
        if "openbt" in f:
            os.system("cp " + cwd +"/" + f + " " + os.getcwd() + "/openbtmixing/"+f)
            #exec_list.append(cwd + "/" + f)
            exec_list.append(f)         


# Get libraries
lib_list = []
lib_files = os.popen("ls " + cwd + "/.libs/").read()
lib_files = lib_files.split("\n")
os.system("mkdir " + startwd + "/openbtmixing/.libs")
for lb in lib_files:
    os.system("cp " + cwd + "/.libs/"+lb+ " " + startwd + "/openbtmixing/.libs")
    #lib_list.append(cwd+"/.libs/"+lb)
    lib_list.append(startwd+"/openbtmixing/.libs/"+lb) 


# Setup
dist_name = distutils.util.get_platform()
dist_name = dist_name.replace("-","_")
dist_name = dist_name.replace(".","_")

if "linux_x86_64" in dist_name:
    dist_name = "manylinux2014_x86_64"


if "macosx" in dist_name:
    dist_name = "macosx_10_9_x86_64"
    #dist_name = "macosx_10_9_arm64"

setup(
    name='openbtmixing',
    version='1.0.1',
    packages=["openbtmixing"],
    package_data={'openbtmixing': ['*.o',"*.lo","*.la",".libs/*"]+exec_list+lib_list},  # Include compiled shared libraries
    zip_safe=False,
    options={'bdist_wheel':{'plat_name':dist_name}}
)

# Setup step

# Create extensions from list
# cppext = [
#     Extension('openbtcpp',sources = cppfiles)
# ]

# setup(
#     name="openbtmixing",
#     version="1.0",
#     packages=["openbtmixing"],
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
#     name="openbtmixing",
#     version="1.0",
#     packages=["openbtmixing"],
#     ext_modules=[
#         Extension(
#             "openbtmixing",
#             include_dirs=[cwd],
#             sources=get_cpp_sources(cwd, include_headers=False),
#         ),
#     ],
#     zip_safe = False
# )

