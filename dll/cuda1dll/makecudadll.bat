call C:\"Program Files (x86)"\"Microsoft Visual Studio 14.0"\VC\vcvarsall.bat
nvcc -o cuda1dll.dll --shared kernel.cu main.cpp

