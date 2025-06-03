#pragma once
#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
string wchar_t2string(const wchar_t *wchar);
wchar_t *string2wchar_t(const string &str);
vector<string> listFilesInDirectory(string directoryName);
