# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/qlt/qiulingteng/app/app-example/example-app

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qlt/qiulingteng/app/app-example/example-app/build

# Include any dependencies generated for this target.
include CMakeFiles/launch.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/launch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/launch.dir/flags.make

CMakeFiles/launch.dir/minist.cpp.o: CMakeFiles/launch.dir/flags.make
CMakeFiles/launch.dir/minist.cpp.o: ../minist.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qlt/qiulingteng/app/app-example/example-app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/launch.dir/minist.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/launch.dir/minist.cpp.o -c /home/qlt/qiulingteng/app/app-example/example-app/minist.cpp

CMakeFiles/launch.dir/minist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/launch.dir/minist.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qlt/qiulingteng/app/app-example/example-app/minist.cpp > CMakeFiles/launch.dir/minist.cpp.i

CMakeFiles/launch.dir/minist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/launch.dir/minist.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qlt/qiulingteng/app/app-example/example-app/minist.cpp -o CMakeFiles/launch.dir/minist.cpp.s

CMakeFiles/launch.dir/minist.cpp.o.requires:

.PHONY : CMakeFiles/launch.dir/minist.cpp.o.requires

CMakeFiles/launch.dir/minist.cpp.o.provides: CMakeFiles/launch.dir/minist.cpp.o.requires
	$(MAKE) -f CMakeFiles/launch.dir/build.make CMakeFiles/launch.dir/minist.cpp.o.provides.build
.PHONY : CMakeFiles/launch.dir/minist.cpp.o.provides

CMakeFiles/launch.dir/minist.cpp.o.provides.build: CMakeFiles/launch.dir/minist.cpp.o


# Object files for target launch
launch_OBJECTS = \
"CMakeFiles/launch.dir/minist.cpp.o"

# External object files for target launch
launch_EXTERNAL_OBJECTS =

launch: CMakeFiles/launch.dir/minist.cpp.o
launch: CMakeFiles/launch.dir/build.make
launch: /home/qlt/LibTorch/lib/libtorch.so
launch: /usr/local/cuda-9.0/lib64/stubs/libcuda.so
launch: /usr/local/cuda-9.0/lib64/libnvrtc.so
launch: /usr/local/cuda/lib64/libnvToolsExt.so
launch: /usr/local/cuda/lib64/libcudart_static.a
launch: /usr/lib/x86_64-linux-gnu/librt.so
launch: /home/qlt/LibTorch/lib/libc10_cuda.so
launch: /home/qlt/LibTorch/lib/libcaffe2.so
launch: /home/qlt/LibTorch/lib/libc10.so
launch: /usr/local/cuda/lib64/libcufft.so
launch: /usr/local/cuda/lib64/libcurand.so
launch: /usr/local/cuda/lib64/libcudnn.so
launch: /usr/local/cuda/lib64/libculibos.a
launch: /usr/local/cuda/lib64/libcublas.so
launch: /usr/local/cuda/lib64/libcublas_device.a
launch: /usr/local/cuda/lib64/libcudart_static.a
launch: /usr/lib/x86_64-linux-gnu/librt.so
launch: CMakeFiles/launch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qlt/qiulingteng/app/app-example/example-app/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable launch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/launch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/launch.dir/build: launch

.PHONY : CMakeFiles/launch.dir/build

CMakeFiles/launch.dir/requires: CMakeFiles/launch.dir/minist.cpp.o.requires

.PHONY : CMakeFiles/launch.dir/requires

CMakeFiles/launch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/launch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/launch.dir/clean

CMakeFiles/launch.dir/depend:
	cd /home/qlt/qiulingteng/app/app-example/example-app/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qlt/qiulingteng/app/app-example/example-app /home/qlt/qiulingteng/app/app-example/example-app /home/qlt/qiulingteng/app/app-example/example-app/build /home/qlt/qiulingteng/app/app-example/example-app/build /home/qlt/qiulingteng/app/app-example/example-app/build/CMakeFiles/launch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/launch.dir/depend

