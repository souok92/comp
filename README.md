# comp

## connect
- 베라소닉스 -> 파이썬 -> 무조코 워크플로우
  1. saveFrameL12_5withButtons_modified.m
  
     saveImg2.m이랑 같이 사용, 버튼 2개 추가.
     
     연속저장버튼은 버퍼에 지속적으로 프레임 저장하는 버튼 / 6개짜리 버튼은 버튼 누르면 분류까지 해서 저장
     
     나머지는 자동화해뒀으니 넘어가고
     
     바탕화면에 Gest, Buffer 폴더 2개가 생성될거고 Buffer는 파이썬 연동용, Gest는 분류 데이터 획득용

  2. LiveMujoco.py
     기존의 폴더에서 랜덤으로 읽어오는 사진을 Buffer 폴더에서 읽어오는 방식으로 진행

## ROS2 기반 워크플로우
이건 추후 업데이트 하도록 할게용


--- stderr: dg3f_m_driver                                        
CMake Error at CMakeLists.txt:10 (find_package):
  By not providing "Findhardware_interface.cmake" in CMAKE_MODULE_PATH this
  project has asked CMake to find a package configuration file provided by
  "hardware_interface", but CMake did not find one.

  Could not find a package configuration file provided by
  "hardware_interface" with any of the following names:

    hardware_interfaceConfig.cmake
    hardware_interface-config.cmake

  Add the installation prefix of "hardware_interface" to CMAKE_PREFIX_PATH or
  set "hardware_interface_DIR" to a directory containing one of the above
  files.  If "hardware_interface" provides a separate development package or
  SDK, be sure it has been installed.


gmake: *** [Makefile:431: cmake_check_build_system] Error 1
---
Failed   <<< dg3f_m_driver [4.17s, exited with code 2]
Aborted  <<< dg5f_driver [4.14s]
Aborted  <<< dg_msgs [4.84s]                                             
Aborted  <<< dg_description [3.09s]      
