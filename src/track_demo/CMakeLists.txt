cmake_minimum_required(VERSION 3.0.2)
project(track_demo)
SET(CMAKE_BUILD_TYPE Debug)

set(CMAKE_PREFIX_PATH "/opt/ros/noetic")
message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  mavros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES track_demo
#  CATKIN_DEPENDS cv_brige geometry_msgs image_transport mavros roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  include/track_demo
  ${catkin_INCLUDE_DIRS}
)


## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/track_demo.cpp
# )
add_library(KeyboardEvent
  include/track_demo/KeyboardEvent.h
  src/lib/KeyboardEvent.cpp
)


## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
add_dependencies(KeyboardEvent ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(KeyboardEvent
  ${catkin_LIBRARIES}
)


## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(hw src/hw.cpp)
add_executable(control src/control.cpp)
add_executable(move_control src/move_control.cpp)

add_executable(move_control_con src/move_control_con.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(move_control_con ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(hw
  ${catkin_LIBRARIES}
)
target_link_libraries(control
  ${catkin_LIBRARIES}
)
target_link_libraries(move_control
  ${catkin_LIBRARIES}
)
target_link_libraries(move_control_con
  ${catkin_LIBRARIES}
)
target_link_libraries(move_control_con
  KeyboardEvent
)



add_executable(auto_control src/auto_control.cpp)
add_dependencies(auto_control ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(auto_control
  ${catkin_LIBRARIES}
)

add_executable(teleop_twist_keyboard src/teleop_twist_keyboard.cpp)
add_dependencies(teleop_twist_keyboard ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(teleop_twist_keyboard
  ${catkin_LIBRARIES}
)


add_executable(state src/state.cpp)
add_dependencies(state ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(state
  ${catkin_LIBRARIES}
)

add_executable(takeoff src/takeoff.cpp)
add_dependencies(takeoff ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(takeoff
  ${catkin_LIBRARIES}
)

add_executable(circle src/circle.cpp)
add_dependencies(circle ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(circle
  ${catkin_LIBRARIES}
)

add_executable(circle_head src/circle_head.cpp)
add_dependencies(circle_head ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(circle_head
  ${catkin_LIBRARIES}
)

add_executable(figure_eight src/figure_eight.cpp)
add_dependencies(figure_eight ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(figure_eight
  ${catkin_LIBRARIES}
)

add_executable(figure_eight_head src/figure_eight_head.cpp)
add_dependencies(figure_eight_head ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(figure_eight_head
  ${catkin_LIBRARIES}
)

add_executable(trajectory src/trajectory.cpp)
add_dependencies(trajectory ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(trajectory
  ${catkin_LIBRARIES}
)

add_executable(trajectory_realtime src/trajectory_realtime.cpp)
add_dependencies(trajectory_realtime ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(trajectory_realtime
  ${catkin_LIBRARIES}
)

add_executable(tf src/tf.cpp)
add_dependencies(tf ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tf
  ${catkin_LIBRARIES}
)

add_executable(takeoff_inplace src/takeoff_inplace.cpp)
add_dependencies(takeoff_inplace ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(takeoff_inplace
  ${catkin_LIBRARIES}
)

add_executable(auto_control_offboard src/auto_control_offboard.cpp)
add_dependencies(auto_control_offboard ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(auto_control_offboard
  ${catkin_LIBRARIES}
)

add_executable(circle_withkey src/circle_withkey.cpp)
add_dependencies(circle_withkey ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(circle_withkey
  ${catkin_LIBRARIES}
)
target_link_libraries(circle_withkey
  KeyboardEvent
)

add_executable(key_helloworld src/key_helloworld.cpp)
add_dependencies(key_helloworld ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(key_helloworld
  ${catkin_LIBRARIES}
)

add_executable(circle_points src/circle_points.cpp)
add_dependencies(circle_points ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(circle_points
  ${catkin_LIBRARIES}
)

add_executable(circle_points_key src/circle_points_key.cpp)
add_dependencies(circle_points_key ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(circle_points_key
  ${catkin_LIBRARIES}
)


add_executable(circle_points_headin_key src/circle_points_headin_key.cpp)
add_dependencies(circle_points_headin_key ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(circle_points_headin_key
  ${catkin_LIBRARIES}
)

add_executable(circle_points_headout_key src/circle_points_headout_key.cpp)
add_dependencies(circle_points_headout_key ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(circle_points_headout_key
  ${catkin_LIBRARIES}
)

add_executable(circle_points_headfollow_key src/circle_points_headfollow_key.cpp)
add_dependencies(circle_points_headfollow_key ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(circle_points_headfollow_key
  ${catkin_LIBRARIES}
)

add_executable(eight_points_key src/eight_points_key.cpp)
add_dependencies(eight_points_key ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(eight_points_key
  ${catkin_LIBRARIES}
)

add_executable(eight_points_head_key src/eight_points_head_key.cpp)
add_dependencies(eight_points_head_key ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(eight_points_head_key
  ${catkin_LIBRARIES}
)

add_executable(backhome src/backhome.cpp)
add_dependencies(backhome ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(backhome
  ${catkin_LIBRARIES}
)

add_executable(move_slow src/move_slow.cpp)
add_dependencies(move_slow ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(move_slow
  ${catkin_LIBRARIES}
)

# add_executable(px4_velocity_control_node src/px4_velocity_control_node.cpp)
# add_dependencies(px4_velocity_control_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(px4_velocity_control_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_track_demo.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
