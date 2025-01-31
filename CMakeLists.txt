cmake_minimum_required(VERSION 3.28.3)

project(NEoS)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
)

add_executable(NEoS ${SOURCES})