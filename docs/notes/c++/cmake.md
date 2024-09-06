---
title: CMake 教程
hide_table_of_contents: false
tags:
  - c++
---

## 概述

CMake 是一个跨平台的项目构建工具，相较于 makefile，其更加简洁，并且允许制定更复杂的规则。CMake 可以根据编译平台，自动生成本地 Makefile 文件，最后用户只需要 make 编译即可，因此 CMake 可以看作一种构建系统（CMake）的构建系统。

总结，其优点为：

1. 跨平台；
2. 便于管理大型项目；
3. 简化编译构建过程；
4. 可扩展性更强。

## 使用

CMake 支持大写、小写、混合大小写的命令。如果在编写 `CMakeLists.txt` 文件时使用的工具有对应的命令提示，那么大小写随缘即可，不要太过在意。

### 注释

使用 `#` 进行行注释，可以放在任何地方；使用 `#[[  ]]` 进行块注释。

### 基础 CMake 命令

`cmake_minimum_required(VERSION 3.0)`，制定使用的 cmake 的最低版本，非必须。

`project`：定义工程的名称，可以指定多个参数，如下：

```cmake
project(<PROJECT-NAME> [<language-name>...])
project(<PROJECT-NAME>
       [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
       [DESCRIPTION <project-description-string>]
       [HOMEPAGE_URL <url-string>]
       [LANGUAGES <language-name>...])
```

`add_executable`：定义工程生成一个可执行程序，第一个参数是可执行程序名称，后面的参数是所有源文件的名称（使用空格或者分号间隔）

```cmake
# 样式1
add_executable(app add.c div.c main.c mult.c sub.c)
# 样式2
add_executable(app add.c;div.c;main.c;mult.c;sub.c)
```

### Set 命令

```cmake
SET(VAR [VALUE] [CACHE TYPE DOCSTRING [FORCE]])
```

将某些值以 **字符串** 的形式存储到对应的变量名中，使用 `${}` 进行调用。

当需要指定使用的 c++ 标准时，在编译时有三种方式：

1. 在 g++ 编译命令中添加 `-std=c++11`；
2. 在 `CMakeLists.txt` 中通过 set 命令设置：`set(CMAKE_CXX_STANDARD)`
3. 在使用 CMake 时指定：`cmake .. -DCMAKE_CXX_STANDARD=11` 。

指定可执行文件输出路径同理，对应宏 `EXECUTABLE_OUTPUT_PATH`，cmake 会自动创建该目录。

### 搜索文件

对于大型项目，将所有的文件手写到 add_excutable 中是不现实的，因此需要某些命令帮助我们自动获取的文件。

`aux_source_directory`，第一个参数为目录，第二个参数为存储的变量名，这回查找某个文件夹下的所有源文件。

`file`: 第一个参数是两种模式，`GLOB` 和 `GLOB_RECURSE`，后者递归搜索指定目录；第二个参数是变量名；第三个参数是目录以及文件类型，形式为 `…/src/…/*.cpp`。

:::caution

file 命令得到的文件路径是绝对路径。

:::

### 头文件

使用命令 `include_directories` 来指定头文件查找目录。

### 动态库与静态库

```cmake
add_library(库名称 STATIC 源文件1 [源文件2] …)
```

在 linux 中，所生成的静态库名称为 `lib` + `库名称` + `.a`；若要指定输出静态库的位置，可以使用宏 `LIBRARY_OUTPUT_PATH`。

```
add_library(库名称 SHARED 源文件1 [源文件2] …)
```

动态库则使用 `SHARED`，生成文件后缀为 `.so`；若要指定输出位置，除了使用上述宏，由于在 Linux 下生成的动态库默认是有执行权限的，也可以使用 `EXECUTABLE_OUTPUT_PATH`。

对于动态库和静态库的链接，可以使用

```cmake
link_libraries(<static lib> [<static lib>…])
```

这里的名字可以是全名 `libxxx.a/so` 或者是掐头去尾的名字，并且可以链接多个库。

:::tip

如果库不是系统提供的（如自己制作或第三方提供），可能会出现找不到库的情况，此时需要将库的路径指定出来：

```cmake
linx_directories(<lib path>)
```

:::

虽然该命令可以链接动态库和静态库，但是使用 `target_` 前缀的命令来链接动库和添加路径会更好。

### 日志

使用 `message` 命令来显示消息：

```cmake
message([STATUS|WARNING|AUTHOR_WARNING|FATAL_ERROR|SEND_ERROR] "message to display" …)
```

其中第一个参数不选，则表示重要消息，STATUS 非重要信息，WARNING 警告，啊 AUTHOR_WARNING 开发者错误信息，SEND_ERROR，错误继续执行，但是跳过生成，FATAL_ERROR，CMake 错误，终止所有处理过程。

其中只有 `STATUS` 消息输出在 stdout，其他都在 stderr。

### 变量操作

拼接，形式为：

```cmake
set(变量名1 ${变量名1} ${变量名2} …)
```

list 命令也可以实现字符串拼接：

```cmake
list(APPEND <list> [<element> …])
```

使用模式 `APPEND` 来使用该功能。

```cmake
list(REMOVE_ITEM <list> <value> [<value> …])
```

使用 `REMOVE_ITEM` 操作来一处列表中的元素，如移除 file 命令得到的文件。

:::tip

list 的其他操作还有 `LENGTH `，`GET`，`JOIN`，`FIND`，`INSERT`，`PREPEND`，`POP_BACK`，`POP_FRONT`，`REMOVE_AT`，`REMOVE_DUPLICATES`，`REVERSE` 和 `SORT`。

其中排序：

```cmake
list (SORT <list> [COMPARE <compare>] [CASE <case>] [ORDER <order>])
```

可以指定排序方法，`STRING`，`FILE_BASENAME` 和 `NATURAL`，CASE 表明是否大小写敏感，`SENSTIVE`， `INSENSITIVE`，ORDER 表示升序或降序，`ASCENDING`，`DESENDING`。

:::

### 宏定义

使用 `add_definitions` 来定义宏，如在代码中使用 `#ifdef DEBUG`，则在 CMAKE 中定义 DEBUG 宏来开启调试，

## CMake 宏

CMAKE_CXX_STANDARD

CMAKE_CURRENT_SOURCE_DIR

CMAKE_CURRENT_BINARY_DIR

EXECUTABLE_OUTPUT_PATH

LIBRARY_OUTPUT_PATH

PROJECT_SOURCE_DIR

PROJECT_BINARY_DIR

PROJECT_NAME

CMAKE_BINARY_DIR

## 嵌套

在大型项目中，可以给每个源代码目录都添加一个 CMakeLists.txt （头文件目录不需要）即可。

对于这样的嵌套关系，根节点中的变量全局有效，父节点的变量也可以在子节点中使用。使用命令：

```cmake
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
```

`source_dir` 指定子目录位置。

## 控制流程

使用 `if` 进行控制，其中表达式可以为常量、变量或字符串，如 1、ON、YES、TRUE、Y、非零值以及非空字符串时，返回 True。

支持逻辑判断，有 `NOT`、`AND` 和 `OR` 等。

比较有 `LESS`、`GREATER`、`EQUAL`、`LESS_EQUAL` 和 `GREATER_EQUAL`。

特别地，对于字符串的比较，添加 `STR` 前缀即可。

`EXISTS` 判断文件或目录是否存在，`IS_DIREACTORY` 判断是否是目录（参数为绝对路径），`IS_SYMLINK` 判断是否是软连接（参数为绝对路径），`is_ABSOLUTE` 判断是否是绝对路径。

判断某个元素是否在列表中

```cmake
if(<variable|string> IN_LIST <variable>)
```

比较两个路径是否相等

```cmake
if(<variable|string> PATH_EQUAL <variable|string>)
```

## 循环

```cmake
foreach(<loop_var> RANGE <stop>)
```

```cmake
while(<condition>)
    <commands>
endwhile()
```

## 参考

[CMake 保姆级教程（上）](https://subingwen.cn/cmake/CMake-primer/)

[CMake 保姆级教程（下）](https://subingwen.cn/cmake/CMake-advanced/)
