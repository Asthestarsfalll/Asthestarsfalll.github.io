---
title: skynet源码阅读
hide_table_of_contents: false
tags:
  - c++
---

## 介绍

它是一个轻量级游戏服务器框架，但也不仅仅用于游戏。

轻量级体现在：

实现了 actor 模型，以及相关的脚手架（工具集）：actor 间数据共享机制以及 c 服务扩展机制。

实现了服务器框架的基础组件。实现了 reactor 并发网络库；并提供了大量连接的接入方案；基于自身网络库，实现了常用的数据库驱动（异步连接方案），并融合了 lua 数据结构；实现了网关服务；时间轮用于处理定时消息。

Skynet 抽象了 actor 并发模型，用户层抽象进程；sknet 通过消息的方式共享内存；通过消息驱动 actor 运行。

Skynet 的 actor 模型使用 **lua 虚拟**，lua 虚拟机非常小（只有几十 kb），代价不高；每个 actor 对应一个 lua 虚拟机；系统中不能启动过多的进程就是因为资源受限，lua 虚拟机占用的资源很少，可以开启很多个，这就能抽象出很多个用户层的进程。Lua 虚拟机可以共享一些数据，比如常量，达到资源复用。

抽象进程而不抽象线程的原因在于进程有独立的工作空间，隔离的运行环境。

Sknet 的所有 actor 都是对等的，通过公平调度实现。

### actor 模型

Actor 模型是一种并发计算的模型，由 Carl Hewitt 在 1973 年提出。它是一种计算模型，用于设计和构建分布式系统。在这个模型中，系统由一系列独立的实体称为 "actors" 组成，每个 actor 都有自己的状态和行为，并且只能通过消息传递与其他 actors 进行通信。

以下是 Actor 模型的一些关键特点：

1. **封装性**：每个 actor 都有自己的状态和行为，它们是封装的，外部不能直接访问或修改 actor 的状态。

2. **并发性**：actors 可以并发执行，它们之间通过消息传递进行通信，而不是共享内存。

3. **消息传递**：actors 之间的通信是通过发送和接收消息完成的。消息是异步的，发送消息的 actor 不需要等待接收者的响应。

4. **无共享状态**：actors 之间不共享状态，每个 actor 都有自己的状态，这有助于避免并发编程中常见的数据竞争和同步问题。

5. **故障隔离**：由于 actors 之间不共享状态，一个 actor 的故障不会影响到其他 actor 的状态，这有助于提高系统的稳定性。

6. **可扩展性**：由于 actors 是独立的，可以很容易地在多个处理器或机器上分布执行，从而提高系统的可扩展性。

## 使用

使用 config 文件来定义行为，如：

## 源代码

### 目录结构

```
.
├── 3rd  # 提供lua语言支持、 jemalloc（内存管理模块）、md5加密等
├── cservice
├── examples
├── luaclib
├── lualib # 调用lua服务的辅助函数
├── lualib-src # 提供C层级的api调用
├── service # lua层服务
├── service-src # 依附于skynet核心模块的c服务
├── skynet-src # 核心代码
└── test
```

调用关系为 service, lualib -> service-src, lualib-src -> skynet-src -> 3rd。

### 基本数据结构

```c
//  skynet_module.h
struct skynet_module {
	const char * name;                       //  C服务的文件名
	void * module;                               // 访问so库的dl句柄，通过dlopen获取
	skynet_dl_create create;
	skynet_dl_init init;
	skynet_dl_release release;
	skynet_dl_signal signal;
};
```

后面使用函数指针指向绑定 so 库中的对应函数，这里使用了 typedef 来定义：

```c
typedef void * (*skynet_dl_create)(void);
typedef int (*skynet_dl_init)(void * inst, struct skynet_context *, const char * parm);
typedef void (*skynet_dl_release)(void * inst);
typedef void (*skynet_dl_signal)(void * inst, int signal);
```

```c
// skynet_module.c
#define MAX_MODULE_TYPE 32

struct modules {
	int count;   // modules数量
	struct spinlock lock;  // 自旋锁
	const char * path;  // 配置表中的cpath指定路径
	struct skynet_module m[MAX_MODULE_TYPE]; 
};

static struct modules * M = NULL;
```

使用 `static` 声明为全局变量。

:::tip

一个符合规范的 skynet c 服务，应当包含 create，init，signal 和 release 四个接口，在该 c 服务编译成 so 库以后，在程序中动态加载到 skynet_module 列表中，这里通过 dlopen 函数来获取 so 库的访问句柄，并通过 dlsym 将 so 库中对应的函数绑定到函数指针中。

dlopen 函数，本质是将 so 库加载内存中，并返回一个可以访问该内存块的句柄，dlsym，则是通过该句柄和指定一个函数名，到内存中找到指定函数。

一个 C 服务，定义以上四个接口时，一定要以文件名作为前缀，然后通过下划线和对应函数连接起来，因为 skynet 加载的时候，就是通过这种方式去寻找对应函数的地址的。

:::

## 参考

[Skynet 设计原理](https://blog.csdn.net/Long_xu/article/details/128274169)

[skynet 源码剖析](https://zhuanlan.zhihu.com/p/698153760)
