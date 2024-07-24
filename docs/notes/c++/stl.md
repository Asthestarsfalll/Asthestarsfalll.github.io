---
title: stl标准库
hide_table_of_contents: false
tags:
  - c++
---

## priority_queue

:::tip

优先级队列，底层通过堆来实现，定义为 `priority_queue<Type, Container, Functional>`，其中 Type 代表数据类型，Container 代表容器类型，缺省状态为 vector; Functional 是比较方式，默认采用的是大顶堆 (`less<>`)，小顶堆则使用 `greater<>`。

:::

方法有 **size empty push top pop**.

对于自定义比较方式，需要重载符号 `()`，或是重载目标对象的 `>` 或 `<`。
