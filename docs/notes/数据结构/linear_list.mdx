---
title: 线性表
tags: [data structure]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';

import TabItem from '@theme/TabItem';

import styles from '/src/css/tab.css';

## 定义和基本操作

:::tip 定义

具有**相同数据类型**的 $n(n\geq 0)$ 个数据元素的有限序列。

:::

以此我们可以得到以下特性

1. 表中元素有限
2. 逻辑上元素具有顺序和先后次序
3. 所有元素占据的存储空间相同
4. 抽象性，仅讨论元素间的逻辑关系，而不必考虑元素表示什么

:::tip 基本操作

初始化、表长、按值查找、按位查找、插入、删除、输出、判空、销毁。

:::

## 广义表

:::tip

广义表是线性表的一种扩展延伸。相对于线性表，广义表最大的特点在于其元素既可以是一个确定的类型，同时也可以是另一个有不定数量的元素组成的表（广义表）。 不难看出从广义表的定义是递归的。广义表是线性表的递归数据结构。

:::

**广义表的两个操作**

1. $head$：从表头取第一个元素；
2. $tail$：除去表头之外，由其余元素构成的 **广义表**。

## 顺序表示

:::tip 顺序表

逻辑上相邻的元素在物理地址上也相邻。

:::

实现上可以使用数组，因此天然地实现了按位查找的操作，其中表长可通过 `sizeof` 除以单个元素的 `size` 实现，但是一般来说都是在结构体中直接定义当前长度。

:::caution

顺序表下标从 $1$ 开始，而数组中是从 $0$ 开始的。

顺序表的下标可能表示结构体当中的当前长度，在逻辑上表示表中有多少元素，实际使用时需要减去 $1$.

:::

**静态分配**：有溢出风险

**动态分配**：不必为表一次申请所有空间，若占满才申请一块更大的空间

:::caution 随机存取和顺序存取

随机存取就是直接存取，可以通过下标直接访问到元素的位置，与存储位置无关；

顺序存取，不能通过下标访问，在存取第 $N$ 个数据时，必须先访问前 $(N-1)$ 个数据，如链表。

:::

顺序表就属于随机存取。

## 链式表示

:::tip 链表

逻辑上相邻的元素在物理地址上不相邻。

:::

每个结点除了存储数据外，还要额外存储一个（或多个）指针，对于单链表，其指针指向后继结点。

通常来说，一般使用一个独立的 `头结点` 来表示链表，头结点中可以不存放任何信息，也可以记录表长等信息，其指针域指向第一个元素。

:::info 引入头结点的优点

对链表第一个位置的处理与其他位置一致；

使空表和非空表的处理得到统一。

:::

## 双链表

:::tip 双链表

双链表中结点的指针域指向结点的前驱和后继结点，查找较为方便，但是存储空间会稍微大点。

:::

## 循环链表

:::tip 循环链表

即将链表的最后一个结点连接至第一个结点，逻辑上是一个环，但是代码实现中还是需要一个起点。

为了保持一致，空表的头结点指向自身，则其判空条件就是头结点的前驱后继都指向自己；因此一个单循环列表最末尾结点的指针应该是指向头结点而不是逻辑上的第一个结点。

:::

根据结点指针域的不同，可分为循环单链表和循环双链表。

:::caution 可恶的循环单链表

注意带尾指针循环单链表，其访问后继时间复杂为 $O(1)$，但是访问前驱的时间复杂度为 $O(n)$，如删除最后一个元素，由于需要与前驱相连，时间复杂度为 $O(n)$.

:::

## 静态链表

:::tip 静态链表

静态链表借助数组来描述线性表的链式存储结构，每个结点中同样有数据域和指针域，但是这里的指针表示相对地址，又称为 `游标`.

:::

静态链表没有但链表使用方便，但是适用于一些没有指针的语言中。

## 时间复杂度

|  操作    |  顺序表    |   链表   |  备注    |
|:-----|:-----|:-----|:-----|
|  存取    |  $O(1)$     |  $O(1)$     |      |
|  按位查找    |  $O(1)$     |  $O(n)$     |      |
|  按值查找    |  $O(n)$     |  $O(n)$     |  若顺序表有序，折半查找时间复杂度 $O(\log_{2}n)$     |
|  插入    |  $O(n)$     |  $O(1)$     | 插入平均移动一半的元素     |
|  删除    |  $O(n)$     |  $O(1)$     | 删除平均移动一半的元素     |

通过上表，可以看出若插入删除操作频繁，可选用链表，若按位访问频繁，可选用顺序表。

## 顺序表实现

顺序表的定义

<Tabs groupId="list">
<TabItem value="C" >

```c
#define MaxSize 50
typedef struct {
  int data[MaxSize];
  int length;
} SqList;
```

</TabItem>
<TabItem value="C++" >

```cpp
#define MaxSize 50
template <typename ElementType>
class SqList {
  ElementType data[MaxSize];
  int length;
};
```

</TabItem>
</Tabs>

按位删除，需要将后面的元素移动到前面的位置，索引从 0 开始，C++ 代码中则可以将其作为类方法。

<Tabs groupId="list">
<TabItem value="C" >

```c
int del(SqList *list, int idx) {
  if (idx >= list->length)
    return 0;
  for (int i = idx; i < list->length - 1; i++) {
    list->data[i] = list->data[i + 1];
  }
  list->length -= 1;
  return 1;
}
```

</TabItem>
<TabItem value="C++" >

```cpp
int del(int idx) {
  if (idx >= length)
    return 0;
  for (int i = idx; i < length - 1; i++) {
    data[i] = data[i + 1];
  }
  length -= 1;
  return 1;
}
```

</TabItem>
</Tabs>

由于顺序表的实现较为简单，不再继续写了。

## 链表实现

也不太难，有空在写。

## std::vector

`std::vector` 是 C++ 提供的动态容器库，其底层使用连续的内存块来存储元素，只在容量不够时进行扩充，其分配的内存可以使用 `capacity()` 进行查询，并且使用 `shrink_to_fit()` 释放空闲的内存。

常用成员方法有

|  方法    |  作用    |
|:-----|:-----|
|   size   |  返回元素的数量    |
|   empty   |   检查 vector 是否为空。如果为空，返回 true；否则，返回 false   |
|   clear   |   清空所有元素   |
|   push_back(element)   |     将元素添加到 vector 的末尾 |
|   pop_back   |   删除最后一个元素   |
|   insert(iterator, element)   |   在指定迭代器位置之前插入元素   |
|   erase(iterator or start_iterator, end_iterator)   |   删除指定范围内的元素   |
|  front    |  访问第一个元素    |
|   back   |   访问 vector 的最后一个元素   |
| data  | 返回指向第一个元素的指针|
|   begin   |    返回指向 vector 开头的迭代器   |
|   end  |    返回指向 vector 末尾的迭代器  |
|   rbegin  |    返回指向 vector 末尾的逆向迭代器  |
|   rend  |    返回指向 vector 开头的逆向迭代器  |
|  resize(new_size)    |   调整 vector 的大小为 new_size.如果 new_size 大于当前大小，则在末尾添加默认构造的元素；如果 new_size 小于当前大小，则删除超出的元素   |
|  reserve(new_capacity)    |   分配足够的内存以容纳至少 new_capacity 个元素，但不改变 vector 的大小   |
|  capacity    |  返回 vector 当前能够容纳的元素数量    |
|   at(index)   |  访问指定索引处的元素，如果索引超出范围，会抛出异常    |
|   operator[] (index)   |   通过索引访问元素，如果索引超出范围，行为未定义   |
