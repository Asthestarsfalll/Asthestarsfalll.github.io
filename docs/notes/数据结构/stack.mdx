---
title: 栈
tags: [data structure]
hide_table_of_contents: false
---

## 定义

:::tip 栈

只允许在一端进行插入或删除操作的线性表。

:::

:::info 栈的数学性质

$n$ 个不同的元素进栈，出栈序列有 $\frac{1}{n+1}C_{2n}^n$ 个。

:::

同样，栈有顺序栈和链栈。

## 功能

由于栈先进后出的性质，其主要功能在管理数据的存储和访问方面，如：

1. **函数调用**，每当一个函数被调用时，当前函数的执行状态（例如局部变量、返回地址等）会被保存在栈中。这样，在函数执行完毕后，程序可以从栈中恢复到调用该函数之前的状态；
2. **表达式求值**，通过后缀表达式（算术表达式所对应二叉树的后序遍历）进行求值；具体方法为按照顺序将后缀表达式入栈，遇到操作符时弹出对应的操作数（需要看是几元操作符），在将计算完的结果入栈，直到遍历完后缀表达式，此时栈中应只剩一个元素，为最终结果；
3. **内存管理**：用于管理内存分配和释放。例如，当定义局部变量时，内存会自动分配在栈上；当函数执行完毕时，这些局部变量所占用的内存会自动释放；
4. **撤销操作**：将操作的状态保存在栈中。若撤销操作，系统可以从栈中取出最近的状态并恢复到该状态；
5. **缓冲区管理**：计算机图形学中，栈可以用于管理绘图操作的缓冲区，将绘制的图像按照绘制顺序依次压入栈中，并在需要时从栈中弹出进行显示。
6. **符号匹配**：判断字符串中的成对符号是否匹配，遍历字符串，入栈左符号，若栈顶匹配到成对右符号，则进行弹出栈顶；知道遍历结束，栈空说明符号匹配;
7. **进制转换**：取模压入栈中，直至数为 0, 再依次弹出即可；
8. **中缀表达式转后缀表达式**：见 [此](#中缀表达式转后缀表达式).

## 进制转换

python 代码

```python
def decimal_to_binary(decimal, mod=2):
    stack = []

    while decimal > 0:
        remainder = decimal % mod
        stack.append(remainder)
        decimal = decimal // mod

    converted = ""
    while len(stack) > 0:
        converted += str(stack.pop())

    return converted
```

## 中缀表达式转后缀表达式

此时栈用来存放一些不能确定的符号，高亮处需要判断栈顶元素和当前运算符的优先级关系，需要按序弹出所有优先级大于等于当前运算符的元素到列表中，再将当前运算符入栈。

```python
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    stack = []
    postfix = []

    for char in expression:
	    # 是操作数添加到列表中
        if char.isdigit():
            postfix.append(char)
        elif char == '(':
            stack.append('(')
        elif char == ')':
            while stack and stack[-1] != '(':
                postfix.append(stack.pop())
            stack.pop()  # 弹出左括号
        else:  # 运算符
			# Im-start
            while stack and stack[-1] != '(' and precedence[char] <= precedence.get(stack[-1], 0):
                postfix.append(stack.pop())
            stack.append(char)
			# Im-end
	# 若栈中还有符号则全部出栈
    while stack:
        postfix.append(stack.pop())

    return ''.join(postfix)

```

## 共享栈

由于栈的底端不变，因此可让两个顺序栈共享一个一维数组，两个底端分别在数据的两侧。

共享栈的判满需要一定注意，以 $0$ 为起点，$n-1$ 为终点，两个栈的起始点便是 $-1$ 和 $n$.当共享栈满时，两个栈的栈顶应该相邻，则栈顶坐标应该满足：

$$
s_{2} - s_{1} = 1
$$
