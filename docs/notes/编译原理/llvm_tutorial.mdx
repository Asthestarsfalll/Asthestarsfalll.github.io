---
title: llvm教程
tags: [Compilation principle]
hide_table_of_contents: false
---

https://llvm.org/docs/tutorial/index.html

## 目标

实现一个简单的编程语言——Kaleidoscope，支持定义函数、条件语句、数学运算等。并且扩展支持if/then/else的结构、for循环、用户自定义操作符、JIT、调试等。

为了经可能简单，Kaleidoscope中的数据类型只有float64——double.因此并不需要类型声明。

同时允许Kaleidoscope调用标准库函数，LLVM JIT可以轻易做到这点：

```c++
extern sin(arg);
extern cos(arg);
extern atan2(arg1 arg2);

atan2(sin(.4), cos(42))
```

**学习目标**

掌握使用llvm实现编程语言的大致流程

了解llvm的使用方法

## 词法分析器

词法分析器是实现一个编程语言应该完成的第一步，这里的实现没啥好说的，或许使用lex实现更好吧。

## Parser

本节所实现的Parser会结合递归下降法和算符优先分析，利用上文中的lexer解析输入的token，并且返回AST——抽象语法树。

Kaleidoscope 中的抽象语法树如下：

```c++
/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() = default;
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(const std::string &Name) : Name(Name) {}
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
    : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
    : Callee(Callee), Args(std::move(Args)) {}
};
```