---
title: Music21
description: music21是一个由MIT开发的功能强大的计算音乐学分析Python库。相比于pretty_midi库只能处理MIDI文件，music21可以处理包括MusicXML，MIDI，abc等多种格式的音乐文件，并可以从零开始构建音乐文件或对音乐进行分析。
authors: [Asthestarsfalll]
tags: [other]
hide_table_of_contents: false
---

[官方文档](https://web.mit.edu/music21/doc/usersGuide/index.html)

建议配合官方图片食用

## 音符（Notes）

标准音符的概念被包含在 `note` 的 `Note` 对象之中

直接输入 `note` 便可得到该模块的位置

如果你想知道 `note` 除了 `Note` 还包含什么，可以输入 `dir(note)`

###  创建音符

让我们使用 `note.Note` 创建一个音符

```python
f = note.Note('F5')
```

### 音符的属性

`F5` 是这个音符的音名

通过 `.name` `.step` 和 `.octave` 可以得到其音名、音级、八度（在第几个八度）信息

```python
f.name f.step f.octave
```

```shell
'F' 'F' 5
```

`.step` 得到不包含变化音及八度信息的音名，这里成为音级严格来说并不准确

当然也可以使用 `.pitch` 直接得到其音名

```python
f.pitch
```

```shell
 <music21.pitch.Pitch F5>
```

使用 `.pitch.frequency` 得到其频率

```python
f.pitch.frequency
```

```shell
音高698.456462866008
```

使用 `.pitch.pitchClass` 同样可以得到其音级（距离同一个八度中 `c` 的半音数字），使用 `.pitch.pitchClassString` 则可以得到一个 `String`

```python
f.pitch.pitchClassf.pitch.pitchClassString type(f.pitch.pitchClassString)
```

```shell
音高5 '5' <class'str'>
```

在 music21 中，升降分别使用 `#` 和 `-`，我们创建一个新的音符

```python
b_flat = note.Note("B2-")
```

使用 `.pitch.accidental` 可获得其属性 (`accidental` 在音乐中表示变音记号)，使用 `.pitch.accident.alter` 获得其半音变化数量，使用 `.pitch.accidental.name` 获得其

```python
b_flat.pitch.accidental b_flat.pitch.accidental.alter b_flat.pitch.accidental.name
```

```shell
<accidental flat> -1.0 'flat'
```

**注意，这里是一个浮点数，这意味着 music21 支持四分音符之类现实中通常不使用的东西**

此外，并不是每一个音符都有 accidental 的，某些音符会返回 `None`

我们可以使用一个判断语句来解决：

```python
if d.pitch.accidental is not None:
    print(d.pitch.accidental.name)
```

如果你安装了 `MusicXML` 阅读器，使用 `f.show()` 可查看其五线谱

### 修改音符

使用 `.transpose` 修改你的音符

```python
d = b_flat.transpose("M3") #将Bb上调大三度，变为D
```

这种用法并没有改变音符本身，而是返回一个变量

可以使用 `inplace=True` 进行原地操作

### 休止符

使用 `note.rest`

```python
a = note.rest() # 记得加括号
```

最后提醒一个点，不要使用 `note` 作为音符的变量

```
note = note.Note("C4")
```

## 音高（Pitch），时值（Duration）

### 音高

使用 `pitch.pitch()` 创建一个音高对象

```python
p1 = pitch.Pitch('b-4')
```

有许多属性和 `note` 是一样的

```python
pl.octave pl.name pl.pitchClass pl.accidental.alter 
```

```shell
4 'B-' 10 -1.0
```

`.transpose()` 同样可以使用

`.nameWithOctave` 和 `.midi`

```python
pl.nameWithOctave pl.midi
```

```shell
'B-4' 70
```

这些属性大多数都可以修改

```python
pl.name = "d#"
pl.octave = 3
pl.nameWithOctave 
```

```python
'D#3'
```

这时 pl 代表的音符已经变为了 `D#3`

**实际上，每一个 Note 对象内部，都有一个 Pitch 对象，我们对 `note.Note` 做的一切，都可以用 `note.Note.pitch` 对象代替**

一些 `note` 不支持的属性

```python
csharp.pitch.spanish # 获得其西班牙名称
```

可以使用一些其他的方法，来更清晰地打印

```python
print(csharp.pitch.unicodeName) 
```

```shell
 C♯
```

获得一些同音的方法

```python
print( csharp.pitch.getEnharmonic() )
print( csharp.pitch.getLowerEnharmonic() )
```

```shell
 D-4
 B##3
```

### 时值

任何音符的存在都离不开时值 `Duration`

创建一个二分音符

```python
halfDuration = duration.Duration('half')
# ‘whole','half','quarter','eighth','16th','32th','64th' 一直到'2048th'，虽太小无法在乐谱上无法显示
# 'breve','longa','maxima' 2,4,8个全音
```

另一种创建方法是说明他有多少个四分音符

```python
dottedQuarter = duration.Duration(1.5) 
```

可以使用 `.quarterLength` 得到时值是多少个四分音符

还可以使用 `Note` 创建

```python
c = note.Note("C4", type='whole')
```

`.type` 可以得到一般类型，如 'half','quarter'

`.dots` 可以得到音符有多少个附点

使用 `.lyric` 添加歌词（具体看文档吧，不具体介绍了）

```python
otherNote = note.Note("F6")
otherNote.lyric = "I'm the Queen of the Night!"
```

## 流 (Stream)

我们可以通过列表对 note 等对象进行处理，但是它们对音乐一无所知，因此需要一个类似于列表的对象具有一定“智能”的对象，成为 `Stream`

流有许多子类 `Score` 乐谱、`Part` 声部、`Measure` 小节

### 创建流

`Stream` 中储存的元素必须是 music21 对象，如果想加入不属于 music21 的对象，请将其放入 `ElementWrapper`

使用 `Stream()` 创建流，`.append()` 方法添加元素，`.repeatAppend()` 方法添加多个相同的音符

```python
stream1 = stream.Stream()
stream1.append(note1)
stream1.append(note2)
stream1.append(note3)

stream2 = stream.Stream()
n3 = note.Note('D#5') # octave values can be included in creation arguments
stream2.repeatAppend(n3, 4)
```

使用 `.show('text')` 查看其中的内容及其偏移量（从 0.0 开始，一般 1 个偏移量指一个四分音符的长度）

```python
stream1.show('text')
```

```shell
 {0.0} <music21.note.Note C>
 {2.0} <music21.note.Note F#>
 {3.0} <music21.note.Note B->
```

流的大部分方法和列表相同，如切片、索引（还可使用 `.index()` 访问）、`pop()`、`.append()`、`len()` 等，并且流中也可以存放列表

### 按类分离元素

提供一种过滤流以获取所需元素 `.getElementByClass()`

```python
for thisNote in stream1.getElementsByClass(["Note", "Rest"]):
    print(thisNote, thisNote.offset)
```

```shell
 <music21.note.Note C> 0.0
 <music21.note.Note F#> 2.0
 <music21.note.Note B-> 3.0
```

此外也可使用 `.notes`、`.notesAndRests`、`.pitches` 等来进行过滤，直接使用会返回所有，如

```python
stream1.pitches
```

```shell
[<music21.pitch.Pitch D#5>,
 <music21.pitch.Pitch D#5>,
 <music21.pitch.Pitch D#5>,
 <music21.pitch.Pitch D#5>]

```

```python
for thisNote in stream1.notesAndRests:
    print(thisNote)
```

```shell
 <music21.note.Note C>
 <music21.note.Note F#>
 <music21.note.Note B->
```

### 通过偏移量分离元素

`getElementsByOffset()`

```python
sOut = stream1.getElementsByOffset(2, 3).stream()
sOut.show('text')
```

```shell
 {2.0} <music21.note.Note F#>
 {3.0} <music21.note.Note B->   
```

还有 `getElementAtOrBefore()`（某个偏移量及其之前 score = stream.Score()

1），`getElementAfterElement()`（某个偏移量之后）

### 更多功能

`.analyze('ambitus')` 获得流中的音域范围

`.lowestOffset` 返回偏移量的最小值

`__repr__`

`.id`，可以自己设定，相当于名字，如

```python
s = stream.Score(id='mainScore')
p0 = stream.Part(id='part0')
p1 = stream.Part(id='part1')
```

`.duration` 储存 Duration 对象的属性

### 小节 (Measure)

可以使用 `corpus` 访问大量的乐谱，使用 `Parse()` 从语料库中解析出 `Score`（一种流的子类）

```python
sBach = corpus.parse('bach/bwv57.8')
```

它包含一个 `Metadata` 对象、一个 `StaffGroup` 对象和四个 `Part` 对象。

可以使用 `measures()` 或 `measure()` 获取多个或一个小节，前者获取整个乐曲所有 `Part` 的小节，后者必须对一个 `Measure` 对象

然而一个问题是，这与使用 `getElementsByClass(stream.Measure)` 并不相同，因为在乐曲中存在小节并不连续的情况

### 递归方法

`.recurse()` 可以访问流中的每一个元素，若任何子元素也是流，他将访问该流中的每一个元素

他会返回一个生成器，使用循环来访问每一个元素

```python
recurseScore = s.recurse()
recurseScore
```

```SHELL
 <music21.stream.iterator.RecursiveIterator for Score:mainScore @:0>
```

```python
for el in s.recurse():
    print(el.offset, el, el.activeSite)
```

```SHELL
 0.0 <music21.stream.Part part0> <music21.stream.Score mainScore>
 0.0 <music21.stream.Measure 1 offset=0.0> <music21.stream.Part part0>
 0.0 <music21.note.Note C> <music21.stream.Measure 1 offset=0.0>
 4.0 <music21.stream.Measure 2 offset=4.0> <music21.stream.Part part0>
 0.0 <music21.note.Note D> <music21.stream.Measure 2 offset=4.0>
 0.0 <music21.stream.Part part1> <music21.stream.Score mainScore>
 0.0 <music21.stream.Measure 1 offset=0.0> <music21.stream.Part part1>
 0.0 <music21.note.Note E> <music21.stream.Measure 1 offset=0.0>
 4.0 <music21.stream.Measure 2 offset=4.0> <music21.stream.Part part1>
 0.0 <music21.note.Note F> <music21.stream.Measure 2 offset=4.0>					
```

大多数过滤方法也可以用于该生成器

### 扁平化流

`.flat` 可以将嵌套流打平，并赋予新的 `offset`

## 和弦 (Chord)

### 创建和弦

```python
cMinor = chord.Chord(["C4","G4","E5-"])  # 通过音名的列表创建

d = note.Note('D4')
fSharp = note.Note('F#4')
a = note.Note('A5')
dMajor = chord.Chord([d, fSharp, a]) # 添加已有的音符

e7 = chord.Chord("E4 G#4 B4 D5") # 通过带有空格的字符串创建

es = chord.Chord("E- G B-") # 如果所有音符的都在一个八度内，则不需要八度信息

```

### 功能

`Chord` 和 `Note` 对象都是 `GneralNote` 对象的子类，因此大部分 Note 的属性，Chord 都能使用

其中，音高由 `.pitch` 被替换为 `.pitches`

一些其他功能

```python
cMinor.isMinorTriad() # 是否是小三和弦
cMinor.isMajorTriad() # 是否是大三和弦
cMinor.inversion() # 是否处于转位状态
cMinor.add() # 添加音符
cMinor.remove() # 移除音符
cClosed = cMinor.closedPosition() # 封闭和弦，不会该改变原对象
semiClosedPosition() # 不太懂
cn1 = cMinor.commonName # 得到和弦的名字
fMajor.fullName # 包含step和时值
fMajor.pitchedCommonName # 加入了pitch信息
```

和弦和音符一样可以添加到流之中

## 加载文件

使用 `corpus.parse()` 从语料库加载文件

使用 `converter.parse()` 从本地磁盘或网络加载文件

如果文件名没有后缀或者后缀有误，可以使用 `format="FORMAT"`

music21 会在第一遍读取文件后保存优化版本，若下次读取时检测到没有更改文件，那么速度将会提升 2 到 5 倍（大概是通过检查文件修改时间是否变动，可以使用 `forceSource=True` 来保证重新加载原文件）

使用 `converter.Converter().subconvertersList('input')` 得到其可以读入的所有格式

使用 `converter.Converter().subconvertersList('output')` 得到其可以写的所有格式

其可以使用的文件格式有 `Humdrum`、`MusicXML`、`MusicXML`、`Musedata`、`MIDI` 等

## Chordify

Chordify ，顾名思义，ify 后缀通常表示使动，即为使…变为和弦，可翻译为 `和弦化`

```python
b = corpus.parse('bwv66.6') # 读取乐谱，该乐谱有很多个声部
bChords = b.chordify() # 和弦化，多个声部变为一个和弦
```

看不懂了…
