---
date: '2025-03-10T14:00:00+08:00'
title: 'Python 多线程与多进程基础知识（Cheatsheet）'
author:
  - Shichao Song
summary: ''
tags: ["Python", "threading", "multiprocessing"]
math: false
---

## 高层抽象与基本任务启动

对于日常简单任务，你可以直接使用 `threading.Thread` 或 `multiprocessing.Process` 来启动任务。但更推荐使用线程池/进程池模型来便于管理和调度任务。

### 直接启动任务

```python
import threading
import time

def task(n):
    print(f"线程 {n} 开始")
    time.sleep(1)
    print(f"线程 {n} 结束")

threads = []
for i in range(5):
    t = threading.Thread(target=task, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

```

```python
import multiprocessing
import time

def task(n):
    print(f"进程 {n} 开始")
    time.sleep(1)
    print(f"进程 {n} 结束")

processes = []
for i in range(5):
    p = multiprocessing.Process(target=task, args=(i,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

```

### 使用线程池/进程池模型

Python 提供了方便的并发库如 `concurrent.futures`，可以使用线程池或进程池执行多个任务。

### ThreadPool 示例

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def task(n):
    print(f"处理任务 {n}")
    time.sleep(1)
    return f"任务 {n} 完成"

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(task, i) for i in range(5)]
    for future in as_completed(futures):
        print(future.result())

```

### ProcessPool 示例

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def task(n):
    print(f"处理任务 {n}")
    time.sleep(1)
    return f"任务 {n} 完成"

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(task, i) for i in range(5)]
        for future in as_completed(futures):
            print(future.result())

```

---

## 常见问题及解决方案

多线程/多进程编程中经常会遇到一些典型问题，比如竞态条件（Race Condition）、死锁（Deadlock）和饥饿现象（Starvation）。以下是一些解决方案及代码示例。

### 竞态条件（Race Condition）

使用同步原语（例如，Lock, Semaphore, Event, Condition）来保证共享资源的互斥访问。

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("最终计数值:", counter)

```

### 死锁（Deadlock）

使用高级同步工具如 `RLock`（可重入锁）和 `Barrier` 可以更好地预防死锁。

```python
import threading

lock = threading.RLock()

def recursive_task(n):
    with lock:
        if n > 0:
            print(f"递归任务: {n}")
            recursive_task(n-1)

thread = threading.Thread(target=recursive_task, args=(5,))
thread.start()
thread.join()

```

### 饥饿现象（Starvation）

使用负载均衡工具（如 `Queue`、`Semaphore`）或者池来调控任务分配。

```python
import threading
import queue
import time
import random

task_queue = queue.Queue(maxsize=10)

def producer(name):
    while True:
        item = random.randint(1, 100)
        task_queue.put(item)
        print(f"生产者 {name} 生产了: {item}")
        time.sleep(random.random())

def consumer(name):
    while True:
        item = task_queue.get()
        print(f"消费者 {name} 消费了: {item}")
        time.sleep(random.random())
        task_queue.task_done()

# 启动生产者和消费者线程
threading.Thread(target=producer, args=("P1",), daemon=True).start()
threading.Thread(target=producer, args=("P2",), daemon=True).start()
threading.Thread(target=consumer, args=("C1",), daemon=True).start()
threading.Thread(target=consumer, args=("C2",), daemon=True).start()

# 主线程等待一段时间
time.sleep(5)

```

## 常见模型

下面是一些在多线程与多进程编程中常用的设计模型：

### 池模型

- 应用：线程池与进程池 (如上例 ThreadPoolExecutor / ProcessPoolExecutor)
- 优点：自动管理工作者线程/进程，调控并发数量

### 锁模型

- 应用：线程或进程间对共享资源的访问控制
- 同步工具：Lock, RLock, Semaphore, Condition

### 生产者-消费者模型

- 应用：任务调度，资源分配等场景 (参见上面的 producer_consumer 示例)

### 事件通知模型

- 应用：线程之间的等待与通知
- 工具：Event, Condition

```python
import threading
import time

event = threading.Event()

def wait_for_event():
    print("等待事件触发...")
    event.wait()
    print("事件已触发，继续执行！")

thread = threading.Thread(target=wait_for_event)
thread.start()

time.sleep(2)
event.set()
thread.join()

```

### 管道通信模型（适用于 multiprocessing）

- 管道与队列在多进程间传递数据

```python
from multiprocessing import Process, Pipe

def sender(conn):
    conn.send("消息通过管道传递")
    conn.close()

def receiver(conn):
    print("接收到:", conn.recv())

if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    p1 = Process(target=sender, args=(child_conn,))
    p2 = Process(target=receiver, args=(parent_conn,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

```

### 共享内存模型（适用于 multiprocessing）

- 使用 `multiprocessing.Value` 与 `multiprocessing.Array` 来分享内存数据

```python
from multiprocessing import Process, Value, Array

def modify_shared(n, a):
    n.value += 1
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == "__main__":
    num = Value('i', 0)
    arr = Array('i', range(10))
    p = Process(target=modify_shared, args=(num, arr))
    p.start()
    p.join()
    print("共享整数值:", num.value)
    print("共享数组:", list(arr))

```

## 总结

本教程对 Python 多线程与多进程编程进行了快速入门和常见问题的讨论，主要涵盖了：

- 如何直接启动线程与进程
- 线程池与进程池模型的使用（`submit` 与 `map` 等方法）
- 竞态条件、死锁和饥饿现象的解决方案
- 典型模型，包括池模型、锁模型、生产者-消费者、事件通知、管道通信以及共享内存