/*
 * 堆数据结构
 * 实践中发现，递归代码版本的调整函数效率可能比迭代的好
*/
#include <cstdio>
#include <algorithm>
#define MAXN 100

using namespace std;

// 定义堆：允许重值出现
int heap[MAXN],  // 大顶堆，下标从 1 开始（若是小顶堆，则类比操作即可）
    size;  // 堆的大小

// 向下调整：以 low 为欲调整节点，将区间 [low, high] 调整为合法的堆
void down_adjust(int low, int high)
{
    int node = low,  // 欲调节点
    max_child = low * 2;  // 欲调节点子结点中较大的
    while (max_child <= high)
    {
        // 当右孩子存在且右孩子大于左孩子时
        if (max_child + 1 <= high && heap[max_child + 1] > heap[max_child])
            max_child++;

        if (heap[max_child] > heap[node])  // 需要调整当前节点
        {
            swap(heap[max_child], heap[node]);

            // 为迭代地向下调整做准备
            node = max_child;
            max_child = node * 2;
        }
        else  // 已经无需调整
            break;
    }
}

// 递归版本的向下调整
void down_adjust_recursion(int low, int high)
{
    if (2 * low > high)  // 当 low 的左子节点不在 high 之下
        return;
    int node = low, max_child = low * 2;
    if (max_child + 1 <= high && heap[max_child + 1] > heap[max_child])
        max_child++;
    if (heap[max_child] > heap[node])
    {
        swap(heap[max_child], heap[node]);
        down_adjust_recursion(max_child, high);
    }
}

// 建堆：条件是 heap 中已有初始数据，size 已赋过初值
void create_heap()
{
    // 一颗 CBT 有 rounded_up(size / 2.0) 个叶子节点，有 rounded_down(size / 2.0) 个非叶子节点
    for (int i = size / 2; i >= 1; i--)
        down_adjust_recursion(i, size);
}

// 删除堆顶元素
void delete_top()
{
    heap[1] = heap[size--];  // 用尾元素替换堆顶元素
    down_adjust_recursion(1, size);  // 堆顶向下调整
}

// 向上调整：以 high 为欲调节点，将区间 [low, high] 调整为合法的堆
// 一般来说，low 为 1，这表示调整至顶
void up_adjust(int low, int high)
{
    int node = high,  // 欲调节点
        parent = high / 2;  // 欲调节点的父亲
    while (parent >= low)
    {
        if (heap[node] > heap[parent])  // 需要调整
        {
            swap(heap[node], heap[parent]);
            node = parent;
            parent = node / 2;
        }
        else  // 无需调整
            break;
    }
}

// 递归版本的向上调整
void up_adjust_recursion(int low, int high)
{
    if (high / 2 < low)  // 当 high 的父亲节点不在 low 之上
        return;
    int node = high, parent = high / 2;
    if (heap[node] > heap[parent])
    {
        swap(heap[node], heap[parent]);
        up_adjust(low, parent);
    }
}

// 添加元素
void insert(int x)
{
    heap[++size] = x;
    up_adjust_recursion(1, size);
}

// 堆排序：大顶堆可以实现递增排序
// 条件是 heap 中已有初始数据，size 已赋过初值
void heap_sort()
{
    create_heap();
    int n = size;
    while (n)
    {
        swap(heap[1], heap[n]);  // 堆顶置于末尾
        down_adjust_recursion(1, --n);  // 调整区间 [1, n-1]
    }
}

// 用于测试的两个函数

void init_from_array(int data[], int n)
{
    for (int i = 1; i <= n; i++)
        heap[i] = data[i - 1];
    size = n;
}

void output()
{
    for (int i = 1; i <= size; i++)
        printf("%d ", heap[i]);
    printf("\n");
}

int main()
{
    int data[] = {85, 55, 82, 57, 68, 92, 99, 98, 66, 56};
    init_from_array(data, 10);
    create_heap();
    output();
    delete_top();
    output();
    insert(99);
    output();
    insert(70);
    output();
    heap_sort();
    output();
    return 0;
}
