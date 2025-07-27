/*
 * 哈夫曼树
 *
 * 以下用了三种方法能够部分或完全实现哈夫曼编码的功能：
 *
 * 1. 自建一个小顶堆，也就是手动实现优先队列，这个方法可以完整的实现哈夫曼树
 * 2. 使用系统自带的 priority_queue 实现优先队列的功能，同时创建二叉树节点
 * 3. 只是为了完成合并果子（即求总带权路径长度 Total WPL）这一具体问题，将 2. 中的二叉树节点删掉，只用 int 保存结果
 *
 * 推荐使用第二种方法，因为他既实现了完整的哈夫曼树，又易于编码
*/
#include <cstdio>
#include <algorithm>
#include <string>
#include <queue>
#define MAXN 100

using namespace std;

// 普通二叉树节点
struct Node {
    int data;
    Node *left, *right;
    // 下面这一行用于生成哈夫曼编码，如果只是要输出编码，那么可以将其注释掉
    string code;  // huffman code
};

// 二叉树：新建节点，这个节点值是用 new 运算符生成的，因此并非临时的
Node* new_node(int val)
{
    Node* root = new Node;
    root->data = val;
    root->left = root->right = NULL;
    return root;
}

/************************* 方法一 *************************/

// 小顶堆：保存指向树的指针
// 之所以保存指针，而不保存值，是因为保存值就会有被覆盖的风险，而导致已构造的哈夫曼树被破坏
Node* heap[MAXN];
int size;

// 堆：初始化，data 从 1 开始计数
void init(int data[], int n)
{
    size = n;
    for (int i = 1; i <= n; i++)
        heap[i] = new_node(data[i-1]);
}

// 堆：向下调整 [low, high]
void down_adjust(int low, int high)
{
    int node = low, min_child = node * 2;
    while (min_child <= high)
    {
        if (min_child + 1 <= high && heap[min_child+1]->data < heap[min_child]->data)
            min_child++;
        if (heap[min_child]->data < heap[node]->data)
        {
            swap(heap[node], heap[min_child]);
            node = min_child;
            min_child = node * 2;
        }
        else
            break;
    }
}

// 堆：向上调整 [low, high]
void up_adjust(int low, int high)
{
    int node = high, parent = high / 2;
    while (parent >= low)
    {
        if (heap[node]->data < heap[parent]->data)
        {
            swap(heap[node], heap[parent]);
            node = parent;
            parent = node / 2;
        }
        else
            break;
    }
}

// 堆：建堆
void create_heap()
{
    for (int i = size / 2; i >= 1; i--)
        down_adjust(i, size);
}

// 堆：删除堆顶元素
void delete_top()
{
    heap[1] = heap[size--];
    down_adjust(1, size);
}

// 堆：插入新元素
void insert(Node* x)
{
    heap[++size] = x;
    up_adjust(1, size);
}

// 堆：弹出堆顶指针元素，并删除堆顶
Node* pop()
{
    Node* top = heap[1];
    delete_top();
    return top;
}

// 哈夫曼树：合并两个节点，返回合并后的根节点，假设 a < b
Node* merge(Node* a, Node* b)
{
    Node* root = new_node(a->data + b->data);
    root->left = a;
    root->right = b;
    return root;
}

// 哈夫曼树：建树
Node* create_huffman()
{
    create_heap();
    while (size > 1)
    {
        Node* first = pop();
        Node* second = pop();
        insert(merge(first, second));
    }
    return heap[1];
}

// 哈夫曼树：生成哈夫曼编码，一般为 init 赋值 ""
// 哈夫曼编码结果
void gen_code(Node* root, string init)
{
    // 递归终止条件
    if (root == NULL)
        return;
    if (root->left == NULL && root->right == NULL)  // 只有叶子节点才有前缀编码
        root->code = init;
    else
    {
        gen_code(root->left, init + "0");
        gen_code(root->right, init + "1");
    }
}

/************************* 方法二 *************************/

// 定义 Node 结构体：上面已有，略

// 二叉树新建节点：上面已有，略

// 定义一个有关 Node 结构体的 cmp 函数
struct cmp {
    bool operator () (Node* a, Node* b)
    {
        return a->data > b->data;
    }
};

// 定义节点指针的优先队列
priority_queue<Node*, vector<Node*>, cmp> node_q;

// 哈夫曼树的 merge 函数和 gen_code：上面已有，略

/************************* 方法三 *************************/

// 定义代表小顶堆的优先队列，greater 实现小的优先
priority_queue<long long, vector<long long>, greater<long long>> q;

int main()
{
    /************************* 方法一 *************************/

    // int data[] = {4, 3, 2, 1}, n = 4;
    // init(data, n);
    // Node* root = create_huffman();
    // gen_code(root, "");

    /************************* 方法二 *************************/

    int data[] = {1, 2, 2, 3, 6}, n = 5;
    // 将节点加载到 node_q 中
    for (int i = 0; i < n; i++)
        node_q.push(new_node(data[i]));

    // 合并
    while (node_q.size() > 1)
    {
        Node* first = node_q.top(); node_q.pop();
        Node* second = node_q.top(); node_q.pop();
        node_q.push(merge(first, second));
    }

    // 生成哈夫曼编码
    gen_code(node_q.top(), "");

    /************************* 方法三 *************************/

    // int data[] = {1, 2, 2, 3, 6}, n = 5;
    // int ans = 0;
    // for (int i = 0; i < n; i++)
    //     q.push(data[i]);  // 将数据压入优先队列
    // while (q.size() > 1)
    // {
    //     int first = q.top(); q.pop();
    //     int second = q.top(); q.pop();
    //     q.push(first + second);
    //     ans += first + second;
    // }
    // printf("%d", ans);

    return 0;
}
