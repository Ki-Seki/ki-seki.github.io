#include<cstdio>
#include<vector>
#include<queue>
#define MAXN 123

using namespace std;

// 静态树节点的定义

// // 方法一：数组保存：占内存空间大
// struct Node
// {
//     int data;
//     int child[MAXN];
// } tree[MAXN];

// // 方法二：vector 保存：方便，效率高
// struct Node
// {
//     int data;
//     vector<int> child;
// } tree[MAXN];

// // 方法三：无数据域：极简形式（实际上是图的邻接表表示法）
// vector<int> child[MAXN];

// 方法四：包括层号
struct Node
{
    int layer,
        data;
    vector<int> child;
} tree[MAXN];

// 新建节点
int index = 0;
int new_node(int data)
{
    tree[index].data = data;
    tree[index].child.clear();
    return index++;
}

// 先序遍历
void preorder(int root)
{
    printf("%d ", tree[root].data);
    for (int i = 0; i < tree[root].child.size(); i++)
        preorder(tree[root].child[i]);
}

// 计算层号的层序遍历
void layerorder(int root)
{
    queue<int> q;
    q.push(root);
    tree[root].layer = 1;
    while (!q.empty())
    {
        int front = q.front();
        q.pop();
        printf("%d", tree[front].data);
        for (int i = 0; i < tree[front].child.size(); i++)
        {
            int kid = tree[front].child[i];
            tree[kid].layer = tree[front].layer + 1;
            q.push(kid);
        }
    }
}
