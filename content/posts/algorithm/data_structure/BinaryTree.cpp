#include <cstdio>
#include <queue>

using namespace std;

// 二叉树节点
struct Node
{
    int data;
    // int layer;  // 层次遍历时所需要的层次号
    Node* left;
    Node* right;
};

// 新建节点
Node* newNode(int val)
{
    Node* p = new Node;
    p->data = val;
    p->left = p->right = NULL;
    return p;
}

// 替换节点：将树中所有为 data 的节点值替换为 new_data
void replace(Node* root, int data, int new_data)
{
    // 递归边界
    if (root == NULL)
        return;
    if (root->data == data)
        root->data = new_data;

    // 分岔口
    replace(root->left, data, new_data);
    replace(root->right, data, new_data);
}

// 为树插上一个值为 data 的新节点
// 由于要创建值，所以 root 当使用引用
void insert(Node* &root, int data)
{
    if (root == NULL)
    {
        root = newNode(data);
        return;
    }
    // 根据二叉树性质改变此行，以实现不同插入方式
    if (root->left == NULL)
        insert(root->left, data);
    else if (root->right == NULL)
        insert(root->right, data);
    else if (data % 2)
        insert(root->left, data);
    else
        insert(root->right, data);
}

// 从数组创建一个满二叉树
Node* createByArray(int data[], int size)
{
    Node* root = NULL;
    for (int i = 0; i < size; i++)
        insert(root, data[i]);
    return root;
}

// 先序遍历
void preorder(Node* root)
{
    if (root == NULL)
        return;
    printf("%d ", root->data);
    preorder(root->left);
    preorder(root->right);
}

// 中序遍历
void inorder(Node* root)
{
    if (root == NULL)
        return;
    inorder(root->left);
    printf("%d ", root->data);
    inorder(root->right);
}

// 后序遍历
void postorder(Node* root)
{
    if (root == NULL)
        return;
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->data);
}

// 层次遍历：需要在结构体中新增属性，层次号，即 layer
// 函数体内被注释掉的行：计算层次号
void layerOrder(Node* root)
{
    queue<Node*> q;
    // root->layer = 1;
    q.push(root);
    while (!q.empty())
    {
        Node* front = q.front();
        q.pop();
        printf("%d ", front->data);
        if (front->left)
        {
            // front->left->layer = front->layer + 1;
            q.push(front->left);
        }
        if (front->right)
        {
            // front->right->layer = front->layer + 1;
            q.push(front->right);
        }
    }
}

// 通过先序和后序遍历序列复原二叉树
Node* createByPreIn(int pre[], int in[], int preL, int preR, int inL, int inR)
{
    // 递归边界
    if (preL > preR)
        return NULL;

    Node* root = new Node;
    root->data = pre[preL];
    int left_len = 0, i;
    for (i = inL; i < inR; i++)
        if (pre[preL] == in[i])
        {
            left_len = i - inL;
            break;
        }

    // 分岔口
    root->left = createByPreIn(pre,
                               in,
                               preL + 1,
                               preL + left_len,
                               inL,
                               i - 1);
    root->right = createByPreIn(pre,
                                in,
                                preL + 1 + left_len,
                                preR,
                                i + 1,
                                inR);
    return root;
}

Node* createByLayerIn()
{

}

int main()
{
    int data[] = {1, 2, 3, 4, 5, 6};
    Node* root = createByArray(data, 6);
    replace(root, 6, 7);

    // 遍历
    preorder(root); printf("\n");
    inorder(root); printf("\n");
    postorder(root); printf("\n");
    layerOrder(root); printf("\n");

    // 通过遍历序列复原二叉树
    int pre[] = {1, 2, 5, 3, 4, 7},
        in[] = {5, 2, 1, 4, 3, 7};
    Node* recovered = createByPreIn(pre, in, 0, 5, 0, 5);
    postorder(recovered); printf("\n");
    return 0;
}
