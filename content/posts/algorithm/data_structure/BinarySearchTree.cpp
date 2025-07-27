/*
 * hint:
 * 任何新建的节点都要用 NULL 初始化，包括头指针 root
*/

#include <cstdio>

struct Node
{
    int data;
    Node *left, *right;
};

// 创建元素
Node* new_node(int data)
{
    Node *root = new Node ;
    root->data = data;
    root->left = root->right = NULL;
    return root;
}

// 查找
void search(Node* root, int x)
{
    if (root == NULL)
    {
        printf("Search failed.\n");
        return;
    }
    if (root->data == x)
        printf("%d\n", root->data);
    else if (x < root->data)
        search(root->left, x);
    else
        search(root->right, x);
}

// 插入
void insert(Node* &root, int x)
{
    if (root == NULL)
    {
        root = new_node(x);
        return;
    }
    if (root->data == x)
        return;  // 表示此二叉树结构中不存在重复元素
    else if (x < root->data)
        insert(root->left, x);
    else
        insert(root->right, x);
}

// 从一个数组建树
Node* create_by_array(int arr[], int n)
{
    Node *root = NULL;
    for (int i = 0; i < n; i++)
        insert(root, arr[i]);
    return root;
}

// 辅助函数：寻找以 root 为根节点的树中最大权值节点
Node* find_max(Node* root)
{
    while (root->right != NULL)
        root = root->right;
    return root;
}

// 辅助函数：寻找以 root 为根节点的树中最小权值节点
Node* find_min(Node* root)
{
    while (root->left != NULL)
        root = root->left;
    return root;
}

// 删除元素：找到 x 的前驱，替换 x 为其前驱，然后删除前驱
void delete_node(Node* &root, int x)
{
    if (root == NULL)
        return;
    if (root->data == x)  // 找到要删除元素
        if (root->left == NULL && root->right == NULL)  // 是叶子节点
            root = NULL;
        else if (root->left != NULL)  // 左子树存在
        {
            Node* pre = find_max(root->left);  // 找到前驱节点
            root->data = pre->data;
            delete_node(root->left, pre->data);
            // 删除节点也可以不采用递归删除的方法，只不过编写代码会麻烦些
            // 采用分类讨论的方法直接删除 root 左子树中最靠右的节点
            // 分为：①当 pre 是 root 的左孩子；② pre 是仅有一个左孩子的节点
            // 下面的右子树也可以类比操作
        }
        else  // 右子树存在
        {
            Node* next = find_min(root->right);  // 找到后继节点
            root->data = next->data;
            delete_node(root->right, next->data);
        }
        // 通过判断左右子树是否存在去删除节点的方法易造成二叉树不平衡
        // 两种解决方法：
        // 1. 交替找前驱或后继
        // 2. 记录子树高度，总是在高的那个里面找
    else if (x < root->data)  // 向左子树寻找删除元素
        delete_node(root->left, x);
    else
        delete_node(root->right, x);  // 向右子树寻找删除节点
}

int main()
{
    int array[] = {1, 32, 3, 90, 1, -9, 12, 11, 9};
    Node* root = create_by_array(array, 9);
    delete_node(root, 8);
    delete_node(root, -9);
    delete_node(root, 32);
    delete_node(root, 1);
    return 0;
}
