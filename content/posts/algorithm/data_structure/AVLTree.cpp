#include <cstdio>
#include <algorithm>

using namespace std;

struct Node {
    int data;
    int height;  // 本质上等于定义了一个 layer
    Node *left, *right;
};

// 新建节点
Node* new_node(int data)
{
    Node *root = new Node;
    root->data = data;
    root->height = 1;
    root->left = root->right = NULL;
    return root;
}

// 查找
void search(Node* root, int x)
{
    if (root == NULL)
    {
        printf("Search failed!");
        return;
    }
    if (root->data == x)
    {
        printf("%d", root->data);
        return;
    }
    else if (root->data < x)
        search(root->left, x);
    else
        search(root->right, x);
}

// insert() 的辅助函数：获取 root 节点的高度
int get_height(Node* root)
{
    if (root == NULL) return 0;
    else return root->height;
}

// insert() 的辅助函数：获取 root 节点的平衡因子
int get_balance_factor(Node* root)
{
    return get_height(root->left) - get_height(root->right);
}

// insert() 的辅助函数：更新 root 节点的高度
void update_height(Node* root)
{
    root->height = max(get_height(root->left), get_height(root->right)) + 1;
}

// insert() 的辅助函数：左旋 root 节点
// 会伴随节点高度的变化
void left_rotation(Node* &root)
{
    Node* temp = root->right;
    root->right = temp->left;
    temp->left = root;

    update_height(root);  // 更新较低节点 root
    update_height(temp);  // 跟新较高节点 temp

    root = temp;
}

// insert() 的辅助函数：右旋 root 节点
// 会伴随节点高度的变化
void right_rotation(Node* &root)
{
    Node* temp = root->left;
    root->left = temp->right;
    temp->right = root;

    update_height(root);
    update_height(temp);

    root = temp;
}

// 插入：递归插入，更新节点高度，旋转保持 AVL 性质
void insert(Node* &root, int v)
{
    if (root == NULL)
    {
        root = new_node(v);
        return;
    }
    if (v < root->data)
    {
        insert(root->left, v);
        update_height(root);
        if (get_balance_factor(root) == 2)
        {
            if (get_balance_factor(root->left)  == 1)  // LL
                right_rotation(root);
            else if (get_balance_factor(root->left) == -1)  // LR
            {
                left_rotation(root->left);
                right_rotation(root->right);
            }
        }
    }
    else
    {
        insert(root->right, v);
        update_height(root);
        if (get_balance_factor(root) == -2)
        {
            if (get_balance_factor(root->right) == -1)  // RR
                left_rotation(root);
            else if (get_balance_factor(root->right) == 1)  // RL
            {
                right_rotation(root->right);
                left_rotation(root);
            }
        }
    }
}

// 创建树
Node* create_by_array(int data[], int n)
{
    Node* root = NULL;
    for (int i = 0; i < n; i++)
        insert(root, data[i]);
    return root;
}

int main()
{
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Node* root = create_by_array(a, 10);
    return 0;
}
