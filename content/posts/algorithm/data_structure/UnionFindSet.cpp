#include <cstdio>
#define N 100

// define
int set[N];

// find root of x

int find_iteration(int x)
{
    while (x != set[x])
        x = set[x];
    return x;
}

int find_recursion(int x)
{
    if (x == set[x])
        return x;
    else
        return find_recursion(set[x]);
}

// union two sets, that is, replaces the set containing x and the set containing y with their union
void union_sets(int a, int b)
{
    int root_a = find_iteration(a), root_b = find_iteration(b);
    if (root_a != root_b)
        set[root_b] = root_a;
    return;
}

// optimize find() 利用路径压缩

int optimized_find_iteration(int x)
{
    // find root
    int root = x;
    while (root != set[root])
        root = set[root];

    // replace all nodes' parent with the root
    int index = x;
    while (index != set[index])
    {
        int temp = index;
        set[temp] = root;
        index = set[index];
    }
    return root;
}

int optimized_find_recursion(int x)
{
    if (x == set[x])
        return x;
    else
    {
        int root = optimized_find_recursion(set[x]);
        set[x] = root;
        return root;
    }
}

int main()
{
    // 1st set
    set[1] = 1;
    set[2] = 1;
    set[3] = 2;
    set[4] = 2;

    // 2nd set
    set[5] = 5;
    set[6] = 5;

    printf("%d %d\n", find_iteration(4), find_recursion(6));
    union_sets(4, 6);
    printf("%d %d\n", find_iteration(4), find_recursion(6));
    printf("%d %d\n", optimized_find_iteration(4), optimized_find_recursion(6));
    return 0;
}
