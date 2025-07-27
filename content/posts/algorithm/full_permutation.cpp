#include <iostream>
#define MAXN 11
using namespace std;

int n, p[MAXN];
bool exist[MAXN] = {};

void perm(int index)
{
    if (index == n + 1)
    {
        for (int i = 1; i <= n; i++)
            printf("%d", p[i]);
        printf("\n");
        return;
    }
    for (int i = 1; i <= n; i++)
        if (! exist[i])
        {
            p[index] = i;
            exist[i] = true;
            perm(index + 1);
            exist[i] = false;
        }
}

/*
参数：
    index: 现在正在为 p[index] 找合适的值
    n: n 阶全排列
    p[]: 暂存排列的数组
    exist[]: 保存是否已经使用过某个数字的布尔状态数组
功能：
    按升序输出 n 阶数字全排列，返回全排列个数
*/
int perm(int index, int n, int p[], bool exist[])
{
    int cnt = 0;
    if (index == n + 1)  // 递归边界
    {
        for (int i = 1; i <= n; i++)
            printf("%d", p[i]);
        printf("\n");
        cnt++;
    }
    else
        for (int x = 1; x <= n; x++)  // 枚举 1 ~ n 所有 x
            if (! exist[x])  // 若 x 未被使用，填在 p[index]
            {
                p[index] = x;
                exist[x] = true;
                cnt += perm(index + 1, n, p, exist);
                exist[x] = false;
            }
    return cnt;
}

int main()
{
    // 1st method, using global variables
    n = 3;
    perm(1);

    // 2nd method
    int p[15];
    bool exist[15];
    printf("%d : \n\n", perm(1, 5, p, exist));
    return 0;
}
