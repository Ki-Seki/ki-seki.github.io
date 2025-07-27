/*
 * hint:
 * 素数相关的算法,包括：
 * 使用 sqrt 来优化的素数判断函数
 * 使用平方技巧来优化的素数判断函数
 * 求素数表：埃氏筛法，Eratosthenes 筛法
*/
#include <iostream>
#include <cmath>

/* 平方技巧的证明：
 * 任何一个数可以拆分成两因子相乘，如 a = n * m；
 * 不失一般性，令 n <= m；
 * 则 max(n) = √a
 * 由于只需要判断所有的 n 是否既不是 1 和 a 且是 a 的因子；
 * 所以循环中遍历到 √a 即可
*/

// 使用 sqrt 来优化的素数判断函数
// 需要 <cmath>
bool is_prime(int n)
{
    for (int i = 2; i <= (int) sqrt(n * 1.0); i++)
        if (n % i == 0)
            return false;
    return true;
}

// 使用平方技巧来优化的素数判断函数
// 缺点是若 n 较大，易产生溢出
bool is_prime_vice(int n)
{
    for (int i = 2; i * i <= n; i++)
        if (n % i == 0)
            return false;
    return true;
}

// 求素数表：埃氏筛法，Eratosthenes 筛法
// 时间复杂度 O(nloglogn)
#define MAXN 100  // 素数表大小
int prime[MAXN + 5], p_len = 0;
bool not_prime[MAXN * 20] = {};
// 找到 [2, n] 范围内的素数，保存至 prime[]
void find_prime(int n)
{
    for (int i = 2; i <= n; i++)
        if (not_prime[i] == false)
        {
            prime[p_len++] = i;
            for (int j = i + i; j <= n; j += i)
                not_prime[j] = true;
        }
}

int main()
{
    find_prime(MAXN);
    for (int i = 0; i < p_len; i++)
        printf(" %d", prime[i]);
}
