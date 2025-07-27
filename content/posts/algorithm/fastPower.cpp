/*
 * hint:
 * 三种快速幂计算 a ^ b % c，假设数据范围如下：
 *      a < 10 ^ 9
 * 0 <= b < 10 ^ 9
 *  1 < c < 10 ^ 9
 *
 * 注意保存中间值的变量一定要定义为 long long 类型的；
 * 如 fpr 中的 tmp，fpi 中的 ans 和 fpib 中的 ans。
 * 如果定义为 int，可能导致意外的变量截断
*/

#include <iostream>
#include <ctime>

typedef long long LL;

LL fastPowerRecursion(LL a, LL b, LL c)
{
    if (a == 0 || c == 1) return 0;  // special judge
    if (b == 0) return 1;  // recursive boundary
    a %= c;  // optimization
    if (b & 1)  // if b is even
        return a * fastPowerRecursion(a, b - 1, c) % c;
    else
    {
        LL tmp = fastPowerRecursion(a, b / 2, c);
        return tmp * tmp % c;
    }
}

LL fastPowerIteration(LL a, LL b, LL c)
{
    if (a == 0 || c == 1) return 0;  // special judge
    a %= c;
    LL ans = 1;
    while (b > 0)
    {
        if (b & 1) ans = ans * a % c;
        a = a * a % c;
        b >>= 1;
    }
    return ans;
}

LL fastPowerIterationBits(LL a, LL b, LL c)
{
    if (a == 0 || c == 1) return 0;  // special judge
    a %= c;
    int len = sizeof(LL) * 8, bit;
    LL ans = 1;
    for (int i = len - 1; i >= 0; i--)  // traversal from high bit to low bit in b
    {
        bit = (b >> i) & 1;
        ans = (ans * ans) * (bit ? a : 1) %c;
    }
    return ans;
}

void test()
{
    LL data_set[300][3] = {
        {55, 100, 450},
        {34, 12, 43},
        {0, 1, 1},
        {1, 0, 1},
        {0, 0, 23},
        {100, 100, 10},
        {5, 7, 99999},
        {-1, 45, 4},
        {99993425, 5345, 456754},
        {987654321, 783589549, 4354359834}  // 正确答案可能是：34099175
    };
    int len = 10;

    for (int i = 0; i < len; i++)
    {
        long begin, end;
        LL fpr, fpi, fpib;
        printf("Test %d:\n(a, b, c) = (%d, %d, %d)\n", i + 1, data_set[i][0], data_set[i][1], data_set[i][2]);
        begin = clock();
        fpr = fastPowerRecursion(data_set[i][0], data_set[i][1], data_set[i][2]);
        end = clock();
        printf("fpr  %lld\t%ldms\n", fpr, (end - begin));

        begin = clock();
        fpi = fastPowerIteration(data_set[i][0], data_set[i][1], data_set[i][2]);
        end = clock();
        printf("fpi  %lld\t%ldms\n", fpi, (end - begin));

        begin = clock();
        fpib = fastPowerIterationBits(data_set[i][0], data_set[i][1], data_set[i][2]);
        end = clock();
        printf("fpib %lld\t%ldms\n", fpib, (end - begin));
        printf("\n");
    }
}

int main()
{
    printf("%d %d %d", fastPowerIteration(2, 499, 100000), fastPowerRecursion(2, 499, 100000), fastPowerIterationBits(2, 499, 100000));
    // test();
    return 0;
}
