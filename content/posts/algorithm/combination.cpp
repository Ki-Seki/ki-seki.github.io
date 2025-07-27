/*
 * hint
 *
 * 警告！本部分所有内容仅供学习，程序健壮性未经严格测试
*/
#include <iostream>
#include <cstring>
#include <cmath>
#define MAXN 80  // 组合数表的最大尺度
typedef long long LL;
using namespace std;

// <<<<<<部分辅助函数>>>>>>

LL primes[MAXN], primes_cnt = 0;
bool not_prime[MAXN] = {};
// 找到 n 以内的素数
void find_prime(LL n)
{
    primes_cnt = 0;
    for (int i = 2; i <= n; i++)
        if (not_prime[i] == 0)
        {
            primes[primes_cnt++] = i;
            for (int j = i + i; j <= n; j += i)
                not_prime[j] = true;
        }
}

// 快速幂计算 a^b%c
LL fastPower(LL a, LL b, LL c)
{
    LL ans = 1;
    a %= c;  // 优化
    while (b)
    {
        if (b & 1) ans = ans * a % c;
        a *= a;
        b >>= 1;
    }
    return ans;
}

// 扩展欧几里得算法
LL exGcd(LL a, LL b, LL& x, LL& y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return a;
    }
    else
    {
        LL gcd = exGcd(b, a%b, x, y),
            tmp = x;
        x = y;
        y = tmp - (a / b) * y;
        return gcd;
    }
}

// 逆元求解，即求解 ax ≡ 1(mod b)
LL inverse(LL a, LL b)
{
    LL x, y, interval = b / exGcd(a, b, x, y);
    return ((x % interval) + interval) % interval;
}

struct Factor {
    int p, k;
} factors[20];
int factors_len = 0;
// 分解质因数
void factorize(LL n)
{
    // 建立素数表
    find_prime(n);

    // 再次初始化，如果去掉本行，factors_len 会改变成其他值，可能因为内存泄漏，暂时不管
    factors_len = 0;

    // 分解质因子
    for (int i = 0; i < primes_cnt; i++)
    {
        if (primes[i] <= (int) sqrt(n * 1.0) && n % primes[i] == 0)
        {
            factors[factors_len].p = primes[i];
            factors[factors_len].k = 0;
            while (n % primes[i] == 0)
            {
                n /= primes[i];
                factors[factors_len].k++;
            }
            factors_len++;
        }
    }
    // 如有剩余
    if (n != 1)
    {
        factors[factors_len].p = n;
        factors[factors_len++].k = 1;
    }
}


// <<<<<<问题 1：计算 prime counts in a factorial，即 n! 中有多少个质因子 p>>>>>>

int pcf_recursion(int n, int p)
{
    if (n >= p) return n / p + pcf_recursion(n / p, p);
    else return 0;
}
int pcf_iteration(int n, int p)
{
    int cnt = 0;
    while (n)
    {
        cnt += n / p;
        n /= p;
    }
    return cnt;
}

LL cbn[MAXN][MAXN] = {};  // 组合数计算记忆优化所用到的表

// <<<<<<问题 2：组合数 C_n^m 的计算>>>>>>

// 递归 + 记忆优化
LL cbn_recursion(LL n, LL m)
{
    if (m == 0 || n == m) return 1;
    else if (cbn[n][m] != 0) return cbn[n][m];
    else return cbn[n][m] = cbn_recursion(n - 1, m - 1) + cbn_recursion(n - 1, m);
}
// 递推 + 记忆优化
LL cbn_iteration(LL n, LL m)
{
    // 初始化边界值
    for (int i = 1; i <= n; i++)
        cbn[i][0] = cbn[i][i] = 1;
    // 递推计算
    for (int i = 2; i <= n; i++)
        for (int j = 1; j <= i / 2; j++)
        {
            cbn[i][j] = cbn[i-1][j-1] + cbn[i-1][j];
            cbn[i][i - j] = cbn[i][j];  // 优化
        }
    return cbn[n][m];
}
// 定义式分解，边乘边除，时间复杂度：O(n)
LL cbn_defination(LL n, LL m)
{
    LL ans = 1;
    for (LL i = 1; i <= m; i++)
        ans = ans * (n - m + i) / i;  // 必须先乘后除，不然不能够整除
        // 不能写成：“ans *= (n - m + i) / i;” ≡ “ans = ans *((n - m + i) / i);”
    return ans;
}

// <<<<<<问题 3：C_n^m % p 的计算>>>>>>

// 递归 + 记忆优化
LL cbn_recursion(LL n, LL m, LL p)
{
    if (m == 0 || n == m) return 1;
    if (cbn[n][m] != 0) return cbn[n][m];
    return cbn[n][m] = (cbn_recursion(n - 1, m) + cbn_recursion(n - 1, m - 1)) % p;
}
// 递推 + 记忆优化
LL cbn_iteration(LL n, LL m, LL p)
{
    for (int i = 0; i <= n; i++)
        cbn[i][0] = cbn[i][i] = 1;
    for (int i = 2; i <= n; i++)
        for (int j = 1; j <= i / 2; j++)
        {
            cbn[i][j] = (cbn[i-1][j-1] + cbn[i-1][j]) % p;
            cbn[i][i - j] = cbn[i][j];  // 优化
        }
    return cbn[n][m];
}
// 定义式 + 阶乘的质因子分解
LL cbn_defination(LL n, LL m, LL p)
{
    LL ans = 1;
    find_prime(n);
    for (int i = 0; i < primes_cnt; i++)
    {
        // 找到 cbn 中质因子 primes[i] 的个数 p_cnt
        LL p_cnt = pcf_recursion(n, primes[i]) - pcf_recursion(m, primes[i]) - pcf_recursion(n - m, primes[i]);
        // 快速幂计算 primes[i] ^ p_cnt % p
        ans = (ans * fastPower(primes[i], p_cnt, p)) % p;
    }
    return ans;
}
// 特殊情况 1（m < p, p 是素数）：利用逆元求解
LL cbn_inverse(LL n, LL m, LL p)
{
    LL ans = 1;
    for (int i = 1; i <= m; i++)
        ans = ans * (n - m + i) % p * inverse(i, p) % p;
    return ans;
}
// 特殊情况 2（m 任意，p 是素数）：去除分子分母中多余素数 p + 边乘边除 + 逆元求解
LL cbn_remove_p(LL n, LL m, LL p)
{
    LL ans = 1, cnt_p = 0;
    for (int i = 1; i <= m; i++)
    {
        LL tmp = n - m + i;  // numerator
        while (tmp % p == 0)
        {
            tmp /= p;
            cnt_p++;
        }
        ans = ans * tmp % p;

        tmp = i; // denominator
        while (tmp % p == 0)
        {
            tmp /= p;
            cnt_p--;
        }
        ans = ans * inverse(tmp, p) % p;
    }
    if (cnt_p > 0) return 0;
    else return ans;
}
// 特殊情况 3（m，p 均任意）：① 对 p 进行质因子分解（下面用此法）；② 对分子分母中每一项都进行质因子分解
LL cbn_factorization(LL n, LL m, LL p)
{
    // 分解质因子
    factorize(p);

    LL ans = 1, p_cnt[20];
    // 初始化 p_cnt
    for (int i = 0; i < factors_len; i++)
        p_cnt[i] = 0;

    for (int i = 1; i <= m; i++)  // 遍历每一对分子，分母
    {
        LL tmp = n - m + i;  // numerator
        for (int j = 0; j < factors_len; j++)  // 遍历 p 中的每一个质因子
        {
            while (tmp % factors[j].p == 0)
            {
                p_cnt[j]++;
                tmp /= factors[j].p;
            }
        }
        ans = ans * tmp % p;  // 乘上分子

        tmp = i;  // denominator
        for (int j = 0; j < factors_len; j++)
        {
            while (tmp % factors[j].p == 0)
            {
                p_cnt[j]--;
                tmp /= factors[j].p;
            }
        }
        ans = ans * inverse(tmp, p) % p;  // 乘上分母的逆元
    }

    // 处理多余的一些质因子
    for (int i = 0; i < factors_len; i++)
    {
        if (p_cnt[i] != 0)
            ans = ans * fastPower(factors[i].p, p_cnt[i], p) % p;
    }
    return ans;
}
// Lucas 定理
LL cbn_lucas(LL n, LL m, LL p)
{
    if (m == 0) return 1;
    else return cbn_inverse(n % p, m % p, p) * cbn_lucas(n / p, m / p, p);
    // 其中的 cbn_inverse 可以改成各种 cbn 计算函数
}

int main()
{
    factorize(120);
    cout << pcf_recursion(10, 2) << ' ' << pcf_iteration(10, 2)  << endl
         << cbn_recursion(41, 17) << ' ' << cbn_iteration(41, 17) << ' ' << cbn_defination(41, 17) << endl;
    memset(cbn, 0, sizeof(cbn));
    cout << cbn_recursion(41, 17, 29) << ' ' << cbn_iteration(41, 17, 29) << endl;
    memset(cbn, 0, sizeof(cbn));
    cout << cbn_defination(41, 17, 29) << ' ' << cbn_inverse(41, 17, 29) << endl;
    cout << cbn_remove_p(41, 17, 29) << ' ' << cbn_factorization(10, 5, 29) << ' ' << cbn_lucas(41, 17, 29);
    return 0;
}
