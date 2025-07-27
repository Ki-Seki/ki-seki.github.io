/*
 * 算法竞赛提示：
 * * 接口规范：非负整数；无前导的 0；对于减法，确保减数大于被减数
 * * 使用最简单的 C++ 语法
 * * 数组下标与数字的高位与低位相对应，以简化计算
 * * 充分使用 carry，remainder 等变量
*/

#include <iostream>
#include <cstring>
#define MAXN 1000  // 大整数的最大位数

// only represents non-negatives
struct BigInteger {
    int digits[MAXN];
    int len;

    BigInteger()
    {
        memset(digits, 0, sizeof(digits));
        len = 0;
    }
};

// c-string to BigInteger
// hypothesis: no 0s at the beginning of the str
BigInteger s2i(char str[])
{
    BigInteger a;
    a.len = strlen(str);
    for (int i = a.len - 1; i >= 0; i--)
        a.digits[a.len - i - 1] = str[i] - '0';
    return a;
}

// comparison: return a - b
int cmp(BigInteger a, BigInteger b)
{
    if (a.len > b.len) return 1;
    else if (a.len < b.len) return -1;
    else for (int i = a.len - 1; i >= 0; i--)
        if (a.digits[i] > b.digits[i]) return 1;
        else if (a.digits[i] < b.digits[i]) return -1;
    return 0;
}

// arithmetic
// hypothesis for minus: a >= b

BigInteger operator + (BigInteger a, BigInteger b)
{
    BigInteger sum;
    int carry = 0;
    for (int i = 0; i < a.len || i < b.len; i++)
    {
        sum.digits[sum.len++] = (a.digits[i] + b.digits[i] + carry) % 10;
        carry = (a.digits[i] + b.digits[i] + carry) / 10;
    }
    if (carry != 0) sum.digits[sum.len++] = carry;
    return sum;
}
BigInteger operator - (BigInteger a, BigInteger b)
{
    BigInteger diff;
    for (int i = 0; i < a.len; i++)
    {
        diff.digits[i] = a.digits[i] - b.digits[i];
        if (diff.digits[i] < 0)  // 借位
        {
            diff.digits[i] += 10;
            a.digits[i + 1] -= 1;
        }
        diff.len += 1;
    }

    // get rid of 0s at the beginning of diff
    while (diff.len > 1 && diff.digits[diff.len - 1] == 0) diff.len--;
    return diff;
}
BigInteger operator * (BigInteger a, int b)
{
    BigInteger prod;
    long long carry;
    for (int i = 0; i < a.len; i++)
    {
        carry += a.digits[i] * b;
        prod.digits[prod.len++] = carry % 10;
        carry /= 10;
    }
    while (carry != 0)
    {
        prod.digits[prod.len++] = carry % 10;
        carry /= 10;
    }
    return prod;
}
BigInteger operator / (BigInteger a, int b)
{
    BigInteger quot;
    int remainder = 0;
    quot.len = a.len;
    for (int i = a.len; i >= 0; i--)
    {
        remainder += a.digits[i];
        quot.digits[i] = remainder / b;
        remainder = (remainder % b) * 10;
    }

    // get rid of 0s at the beginning of diff
    while (quot.len > 1 && quot.digits[quot.len - 1] == 0) quot.len--;
    return quot;
}

// output
void output(const BigInteger& n)
{
    for (int i = n.len - 1; i >= 0; i--)
        printf("%d", n.digits[i]);
}

int main()
{
    char sa[MAXN], sb[MAXN];
    int c, d;
    scanf("%s %s %d %d", sa, sb, &c, &d);
    BigInteger a = s2i(sa), b = s2i(sb);
    BigInteger ans = (a + b) / c + (a - b) * d;
    output(ans);
    return 0;
}

/* test data
1223456789012345678901234567890 5 5 30
that is (1223456789012345678901234567890 + 5) / 5 + (1223456789012345678901234567890 - 5) * 30
*/
