// 扩展欧几里得算法 Extended Euclidean algorithm
// 求解方程 ax + by = gcd(a, b)
#include <iostream>
using namespace std;

// 得到其中一组解
int exGcd(int a, int b, int& x, int& y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return a;
    }
    else
    {
        int gcd = exGcd(b, a % b, x, y), tmp = x;
        x = y;
        y = tmp - (a / b) * y;
        return gcd;
    }
}
// 输出全部解。(x, y) 是组已知解，输出已知解左右共 n 个解
void output_all(int a, int b, int n)
{
    int x, y, g = exGcd(a, b, x, y);
    for (int i = -n / 2; i <= n / 2; i++)
        printf("(%d, %d)\n", x + b / g * i, y - a / g * i);
}
// 得到 x 的最小正整数解
void get_min_positive(int a, int b, int& x, int& y)
{
    int gcd = exGcd(a, b, x, y),
        b_gcd = b / gcd;
    x = (x % b_gcd + b_gcd) % b_gcd;
    y = (gcd - a * x) / b;
}

int main()
{
    int a, b;
    cin >> a >> b;
    output_all(a, b, 10);
    int x, y;
    get_min_positive(a, b, x, y);
    cout << x << ' ' << y << endl;
    return 0;
}
