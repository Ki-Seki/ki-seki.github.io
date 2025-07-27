#include <iostream>

using namespace std;

// 最大公倍数的核心要点是:
// 1. a, b, a%b 三个数的公约数相同
// 2. a%b < a, a%b < b
// 3. 假设 gcd(a, b) 中 a >= b 会使问题简化
// 4. 即使初始时 a < b, (a, b) -> (b, a%b) 后也会重回标准
int gcd(int a, int b)
{
    return (!b ? a : gcd(b, a%b));
}

int lcm(int a, int b)
{
    return a / gcd(a, b) * b;
}

int main()
{
    cout << lcm(1, 3) << lcm(2, 4) << lcm(3, 1) << lcm(4, 2);
    return 0;
}
