#include <iostream>
typedef long long LL;
using namespace std;

struct Fraction {
    LL numerator, denominator;

    // 构造函数
    Fraction(){}
    Fraction(LL integer)
    {
        this->numerator = integer;
        this->denominator = 1;
    }
    Fraction(LL a, LL b)
    {
        this->numerator = a;
        this->denominator = b;
    }

    // 分数形式合法性判断
    bool isValid()
    {
        if (this->numerator == 0)
            return (this->denominator == 1);
        else
            return (this->denominator > 0);
    }
};

LL gcd(LL a, LL b)
{
    return (b ? gcd(b, a%b) : a);
}

// 约分化简；对非法的数据能改造则改造
Fraction reduction(Fraction n)
{
    if (n.numerator == 0)
        return Fraction(0, 1);
    else if (n.denominator < 0)
        return Fraction(n.numerator * -1, n.denominator * -1);
    else if (n.denominator == 0)
        return Fraction(n.numerator, n.denominator);
    else
    {
        LL d = gcd(n.numerator, n.denominator);
        return Fraction(n.numerator / d, n.denominator / d);
    }
}

// arithmetic
Fraction operator + (const Fraction& a, const Fraction& b)
    {
        Fraction ans;
        ans.numerator = a.numerator * b.denominator + b.numerator * a.denominator;
        ans.denominator = a.denominator * b.denominator;
        return reduction(ans);
    }
Fraction operator - (const Fraction& a, const Fraction& b)
    {
        Fraction ans;
        ans.numerator = a.numerator * b.denominator - b.numerator * a.denominator;
        ans.denominator = a.denominator * b.denominator;
        return reduction(ans);
    }
Fraction operator * (const Fraction& a, const Fraction& b)
    {
        return reduction(Fraction(a.numerator * b.numerator, a.denominator * b.denominator));
    }
Fraction operator / (const Fraction& a, const Fraction& b)
    {
        if (b.numerator == 0)  // 当除数为 0
            return Fraction(LLONG_MAX, 1);
        else
            return reduction(Fraction(a.numerator * b.denominator, a.denominator * b.numerator));
    }


ostream& operator << (ostream& out, const Fraction& f)
{
    if (f.denominator == 1) out << f.numerator;
    else out << f.numerator << " / " << f.denominator;
    return out;
}

int main()
{
    Fraction a(1, 2), b(-2, -3), c(5, 1), d(-9, 5), e(4, -8);
    cout << a + b - c * d / e;
    return 0;
}
