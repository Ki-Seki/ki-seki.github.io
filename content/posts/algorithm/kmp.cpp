// 代码参考王道教程

#include <iostream>
#include <cstring>
#define MAXN 50
using namespace std;
char s[MAXN], t[MAXN];
int ls, lt, nxt[MAXN] = {0};

void get_next()
{
    int i = 1, j = 0;
    nxt[1] = 0;
    while (i <= ls)
    {
        if (j == 0 || s[i] == t[j])
        {
            i++; j++;

            // 初级的方法
            // nxt[i] = j;

            // 高级的方法
            if (s[i] != s[j]) nxt[i] = j;
            else nxt[i] = nxt[j];
        }
        else
            j = nxt[j];
    }
}

int kmp_index()
{
    int i = 1, j = 1;
    while (i <= ls && j <= lt)
    {
        if (j == 0 || s[i] == t[j])
        {
            i++;
            j++;
        }
        else
            j = nxt[j];
    }
    if (j > lt) return i - lt;
    else return 0;
}

int main()
{
    cin >> s+1 >> t+1;
    ls = strlen(s+1);
    lt = strlen(t+1);
    cout << kmp_index();
    return 0;
}
