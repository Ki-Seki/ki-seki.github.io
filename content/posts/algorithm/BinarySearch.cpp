/*
1. `二分查找的本质：查询有序序列第一个满足（或最后一个不满足）给定条件元素的位置`

2. `要注意的关键点：`

   * `while 循环中是 left <= right or left < right`
   * `接收参数 left，right 所代表的区间开闭`
   * `判断时的 array[mid] > or < or >= or <= x`
   * `不满足情况时的返回值`
   * `返回值返回什么`
*/

#include <iostream>
#include <algorithm>

// search x in a[] from left to right (both ends included)
// return -1 if nothing was found
int binarySearch(int x, int a[], int left, int right)
{
    int mid;
    while (left <= right)  // to make mid cover all the possible points
    {
        mid = left + (right - left) / 2;  // using (left + right) / 2 might cause overflow
        if (a[mid] == x)
            return mid;
        else if (a[mid] < x)
            left = mid + 1;
        else
            right = mid -1;
    }
    return -1;
}

void binarySearchTest()
{
    const int n = 10;
    int a[] = {-2, 0, 1, 2, 34, 56, 999, 1990, 11999, 12000};
    printf("Array: {-2, 0, 1, 2, 34, 56, 999, 1990, 11999, 12000}\n");
    printf("Search %d: %d\n", 9, binarySearch(9, a, 0, n - 1));
    printf("Search %d: %d\n", 0, binarySearch(0, a, 0, n - 1));
    printf("Search %d: %d\n", -5, binarySearch(-5, a, 4, n - 1));
    printf("Search %d: %d\n", 100000, binarySearch(100000, a, 0, n - 1));
    printf("Search %d: %d\n", 56, binarySearch(56, a, 0, n - 7));
    printf("Search %d: %d\n", -2, binarySearch(-2, a, 0, n - 1));
    printf("Search %d: %d\n", 12000, binarySearch(12000, a, 0, n - 1));
    printf("Search %d: %d\n", 1990, binarySearch(1990, a, 0, n - 1));
}

// find the first one who is equal to x from left to right (both ends included)
// return -1 if nothing was found
int lowerBound(int x, int a[], int left, int right)
{
    if (left > right) return -1;
    int mid;
    while (left < right)  // 保证结束时 left 和 right 具有相同值
    {
        mid = left + (right - left) / 2;
        if (a[mid] < x)
            left = mid + 1;
        else
            right = mid - 1;
    }
    if (a[left] == x)
        return left;
    else
        return -1;
}

// find the first one who is equal to x from left (included) to right (excluded)
// return original right if no one bigger than x
int upperBound(int x, int a[], int left, int right)
{
    int mid, r = right;
    while (left < right)  // 保证结束时 left 和 right 具有相同值
    {
        mid = left + (right - left) / 2;
        if (a[mid] <= x)
            left = mid + 1;
        else
            right = mid;  // 不满足条件，所以不能 - 1
    }
    if (left >= r)
        return r;
    else
        return left;
}

void lowerBoundUpperBoundTest()
{
    const int n = 10;
    int a[] = {-2, 0, 2, 2, 34, 56, 999, 999, 999, 12000};
    printf("Array: {-2, 0, 2, 2, 34, 56, 999, 999, 999, 12000}\n");
    printf("Search %d: [%d, %d)\n", 9, lowerBound(9, a, 0, n - 1), upperBound(9, a, 0, n));
    printf("Search %d: [%d, %d)\n", 2, lowerBound(2, a, 0, n - 1), upperBound(2, a, 0, n));
    printf("Search %d: [%d, %d)\n", -5, lowerBound(-5, a, 4, n - 1), upperBound(-5, a, 4, n));
    printf("Search %d: [%d, %d)\n", 100000, lowerBound(100000, a, 0, n - 1), upperBound(100000, a, 0, n));
    printf("Search %d: [%d, %d)\n", 56, lowerBound(56, a, 0, n - 7), upperBound(56, a, 0, n - 6));
    printf("Search %d: [%d, %d)\n", -2, lowerBound(-2, a, 0, n - 1), upperBound(-2, a, 0, n));
    printf("Search %d: [%d, %d)\n", 12000, lowerBound(12000, a, 0, n - 1), upperBound(12000, a, 0, n));
    printf("Search %d: [%d, %d)\n", 999, lowerBound(999, a, 0, n - 1), upperBound(999, a, 0, n));
}

// 有序序列第一个满足（最后一个不满足）给定条件元素位置查询问题的模板
int sequentialProblemTemplate(int left, int right)
{
    int mid;
    bool condition; // 条件
    while (left < right)  // 最终 left == right
    {
        mid = left + (right - left) / 2;
        if (condition)  // 条件成立，且待查元素在右
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

int main()
{
    printf("BEGIN OF binarySearch() TEST\n");
    binarySearchTest();
    printf("END OF binarySearch() TEST\n");

    printf("BEGIN OF lowerBound() & upperBound() TEST\n");
    lowerBoundUpperBoundTest();
    printf("END OF lowerBound() & upperBound() TEST\n");
    return 0;
}
