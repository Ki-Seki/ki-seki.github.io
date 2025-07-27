#include <iostream>
#include <algorithm>
#include <cstdlib>  // 用于随机生成待排序的测试数据
#include <cmath>  // 为使用 round
#define MAXN 10005  // 待排序数组的元素大小范围
using namespace std;

// 选择排序：一趟结束，找出最值放在前面，start 和 end 左闭右开
void select_sort(int *start, int *end)
{
    for (int i = 0; start + i < end; i++)  // 从 start + i 到 end 是 待排部分
    {
        int min = i;
        for (int j = i + 1; start + j < end; j++)  // 找到最小值下标
            if (*(start + j) < *(start + min))
                min = j;
        if (start[min] != start[i])  // 当找到的最小值需要交换时
        {
            int tmp = start[min];
            start[min] = start[i];
            start[i] = tmp;
        }
    }
}

// 插入排序：每一趟找到一个合适的值，插入前面，其余值后移，start 和 end 左闭右开
void insertion_sort(int *start, int *end)
{
    for (int i = 0; start + i < end; i++)  // 从 start + i 到 end 是 待排部分
    {
        int tmp = start[i],  // 取待排的首个
        pos = i - 1;  // 找到要插入的位置
        for (; pos >= 0; pos--)
            if (start[pos] > tmp)
                start[pos+1] = start[pos];  // 后移以腾开要插入的位置
            else
                break;  // 找到位置
        start[pos+1] = tmp;  // 插入
    }
}

// 归并排序：二分 + 归并的思想

// merge() 是辅助函数，归并一个数组的任意两个不相交的部分
// 假设其中 l2 = r1 + 1
void merge(int a[], int l1, int r1, int l2, int r2)
{
    int i = l1, j = l2;
    int tmp[MAXN], index = 0;
    while (i <= r1 && j <= r2)
        if (a[i] < a[j]) tmp[index++] = a[i++];
        else tmp[index++] = a[j++];
    while (i <= r1)
        tmp[index++] = a[i++];
    while (j <= r2)
        tmp[index++] = a[j++];
    memcpy(a + l1, tmp, sizeof(int) * index);
}
// 递归实现的归并排序，left 和 right 左闭右闭
void merge_sort_recursion(int a[], int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;
        merge_sort_recursion(a, left, mid);
        merge_sort_recursion(a, mid + 1, right);
        merge(a, left, mid, mid + 1, right);
    }
}
// 迭代实现的归并排序，n 是数组长度
void merge_sort_iteration(int a[], int n)
{
    for (int step = 2; step / 2 <= n; step *= 2)
    {
        for (int l1 = 0; l1 < n; l1 += step)  // 左区间的左端点按照 step 来遍历
        {
            int r1 = l1 + step / 2 - 1;
            if (r1 + 1 < n)  // 当右区间存在元素，则合并
                // 左区间 [l1, r1]，右区间 [r1 + 1, min(l1 + step - 1, n - 1)]
                merge(a, l1, r1, r1 + 1, min(l1 + step - 1, n - 1));
        }
    }
}

// 快速排序：按主元分割数组，分而治之

// 分割是辅助函数，返回最终主元下标，left 和 right 左闭右闭
int partition(int a[], int left, int right)
{
    // 删去前两行，将固定 a[left] 为主元
    int p = round(1.0 * rand() / RAND_MAX * (right - left) + left);
    swap(a[p], a[left]);

    int temp = a[left];
    while (left < right)
    {
        while (left < right && a[right] > temp) right--;
        a[left] = a[right];
        while (left < right && a[left] <= temp) left++;
        a[right] = a[left];
    }
    a[left] = temp;
    return left;
}

// 分割函数的另一种写法
// decreasingly partition the array in [left, right]
int randPartition(int array[], int left, int right)
{
    int p = rand() % (right - left) + left;
    swap(array[p], array[left]);

    int prime = left++;  // 主元为初始的 left 值，left 值 然后向后位移一位
    while (left < right)  // until left == right
    {
        while (left < right && array[left] >= array[prime]) left++;
        while (left < right && array[right] < array[prime]) right--;
        swap(array[left], array[right]);
    }
    swap(array[prime], array[left - 1]);  // 交换主元到中间
    return left;
}

// 快速排序主函数，left 和 right 左闭右闭
void quick_sort(int a[], int left, int right)
{
    if (left < right)
    {
        // pos 为分割点
        int pos = partition(a, left, right);
        quick_sort(a, left, pos - 1);
        quick_sort(a, pos + 1, right);
    }
}

// 测试 ↓

// 根据 seed 生成随机的 n 个随机数
void gen_data(int data[], unsigned seed, int n)
{
    srand(seed);
    for (int i = 0; i < n; i++)
        data[i] = rand() - (RAND_MAX / 2);
}

void output_array(int data[], int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", data[i]);
    printf("\n");
}

void sort_test()
{
    int data[MAXN];
    // test 1
    {
        int seed = 23, n = 9;
        gen_data(data, seed, n);
        printf("test data: "); output_array(data, n);
        select_sort(data, data + n);
        printf("ss   "); output_array(data, n);
        gen_data(data, seed, n); insertion_sort(data, data + n);
        printf("is   "); output_array(data, n);
        gen_data(data, seed, n); merge_sort_recursion(data, 0, n - 1);
        printf("msr  "); output_array(data, n);
        gen_data(data, seed, n); merge_sort_iteration(data, n);
        printf("msi  "); output_array(data, n);
        gen_data(data, seed, n); quick_sort(data, 0, n - 1);
        printf("qs   "); output_array(data, n);
        printf("\n");
    }
    // test 2
    {
        int seed = 3, n = 15;
        gen_data(data, seed, n);
        printf("test data: "); output_array(data, n);
        select_sort(data, data + n);
        printf("ss   "); output_array(data, n);
        gen_data(data, seed, n); insertion_sort(data, data + n);
        printf("is   "); output_array(data, n);
        gen_data(data, seed, n); merge_sort_recursion(data, 0, n - 1);
        printf("msr  "); output_array(data, n);
        gen_data(data, seed, n); merge_sort_iteration(data, n);
        printf("msi  "); output_array(data, n);
        gen_data(data, seed, n); quick_sort(data, 0, n - 1);
        printf("qs   "); output_array(data, n);
        printf("\n");
    }
    // test 3
    {
        int seed = 1, n = 4;
        gen_data(data, seed, n);
        printf("test data: "); output_array(data, n);
        select_sort(data, data + n);
        printf("ss   "); output_array(data, n);
        gen_data(data, seed, n); insertion_sort(data, data + n);
        printf("is   "); output_array(data, n);
        gen_data(data, seed, n); merge_sort_recursion(data, 0, n - 1);
        printf("msr  "); output_array(data, n);
        gen_data(data, seed, n); merge_sort_iteration(data, n);
        printf("msi  "); output_array(data, n);
        gen_data(data, seed, n); quick_sort(data, 0, n - 1);
        printf("qs   "); output_array(data, n);
        printf("\n");
    }
    // test 4
    {
        int seed = 8967, n = 10;
        gen_data(data, seed, n);
        printf("test data: "); output_array(data, n);
        select_sort(data, data + n);
        printf("ss   "); output_array(data, n);
        gen_data(data, seed, n); insertion_sort(data, data + n);
        printf("is   "); output_array(data, n);
        gen_data(data, seed, n); merge_sort_recursion(data, 0, n - 1);
        printf("msr  "); output_array(data, n);
        gen_data(data, seed, n); merge_sort_iteration(data, n);
        printf("msi  "); output_array(data, n);
        gen_data(data, seed, n); quick_sort(data, 0, n - 1);
        printf("qs   "); output_array(data, n);
        printf("\n");
    }
}

int main()
{
    sort_test();
    return 0;
}
