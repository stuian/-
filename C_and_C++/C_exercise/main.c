#include <stdio.h>
#include <stdlib.h>

int main()
{
        int array[2][3] = {{1,2,3},{4,5,6}};
        int (*p)[3] = array;

        // 同一个地址
        printf("array:%p\n",array);
        printf("array[0]:%p\n",array[0]);
        printf("&array[0][0]:%p\n",&array[0][0]);
        printf("\n");

        //同一个地址
        printf("*(array+1):%p\n",*(array+1));
        printf("array[1]:%p\n",array[1]);
        printf("&array[1][0]:%p\n",&array[1][0]);
        printf("\n");

        printf("*(*(array+1)+2):%p\n",*(*(array+1)+2));
        printf("array[1][2]:%p\n",array[1][2]);
        printf("array[1][2]:%d\n",array[1][2]);
        printf("\n");

        //指针操作数组
        printf("**p:%d\n",**(p+1));
        printf("**(array+1):%d\n",**(array+1));
        printf("array[1][0]:%d\n",array[1][0]);

        printf("*(*(p+1)+2): %d\n", *(*(p+1)+2));
        printf("*(*(array+1)+2): %d\n", *(*(array+1)+2));
        printf("array[1][2]: %d\n", array[1][2]);

        return 0;
}
