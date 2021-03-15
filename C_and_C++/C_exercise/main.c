#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>

int main()
{
        char *p1[5] = {
                "让编程改变世界 -- 鱼C工作室",
                "Just do it -- NIKE",
                "一切皆有可能 -- 李宁",
                "永不止步 -- 安踏",
                "One more thing... -- 苹果"
        };
        int i;

        for (i = 0; i < 5; i++)
        {
                printf("%s\n", p1[i]);
        }

        return 0;
}
