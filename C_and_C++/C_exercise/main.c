#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>

int main()
{
        char *p1[5] = {
                "�ñ�̸ı����� -- ��C������",
                "Just do it -- NIKE",
                "һ�н��п��� -- ����",
                "����ֹ�� -- ��̤",
                "One more thing... -- ƻ��"
        };
        int i;

        for (i = 0; i < 5; i++)
        {
                printf("%s\n", p1[i]);
        }

        return 0;
}
