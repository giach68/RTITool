#include "rtitool.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    RTITool w;
    w.show();

    return a.exec();
}
