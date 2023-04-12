#include "superhdr.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	SuperHDR w;
	w.show();
	return a.exec();
}
