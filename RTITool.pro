#-------------------------------------------------
#
# Project created by QtCreator 2017-03-14T16:40:09
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

QMAKE_LFLAGS += -Wl,-rpath,\\$\$ORIGIN/lib/:\\$\$ORIGIN/../mylib/

TARGET = RTITool
TEMPLATE = app

LIBS += `pkg-config opencv --libs`

SOURCES += main.cpp\
        rtitool.cpp \
    imageview.cpp \
    s_hull_pro.cpp

HEADERS  += rtitool.h \
    imageview.h \
    s_hull_pro.h

FORMS    += rtitool.ui \
    imageview.ui
