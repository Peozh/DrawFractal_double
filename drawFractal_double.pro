QT       += core gui openglwidgets opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    coordinatestatusbar.cpp \
    main.cpp \
    mainwindow.cpp \
    openglfractalwidget.cpp

HEADERS += \
    coordinatestatusbar.h \
    cudafractal.h \
    mainwindow.h \
    openglfractalwidget.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += \
    "C:\Coding\projects\VSCode\test_cuda\linking\includes\cuda_12.1" \

LIBS += \
    -l"opengl32" \
    -L"C:\Coding\projects\Qt\drawFractal_double" \
    -l"cudart" \
    -l"cudafractal"\

DISTFILES += \
    cudafractal.cu
