#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->openGLWidget, SIGNAL(constantChanged(float,float)), ui->statusbar, SLOT(setConstantValue(float,float)));
    connect(ui->openGLWidget, SIGNAL(exponentChanged(float)), ui->statusbar, SLOT(setExponentValue(float)));

    ui->openGLWidget->constantChanged(ui->openGLWidget->constant.real(), ui->openGLWidget->constant.imag());
    ui->openGLWidget->exponentChanged(ui->openGLWidget->exponent);
}

MainWindow::~MainWindow()
{
    delete ui;
}

