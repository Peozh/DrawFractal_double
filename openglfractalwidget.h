#ifndef OPENGLFRACTALWIDGET_H
#define OPENGLFRACTALWIDGET_H

#include <QWidget>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QWheelEvent>
#include <vector>
#include <complex>
#include <random>

class OpenGLFractalWidget : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT

    GLuint textureId;
    double left = -1.0;
    double right = 1;
    double top = -1.0;
    double bottom = 1;
    double scale;
    double xPivot;
    double yPivot;

    bool exponentChangeMode = false;
    QPointF dragStartPoint;
    double dragStartExponent;
    float* host_texture = nullptr;
    float* dev_texture = nullptr;

public:
    int max_iteration = 100;
    int xPixels = 1000;
    int yPixels = 1000;
    std::complex<double> constant;
    double exponent;
    bool log_expression;


public:
    explicit OpenGLFractalWidget(QWidget *parent = nullptr);
    ~OpenGLFractalWidget();
    virtual void mousePressEvent(QMouseEvent *event) override;
    virtual void mouseMoveEvent(QMouseEvent *event) override;
    virtual void mouseReleaseEvent(QMouseEvent *event) override;
    virtual void wheelEvent(QWheelEvent *event) override;


protected:
    virtual void initializeGL() override;
    virtual void resizeGL(int w, int h) override;
    virtual void paintGL() override;

private:
    void zoomIn(QPointF position);
    void zoomOut(QPointF position);
    void render(const int xPixels, const int yPixels);

signals:
    void constantChanged(float real, float imag);
    void exponentChanged(float exponent);
public slots:
    void updateTexture();
};

#endif // OPENGLFRACTALWIDGET_H
