#include "openglfractalwidget.h"
#include "cudafractal.h"

#include <QFile>

OpenGLFractalWidget::OpenGLFractalWidget(QWidget *parent)
    : QOpenGLWidget{parent}
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine { rnd_device() };
    std::uniform_real_distribution<double> dist {-1.0, 1.0};
    constant.real(dist(mersenne_engine));
    constant.imag(dist(mersenne_engine));
    exponent = 2.0;
    log_expression = false;

    this->xPixels = 1000;
    this->yPixels = 1000;
    this->scale = 1.0 / double(xPixels / 2);
    this->xPivot = double(xPixels/2.0);
    this->yPivot = double(yPixels/2.0);

    myCUDA::allocateHostPinnedMemory(this->host_texture, 1920, 1080);
    myCUDA::allocateDeviceMemory(this->dev_texture, 1920, 1080);
}

OpenGLFractalWidget::~OpenGLFractalWidget()
{
    myCUDA::deleteHostPinnedMemory(this->host_texture);
    myCUDA::deleteDeviceMemory(this->dev_texture);
}

void OpenGLFractalWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->buttons() & Qt::LeftButton)
    {
        this->dragStartPoint = event->position();
        if (event->modifiers() & Qt::KeyboardModifier::AltModifier) { exponentChangeMode = true; dragStartExponent = exponent; }
    }
    else if (event->buttons() & Qt::RightButton)
    {
        this->log_expression = !log_expression;
        this->updateTexture();
    }
}

void OpenGLFractalWidget::mouseMoveEvent(QMouseEvent *event)
{
    if ((event->buttons() & Qt::LeftButton) && exponentChangeMode) // change exponent
    {
        this->exponent = dragStartExponent + (dragStartPoint.y() - event->position().y())*0.001f;
        emit exponentChanged(exponent);
        this->updateTexture();
    }
    else if ((event->buttons() & Qt::LeftButton))  // change constant
    {
        this->constant.real((event->position().x() - this->xPixels/2)/(this->xPixels/2));
        this->constant.imag((event->position().y() - this->yPixels/2)/(this->yPixels/2));
        emit constantChanged(constant.real(), constant.imag());
        this->updateTexture();
    }
}

void OpenGLFractalWidget::mouseReleaseEvent(QMouseEvent *event)
{
    exponentChangeMode = false;
}

void OpenGLFractalWidget::wheelEvent(QWheelEvent *event)
{
    if(event->angleDelta().y() > 0) // up Wheel
        zoomIn(event->position());
    else if(event->angleDelta().y() < 0) //down Wheel
        zoomOut(event->position());

    this->updateTexture();
}

void OpenGLFractalWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(0, 0, 0, 1.0f);

    glGenTextures(1, &textureId); // 텍스쳐 id 에 텍스쳐를 하나 할당합니다.
    glActiveTexture(GL_TEXTURE0); // 활성화할 텍스쳐 슬롯을 지정합니다.
    glBindTexture(GL_TEXTURE_2D, textureId); // 현재 활성화된 텍스쳐 슬롯에 실질 텍스쳐를 지정합니다.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, xPixels, yPixels, 0, GL_RED, GL_FLOAT, this->host_texture); // 텍스쳐 이미지가 RED 단일 채널이며, float 입니다. border 는 0 만 유효합니다.
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER); // s 축의 비어있는 텍스쳐 외부를 border 색상으로 채웁니다.
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER); // t 축의 비어있는 텍스쳐 외부를 border 색상으로 채웁니다.
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // 텍스쳐 확대 시 fragment 를 최근접 값으로 설정합니다.
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // 텍스쳐 축소 시 fragment 를 최근접 값으로 설정합니다.
    float colour[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, colour); // 텍스쳐 border 색상을 결정합니다.
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); // (default)

    glEnable(GL_DEPTH_TEST); // 최적화 : Depth 테스트 실패 시 그려지지 않음
    glDepthFunc(GL_LEQUAL); // 겹치거나 가까우면 그리기
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE); // 스텐실 테스트와 뎁스 테스트 모두 통과 시 stencil buffer 를 glStencilFunc 에서 지정한 ref 로 설정합니다. 나머지 경우 유지.

    glEnable(GL_BLEND); // 아래에서 설정할 블렌딩 효과 활성화
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 전면(SRC)과 뒷면(DST)에 각각 곱해줄 계수
    glBlendEquation(GL_FUNC_ADD); // 위에서 얻은 두 항 간의 연산 방법 (default)

    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);

    glEnable(GL_CULL_FACE); // 최적화 : 후면 제거 활성화
    glFrontFace(GL_CCW); // 전면/후면 판단 기준 (default)
    glCullFace(GL_BACK); // 후면만 폐기 (default)
}

void OpenGLFractalWidget::resizeGL(int w, int h)
{
    this->xPixels = w;
    this->yPixels = h;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(1.0*left, 1.0*right, 1.0*top, 1.0*bottom, -1, 1);

    glMatrixMode(GL_MODELVIEW); // model transformation
    glLoadIdentity();
//    glRotated(90, 1, 1, 1);
    updateTexture();
}

void OpenGLFractalWidget::paintGL()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(1.0*left, 1.0*right, 1.0*top, 1.0*bottom, -1, 1);

    glMatrixMode(GL_MODELVIEW); // model transformation
    glLoadIdentity();
//    glRotated(90, 1, 1, 1);

    GLfloat backgroundVertices[4][3] = {
        {-1, -1, 0},
        { 1, -1, 0},
        { 1,  1, 0},
        {-1,  1, 0} };
    GLubyte VertexOrder[4] = { 0, 1, 2, 3 };
    GLfloat texture2DCoords[4][2] = {
        {0, 1},
        {1, 1},
        {1, 0},
        {0, 0}
    };

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // draw background
    glStencilMask(0x00);
    // 바닥을 그리는 동안에는 stencil buffer를 수정하지 않습니다
    // stencil buffer 작성 (1 & 0x00 = 0) 비활성화
    glEnable(GL_TEXTURE_2D);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glVertexPointer(3, GL_FLOAT, 0, backgroundVertices);
    glTexCoordPointer(2, GL_FLOAT, 0, texture2DCoords);
    glDrawElements(GL_POLYGON, 4, GL_UNSIGNED_BYTE, VertexOrder);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_TEXTURE_2D);
}

void OpenGLFractalWidget::zoomIn(QPointF position)
{
    double zoomRatio = 0.9;

    double x_left_old = -xPivot * scale;
    double x_right_old = double(xPixels - xPivot) * scale;
    double xRatio = position.x()/this->xPixels; // [0, 1]
    double xCoord = x_left_old*(1.0-xRatio) + x_right_old*xRatio; // [left, right]
    double x_left_new = x_left_old*zoomRatio + xCoord*(1.0-zoomRatio);

    double y_top_old = -yPivot * scale;
    double y_bottom_old = double(yPixels - yPivot) * scale;
    double yRatio = position.y()/this->yPixels; // [0, 1]
    double yCoord = y_top_old*(1.0-yRatio) + y_bottom_old*yRatio; // [top, bottom]
    double y_top_new = y_top_old*zoomRatio + yCoord*(1.0-zoomRatio);

    this->scale *= zoomRatio;
    this->xPivot = -x_left_new/scale;
    this->yPivot = -y_top_new/scale;
}

void OpenGLFractalWidget::zoomOut(QPointF position)
{
    double zoomRatio = 0.9;

    double x_left_old = -xPivot * scale;
    double x_right_old = double(xPixels - xPivot) * scale;
    double xRatio = position.x()/this->xPixels; // [0, 1]
    double xCoord = x_left_old*(1.0-xRatio) + x_right_old*xRatio; // [left, right]
    double x_left_new = (x_left_old-xCoord)/zoomRatio + xCoord;

    double y_top_old = -yPivot * scale;
    double y_bottom_old = double(yPixels - yPivot) * scale;
    double yRatio = position.y()/this->yPixels; // [0, 1]
    double yCoord = y_top_old*(1.0-yRatio) + y_bottom_old*yRatio; // [top, bottom]
    double y_top_new = (y_top_old-yCoord)/zoomRatio + yCoord;

    this->scale /= zoomRatio;
    this->xPivot = -x_left_new/scale;
    this->yPivot = -y_top_new/scale;
}

void OpenGLFractalWidget::render(const int xPixels, const int yPixels)
{
    this->xPixels = xPixels;
    this->yPixels = yPixels;
    myCUDA::generateTexture(this->host_texture, this->dev_texture, xPixels, yPixels, exponent, { constant.real(), constant.imag() }, max_iteration, xPivot, yPivot, scale, log_expression);
}

void OpenGLFractalWidget::updateTexture()
{
    render(xPixels, yPixels);
    glActiveTexture(GL_TEXTURE0); // 활성화할 텍스쳐 슬롯을 지정합니다.
    glBindTexture(GL_TEXTURE_2D, textureId); // 현재 활성화된 텍스쳐 슬롯에 실질 텍스쳐를 지정합니다.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, xPixels, yPixels, 0, GL_RED, GL_FLOAT, this->host_texture); // 텍스쳐 이미지가 RED 단일 채널이며, float 입니다. border 는 0 만 유효합니다.}
    this->update();
}

