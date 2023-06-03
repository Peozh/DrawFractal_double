#ifndef COORDINATESTATUSBAR_H
#define COORDINATESTATUSBAR_H

#include <QStatusBar>
#include <QWidget>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>

class CoordinateStatusBar : public QStatusBar
{
    Q_OBJECT

    QLabel* exponentValue;
    QLabel* constantValue;
    std::vector<QLabel*> labels;  // Zn+1 = Zn^(x) + C
public:
    CoordinateStatusBar(QWidget* parent = nullptr);


public slots:
    void setExponentValue(float exponent);
    void setConstantValue(float real, float imag);
};

#endif // COORDINATESTATUSBAR_H
