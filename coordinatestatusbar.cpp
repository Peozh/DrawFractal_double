#include "coordinatestatusbar.h"

CoordinateStatusBar::CoordinateStatusBar(QWidget* parent) : QStatusBar(parent)
{
    // Zn+1 = Zn^X + C
    QLabel* Z = new QLabel(this); Z->setText("Z");
    QLabel* _np1 = new QLabel(this); _np1->setText("n+1"); auto font = _np1->font(); font.setPointSize(7); _np1->setFont(font); _np1->setAlignment(Qt::AlignBottom);
    QLabel* eq = new QLabel(this); eq->setText(" = Z");
    QLabel* _n = new QLabel(this); _n->setText("n"); font = _n->font(); font.setPointSize(7); _n->setFont(font); _n->setAlignment(Qt::AlignBottom);
    QLabel* XpC = new QLabel(this); XpC->setText("^X + C");
    this->labels.push_back(Z);
    this->labels.push_back(_np1);
    this->labels.push_back(eq);
    this->labels.push_back(_n);
    this->labels.push_back(XpC);

    QString exponentString = QString("\tX = ") + "+" + QString::number(2.0, 'f', 6);
    this->exponentValue = new QLabel(this); exponentValue->setText(exponentString);
    QString constantString = QString("\tX = ") + "+" + QString::number(0.0, 'f', 6) + " +i" + QString::number(0.0, 'f', 6);
    this->constantValue = new QLabel(this); constantValue->setText(constantString);
    this->labels.push_back(exponentValue);
    this->labels.push_back(constantValue);

    for (auto* label : labels) { /*auto f = label->font(); f.setFamily( setFont()*/ this->addPermanentWidget(label, 0); }
    this->addPermanentWidget(new QLabel(this), 1);
    this->layout()->setSpacing(0);
    this->layout()->setContentsMargins(10, 0, 0, 3);
//    this->layout()->setAlignment(Qt::AlignBottom);
    this->setStyleSheet("QStatusBar::item { border: none; };");
}

void CoordinateStatusBar::setExponentValue(float exponent)
{
    QString exponentString = QString("\tX = ") + ((exponent < 0)? "-" : R"(+)") + QString::number(float(abs(exponent)), 'f', 6);// arg(qFabs(x),5,'f',2,'0');
    this->exponentValue->setText(exponentString);
}

void CoordinateStatusBar::setConstantValue(float real, float imag)
{
    QString constantString = QString("\tC = ") + ((real < 0)? "-" : R"(+)") + QString::number(float(abs(real)), 'f', 6) + ((imag < 0)? " -" : " +") + "i" + QString::number(float(abs(imag)), 'f', 6);
    this->constantValue->setText(constantString);
}
