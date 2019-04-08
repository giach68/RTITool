#include "imageview.h"
#include "ui_imageview.h"
#include <QFileDialog>
#include <QString>
#include <QMessageBox>
#include <QRubberBand>
#include <QRect>
#include <QDebug>
#include <QPalette>
#include <QPainter>
#include <QPixmap>
#include <QShortcut>

#define BASE_WIDTH 600

void GammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma)

{

unsigned char lut[256];

for (int i = 0; i < 256; i++)

{

lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);

}

dst = src.clone();

const int channels = dst.channels();

switch (channels)

{

case 1:

{

cv::MatIterator_<uchar> it, end;

for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)

*it = lut[(*it)];

break;

}

case 3:

{

cv::MatIterator_<cv::Vec3b> it, end;

for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)

{

(*it)[0] = lut[((*it)[0])];

(*it)[1] = lut[((*it)[1])];

(*it)[2] = lut[((*it)[2])];

}

break;

}

}

}

ImageView::ImageView(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ImageView)
{
    ui->setupUi(this);


    gc=0;
    gamma = 2.2;
    zoomInAct = ui->zoomInAct;
    zoomOutAct = ui->zoomOutAct;
   // ui->zoomInAct->setShortcut(QKeySequence::ZoomIn);
    //ui->zoomOutAct->setShortcut(QKeySequence::ZoomOut);
    ui->zoomOutAct->setEnabled(true);
    ui->zoomInAct->setEnabled(true);
    loadAct = ui->loadAct;
    normalSizeAct = ui->normalSizeAct;
    //   QImage image("/home/giach/Data/Repos/app/test.jpg");
    //   ui->imageLabel->setPixmap(QPixmap::fromImage(image));

    imageLabel = new QLabel;
    imageLabel->setBackgroundRole(QPalette::Base);
    imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageLabel->setScaledContents(true);

    scrollArea = new QScrollArea;
    scrollArea->setBackgroundRole(QPalette::Dark);
    scrollArea->setWidget(imageLabel);
    setCentralWidget(scrollArea);

    setWindowTitle(tr("Image Viewer"));
    baseFactor=1;
    scaleFactor=1;
    resize(400, 400);

    // rub1= new QRubberBand(QRubberBand::Rectangle, this);
    // rub1->setGeometry(QRect(10,10,100,100));
    force8bit=false;
    active=0;
    // rub1->show();
    cropArea= new QRubberBand(QRubberBand::Rectangle, imageLabel);

    sphere1= new QRubberBand(QRubberBand::Rectangle, imageLabel);
    sphere2= new QRubberBand(QRubberBand::Rectangle, imageLabel);
    sphere3= new QRubberBand(QRubberBand::Rectangle, imageLabel);
    sphere4= new QRubberBand(QRubberBand::Rectangle, imageLabel);
    sphere1->setGeometry(QRect(0,0,0,0));
    sphere2->setGeometry(QRect(0,0,0,0));
    sphere3->setGeometry(QRect(0,0,0,0));
    sphere4->setGeometry(QRect(0,0,0,0));

    for(int k=0; k<4; k++){
        origins[k]=QPoint(0,0);
        originw[k]=QPoint(0,0);
        radius[k]=0;
        cx[k]=0;
        cy[k]=0;
        ls[k] = new QLabel(imageLabel);
        ls[k]->setVisible(false);

    }

    white1= new QRubberBand(QRubberBand::Rectangle, imageLabel);
    white2= new QRubberBand(QRubberBand::Rectangle, imageLabel);
    white3= new QRubberBand(QRubberBand::Rectangle, imageLabel);
    white4= new QRubberBand(QRubberBand::Rectangle, imageLabel);
    white1->setGeometry(QRect(0,0,0,0));
    white2->setGeometry(QRect(0,0,0,0));
    white3->setGeometry(QRect(0,0,0,0));
    white4->setGeometry(QRect(0,0,0,0));

    pointsCounter=0;

    shortcut = new QShortcut(QKeySequence(QKeySequence::ZoomIn),parent);
    connect(shortcut, SIGNAL(activated()), zoomInAct, SLOT(triggered()));
    shortcut2 = new QShortcut(QKeySequence(QKeySequence::ZoomOut),parent);
    connect(shortcut2, SIGNAL(activated()), this, SLOT(zoomOut()));

    connect(zoomInAct, SIGNAL(triggered()), this, SLOT(zoomIn()));
    connect(zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOut()));
    connect(normalSizeAct, SIGNAL(triggered()), this, SLOT(normalSize()));

    connect(loadAct,SIGNAL(triggered()),this,SLOT(load()));

    connect(this,SIGNAL(sph1Sig()),(this->parent()),SLOT(toggleSph1()));
    connect(this,SIGNAL(sph2Sig()),(this->parent()),SLOT(toggleSph2()));
    connect(this,SIGNAL(sph3Sig()),(this->parent()),SLOT(toggleSph3()));
    connect(this,SIGNAL(sph4Sig()),(this->parent()),SLOT(toggleSph4()));
    connect(this,SIGNAL(w1Sig()),(this->parent()),SLOT(toggleW1()));
    connect(this,SIGNAL(w2Sig()),(this->parent()),SLOT(toggleW2()));
    connect(this,SIGNAL(w3Sig()),(this->parent()),SLOT(toggleW3()));
    connect(this,SIGNAL(w4Sig()),(this->parent()),SLOT(toggleW4()));
    connect(this,SIGNAL(cropSig()),(this->parent()),SLOT(areaCrop()));
}

ImageView::~ImageView()
{
    delete ui;
}

void ImageView::load()
{


    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open File"), QDir::currentPath());
    if (!fileName.isEmpty()) {
        QImage image(fileName);
        if (image.isNull()) {
            QMessageBox::information(this, tr("Image Viewer"),
                                     tr("Cannot load %1.").arg(fileName));
            return;
        }
        imageLabel->setPixmap(QPixmap::fromImage(image));
        s = image.size();
        ratio = (float)s.width()/(float)s.height();
        qDebug() << " ee " << ratio;
        qDebug() << (int)std::max(BASE_WIDTH,s.width());
        resize((int)std::min(BASE_WIDTH,s.width()), (int)std::min(BASE_WIDTH,s.width())/ratio);
        //rub1->setGeometry(QRect(10,10,100,100));

        // scaleImage(scaleFactor);
        //fitToWindowAct->setEnabled(true);
        updateActions();


        imageLabel->adjustSize();
    }

}


void ImageView::load( QString fileName)
{

    if (!fileName.isEmpty()) {
        cv::Mat image = cv::imread(fileName.toStdString(), CV_LOAD_IMAGE_COLOR);


        //QImage image(fileName);
        //if (image.isNull()) {
         if (   ! image.data ) {
            QMessageBox::information(this, tr("Image Viewer"),
                                     tr("Cannot load %1.").arg(fileName));
            return;
        }
        cv::cvtColor(image,image, cv::COLOR_BGR2RGB);

        cv::Mat imagev;
        if(!force8bit)
            imagev = cv::imread(fileName.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
        else
            imagev = cv::imread(fileName.toStdString(), CV_LOAD_IMAGE_COLOR);

        cv::cvtColor(imagev,imagev, cv::COLOR_BGR2RGB);

        double min, max;
         cv::minMaxLoc(imagev, &min, &max);
        qDebug() << "minmax " << min << max;
        qDebug() << (image.at<cv::Vec3b>(1,1))[0]<< (image.at<cv::Vec3b>(1,1))[1]<< (image.at<cv::Vec3b>(1,1))[2];

qDebug() << "gamma " << gamma ;
        if(gc)
        GammaCorrection(image, image, 1/gamma);

        qDebug() << "Minmax " << min << max;
        qDebug() << (image.at<cv::Vec3b>(1,1))[0]<< (image.at<cv::Vec3b>(1,1))[1]<< (image.at<cv::Vec3b>(1,1))[2];

        imageLabel->setPixmap(QPixmap::fromImage(QImage(image.data,image.cols,image.rows,image.step,QImage::Format_RGB888)));

        s = QSize(image.cols,image.rows);
        ratio = (float)s.width()/(float)s.height();
        int r,g,b,a;
        int cx= s.width() /2;
        int cy= s.height()/2;
        depth=imagev.depth();
        if(imagev.depth() == 0){
            qDebug() << imagev.depth() << "8 bit image";
            maxval = 255;
        }
        if(imagev.depth() == 2){
            qDebug() << imagev.depth() << "16 bit image";
            maxval = 65535;
        }
        qDebug() << (imagev.at<cv::Vec3b>(cy,cx))[0]<< (imagev.at<cv::Vec3b>(cy,cx))[1]<< (imagev.at<cv::Vec3b>(cy,cx))[2];
        qDebug() << (imagev.at<cv::Vec3s>(cy,cx))[0]<< (imagev.at<cv::Vec3s>(cy,cx))[1]<< (imagev.at<cv::Vec3s>(cy,cx))[2];






              originw[0] /= (scaleFactor/baseFactor);
              endw[0] /= (scaleFactor/baseFactor);
              white1->setGeometry(QRect(originw[0],endw[0]));

              originw[1] /= (scaleFactor/baseFactor);;
              endw[1] /= (scaleFactor/baseFactor);;
              white2->setGeometry(QRect(originw[1],endw[1]));

              originw[2] /= (scaleFactor/baseFactor);;
              endw[2] /= (scaleFactor/baseFactor);;
              white3->setGeometry(QRect(originw[2],endw[2]));

              originw[3] /= (scaleFactor/baseFactor);;
              endw[3] /= (scaleFactor/baseFactor);
              white4->setGeometry(QRect(originw[3],endw[3]));

              origins[0] /= (scaleFactor/baseFactor);
              ends[0] /= (scaleFactor/baseFactor);
              sphere1->setGeometry(QRect(origins[0],ends[0]));
              ls[0]->setGeometry(QRect(origins[0],ends[0]));

              origins[1] /= (scaleFactor/baseFactor);
              ends[1] /= (scaleFactor/baseFactor);
              sphere2->setGeometry(QRect(origins[1],ends[1]));
              ls[1]->setGeometry(QRect(origins[1],ends[1]));

              origins[2] /= (scaleFactor/baseFactor);
              ends[2] /= (scaleFactor/baseFactor);;
              sphere3->setGeometry(QRect(origins[2],ends[2]));
              ls[2]->setGeometry(QRect(origins[2],ends[2]));

              origins[3] /= (scaleFactor/baseFactor);
              ends[3] /= (scaleFactor/baseFactor);
              sphere4->setGeometry(QRect(origins[3],ends[3]));
              ls[3]->setGeometry(QRect(origins[3],ends[3]));

        // resize(1000, int(1000/ratio));
        baseFactor = BASE_WIDTH/(float)s.width();
        scaleFactor = baseFactor;
        zoomFactor=1.0;
        imageLabel->resize(scaleFactor * imageLabel->pixmap()->size());

        qDebug() << (int)std::min(BASE_WIDTH,s.width());
         resize(scaleFactor * imageLabel->pixmap()->size());
       //resize((int)std::min(BASE_WIDTH,s.width()), (int)std::min(BASE_WIDTH,s.width())/ratio);

        adjustScrollBar(scrollArea->horizontalScrollBar(), baseFactor);
        adjustScrollBar(scrollArea->verticalScrollBar(), baseFactor);

        this->repaint();
        qApp->processEvents();
  //      qDebug() << s.width();
//        qDebug() << (baseFactor);

    }

}




void ImageView::updateActions()
{
    /* ui->zoomInAct->setEnabled(!ui->fitToWindowAct->isChecked());
    ui->zoomOutAct->setEnabled(!ui->fitToWindowAct->isChecked());
    normalSizeAct->setEnabled(!ui->fitToWindowAct->isChecked());*/
}

void ImageView::scaleImage(double factor)
{
    Q_ASSERT(imageLabel->pixmap());
    scaleFactor *= factor;
    zoomFactor *= factor;
    imageLabel->resize(scaleFactor * imageLabel->pixmap()->size());
    int ax,ay,bx,by;


    originw[0] *= factor;
    endw[0] *= factor;
    white1->setGeometry(QRect(originw[0],endw[0]));

    originw[1] *= factor;
    endw[1] *= factor;
    white2->setGeometry(QRect(originw[1],endw[1]));

    originw[2] *= factor;
    endw[2] *= factor;
    white3->setGeometry(QRect(originw[2],endw[2]));

    originw[3] *= factor;
    endw[3] *= factor;
    white4->setGeometry(QRect(originw[3],endw[3]));

    origins[0] *= factor;
    ends[0] *= factor;
    sphere1->setGeometry(QRect(origins[0],ends[0]));

    ls[0]->setGeometry(QRect(origins[0],ends[0]));


    origins[1] *= factor;
    ends[1] *= factor;
    sphere2->setGeometry(QRect(origins[1],ends[1]));
    ls[1]->setGeometry(QRect(origins[1],ends[1]));

    origins[2] *= factor;
    ends[2] *= factor;
    sphere3->setGeometry(QRect(origins[2],ends[2]));
    ls[2]->setGeometry(QRect(origins[2],ends[2]));

    origins[3] *= factor;
    ends[3] *= factor;
    sphere4->setGeometry(QRect(origins[3],ends[3]));
    ls[3]->setGeometry(QRect(origins[3],ends[3]));

    originc *= factor;
    endc *= factor;
    cropArea->setGeometry(QRect(originc,endc));

    adjustScrollBar(scrollArea->horizontalScrollBar(), factor);
    adjustScrollBar(scrollArea->verticalScrollBar(), factor);

 //   zoomInAct->setEnabled(scaleFactor < 3.0);
  //  zoomOutAct->setEnabled(scaleFactor > 0.333);

}

void ImageView::adjustScrollBar(QScrollBar *scrollBar, double factor)
{
    scrollBar->setValue(int(factor * scrollBar->value()
                            + ((factor - 1) * scrollBar->pageStep()/2)));
}


void ImageView::zoomIn()
{
    scaleImage(1.25);

qDebug() << "zoom";
}

void ImageView::zoomOut()
{
    scaleImage(0.8);
}

void ImageView::normalSize()
{
 scaleImage(1/zoomFactor);
//    imageLabel->adjustSize();
    //scaleFactor = baseFactor;
}

//bool ImageView::eventFilter


void ImageView::mousePressEvent(QMouseEvent *event)
{

    if(active==11){
        cpoints[pointsCounter].setX(imageLabel->mapFromParent(event->pos()).x()/scaleFactor);
        cpoints[pointsCounter].setY(imageLabel->mapFromParent(event->pos()).y()/scaleFactor);
        qDebug() << imageLabel->mapFromParent(event->pos());

        qDebug() << pointsCounter << " " << cpoints[pointsCounter].x() << " "  << cpoints[pointsCounter].y() ;


        QImage tmp(imageLabel->pixmap()->toImage());
                  QPainter painter(&tmp);
                  QPen paintpen(Qt::red);
                  paintpen.setWidth(10);

                  painter.setPen(paintpen);
                  painter.drawPoint(cpoints[pointsCounter]);
                  imageLabel->setPixmap(QPixmap::fromImage(tmp));

//      QPixmap pixmap(11,11);
//        pixmap.fill(QColor("red"));

//        QPainter painter(&pixmap);
//        painter.setPen(QPen(Qt::green));
//        painter.save();
//        QTransform trans;
//        // Move to the center of the widget
//        //trans.translate(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor);
//        trans.translate(event->pos().x(),event->pos().y());

//        // Move to the center of the image
//        painter.setTransform(trans);
//        painter.drawEllipse(QPointF(0,0),5,5);
//        painter.restore();


        pointsCounter = pointsCounter+1;
    }
    
    if(active==10){
        originc = imageLabel->mapFromParent(event->pos());

        cropArea->setGeometry(QRect(originc, QSize()));
        cropArea->show();
    }

    if(active==1){
        originw[0] = imageLabel->mapFromParent(event->pos());

        white1->setGeometry(QRect(originw[0], QSize()));
        white1->show();
    }
    else if (active==2){

        originw[1] = imageLabel->mapFromParent(event->pos());

        white2->setGeometry(QRect(originw[1], QSize()));
        white2->show();
    }
    else if (active==3){
        originw[2] = imageLabel->mapFromParent(event->pos());
        white3->setGeometry(QRect(originw[2], QSize()));
        white3->show();
    }
    else if (active==4){

        originw[3] = imageLabel->mapFromParent(event->pos());
        white4->setGeometry(QRect(originw[3], QSize()));
        white4->show();
    }


    if(active==5){
        origins[0] = imageLabel->mapFromParent(event->pos());
        qDebug() << origins[0].x() << " " << origins[0].y() << endl;

        sphere1->setGeometry(QRect(origins[0], QSize()));
        sphere1->show();
    }
    if(active==6){
        origins[1] = imageLabel->mapFromParent(event->pos());

        sphere2->setGeometry(QRect(origins[1], QSize()));
        sphere2->show();
    }
    if(active==7){
        origins[2] = imageLabel->mapFromParent(event->pos());

        sphere3->setGeometry(QRect(origins[2], QSize()));
        sphere3->show();
    }
    if(active==8){
        origins[3] = imageLabel->mapFromParent(event->pos());

        sphere4->setGeometry(QRect(origins[3], QSize()));
        sphere4->show();
    }


}

void ImageView::mouseMoveEvent(QMouseEvent *event)
{

    if(active==1)
        white1->setGeometry(QRect(originw[0], imageLabel->mapFromParent(event->pos()) ).normalized());
    else if (active==2)
        white2->setGeometry(QRect(originw[1], imageLabel->mapFromParent(event->pos()) ).normalized());
    else if (active==3)
        white3->setGeometry(QRect(originw[2], imageLabel->mapFromParent(event->pos()) ).normalized());
    else if (active==4)
        white4->setGeometry(QRect(originw[3], imageLabel->mapFromParent(event->pos()) ).normalized());
    else if(active==5)
        sphere1->setGeometry(QRect(origins[0], imageLabel->mapFromParent(event->pos()) ).normalized());
    else if (active==6)
        sphere2->setGeometry(QRect(origins[1], imageLabel->mapFromParent(event->pos()) ).normalized());
    else if (active==7)
        sphere3->setGeometry(QRect(origins[2], imageLabel->mapFromParent(event->pos()) ).normalized());
    else if (active==8)
        sphere4->setGeometry(QRect(origins[3], imageLabel->mapFromParent(event->pos()) ).normalized());
    else if (active==10)
        cropArea->setGeometry(QRect(originc, imageLabel->mapFromParent(event->pos()) ).normalized());

}

void ImageView::mouseReleaseEvent(QMouseEvent *event)
{

    if (active==10){
        endc= imageLabel->mapFromParent(event->pos());
        if(endc.x() > originc.x() && endc.y() > originc.y()){
            cropArea->setGeometry(QRect(originc,endc));

            active=0;
            qDebug() << scaleFactor << "!!!";
            emit cropSig();
        }
    }

    if (active==1){
        endw[0] = imageLabel->mapFromParent(event->pos());
        if(endw[0].x() > originw[0].x() && endw[0].y() > originw[0].y()){
            white1->setGeometry(QRect(originw[0],endw[0]));
            active=0;
           emit w1Sig();
        }
    }
    else if (active==2)  {
        endw[1] = imageLabel->mapFromParent(event->pos());
        if(endw[1].x() > originw[1].x() && endw[1].y() > originw[1].y()){
            white2->setGeometry(QRect(originw[1],endw[1]));
            active=0;
            emit w2Sig();
        }
    }
    else if (active==3){
        endw[2] = imageLabel->mapFromParent(event->pos());
        if(endw[2].x() > originw[2].x() && endw[2].y() > originw[2].y()){
            white3->setGeometry(QRect(originw[2],endw[2]));
            active=0;
             emit w3Sig();
        }
    }
    else if (active==4){
        endw[3] = imageLabel->mapFromParent(event->pos());
        if(endw[3].x() > originw[3].x() && endw[3].y() > originw[3].y()){
            white4->setGeometry(QRect(originw[3],endw[3]));
            active=0;
             emit w4Sig();
        }
    }
    else     if (active==5){
        ends[0] = imageLabel->mapFromParent(event->pos());
        if(ends[0].x() > origins[0].x() && ends[0].y() > origins[0].y()){
            sphere1->setGeometry(QRect(origins[0],ends[0]));
            active=0;
            emit sph1Sig();
        }
    }
    else if (active==6)  {
        ends[1] = imageLabel->mapFromParent(event->pos());
        if(ends[1].x() > origins[1].x() && ends[1].y() > origins[1].y()){
            sphere2->setGeometry(QRect(origins[1],ends[1]));
            active=0;
            emit sph2Sig();
        }
    }
    else if (active==7){
        ends[2] = imageLabel->mapFromParent(event->pos());
        if(ends[2].x() > origins[2].x() && ends[2].y() > origins[2].y()){
            sphere3->setGeometry(QRect(origins[2],ends[2]));
            active=0;
            emit sph3Sig();
        }
    }
    else if (active==8){
        ends[3] = imageLabel->mapFromParent(event->pos());
        if(ends[3].x() > origins[3].x() && ends[3].y() > origins[3].y()){
            sphere4->setGeometry(QRect(origins[3],ends[3]));
            active=0;
            emit sph4Sig();
        }


    }


}

void ImageView::on_frame1Act_triggered()
{
    active=1;

}

void ImageView::on_frame2Act_triggered()
{
    active=2;
}

void ImageView::on_frame3Act_triggered()
{
    active=3;
}

void ImageView::on_frame4Act_triggered()
{
    active=4;
}

void ImageView::on_actionOff_triggered()
{
    active=0;
}

void ImageView::on_sphere1Act_triggered()
{
    active=5;
}

void ImageView::on_sphere2Act_triggered()
{
    active=6;
}

void ImageView::on_sphere3Act_triggered()
{
    active=7;
}

void ImageView::on_sphere4Act_triggered()
{
    active=8;
}




void ImageView::on_actionClear_triggered()
{

    sphere1->setGeometry(QRect(0,0,0,0));
    sphere2->setGeometry(QRect(0,0,0,0));
    sphere3->setGeometry(QRect(0,0,0,0));
    sphere4->setGeometry(QRect(0,0,0,0));
}
