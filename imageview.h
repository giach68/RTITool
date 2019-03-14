#ifndef IMAGEVIEW_H
#define IMAGEVIEW_H

#include <QMainWindow>
#include <QLabel>
#include <QScrollArea>
#include <QAction>
#include <QScrollBar>
#include <QRubberBand>
#include <QPoint>
#include <QMouseEvent>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace Ui {
class ImageView;
}

class ImageView : public QMainWindow
{
    Q_OBJECT

public:
    explicit ImageView(QWidget *parent = 0);
    ~ImageView();
    void load(QString name);

    bool gc;
    double gamma;
    QSize s;
    QRubberBand *sphere1;
    QRubberBand *sphere2;
    QRubberBand *sphere3;
    QRubberBand *sphere4;
    QRubberBand *white1;
    QRubberBand *white2;
    QRubberBand *white3;
    QRubberBand *white4;

    QRubberBand *cropArea;
    QLabel* ls[4];
    QPixmap* pixs[4];
    QPainter* paints[4];

    int pointsCounter;

    int maxval;
    int depth;

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    double cx[4];
    double cy[4];
    double radius[4];
    double rx[4];
    double ry[4];
    double angle[4];
    cv::RotatedRect boxE[4];

    int active;
    bool force8bit;

    QPoint originw[4];
    QPoint endw[4];
    QPoint origins[4];
    QPoint ends[4];
    cv::RotatedRect box[4];

    QPoint originc;
    
    QPoint cpoints[20];
    QPoint endc;

    double baseFactor;
    double scaleFactor;
    double zoomFactor;
    double ratio;

    std::vector<double*> lights[4];
    std::vector<double> coeffs[10];
    std::vector<double> coilix[3];
    std::vector<double> coiliy[3];
    std::vector<double> coiliz[3];
    QLabel *imageLabel;
    QScrollArea *scrollArea;

private slots:
    //void open();
    void zoomIn();
    void zoomOut();
    //    void fitToWindow();
    //    void about();
    void normalSize();
    void load();



    void on_frame1Act_triggered();

    void on_frame2Act_triggered();

    void on_frame3Act_triggered();

    void on_frame4Act_triggered();

    void on_actionOff_triggered();

    void on_sphere1Act_triggered();

    void on_sphere2Act_triggered();

    void on_sphere3Act_triggered();

    void on_sphere4Act_triggered();

    void on_actionClear_triggered();

private:
    Ui::ImageView *ui;
    QAction *normalSizeAct;
    //   QAction *aboutAct;
    QAction *loadAct;
    QAction *zoomInAct;
    QAction *zoomOutAct;

    QRubberBand *rub1;




    void updateActions();
    void scaleImage(double factor);
    void resetImage(double factor);
    void adjustScrollBar(QScrollBar *scrollBar, double factor);

    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);



signals:
    void sph1Sig();
    void sph2Sig();
    void sph3Sig();
    void sph4Sig();
    void w1Sig();
    void w2Sig();
    void w3Sig();
    void w4Sig();
    void cropSig();
};

#endif // IMAGEVIEW_H
