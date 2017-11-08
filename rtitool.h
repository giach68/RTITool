#ifndef RTITOOL_H
#define RTITOOL_H

#include <QMainWindow>
#include <QListWidgetItem>
#include "imageview.h"
#include <QString>

namespace Ui {
class RTITool;
}

class RTITool : public QMainWindow
{
    Q_OBJECT

public:
    explicit RTITool(QWidget *parent = 0);
    ~RTITool();

    ImageView* iw;
    QString projectFolder;

    cv::Mat maxi;
    cv::Mat mini;
    cv::Mat chrome;
    double grayref;
    double lzmax;

    QStringList  bList;

private slots:

  //  void processAPA(QString filename);

    void on_actionImage_list_triggered();

    void on_listWidget_itemDoubleClicked(QListWidgetItem *item);

    void on_actionEstimate_triggered();

    void toggleSph1();
    void toggleSph2();
    void toggleSph3();
    void toggleSph4();
    void toggleW1();
    void toggleW2();
    void toggleW3();
    void toggleW4();

    void on_sph1but_clicked();

    void on_sph2but_clicked();

    void on_sph3but_clicked();

    void on_sph4but_clicked();

    void on_rems1_clicked();

    void on_rems2_clicked();

    void on_rems3_clicked();

    void on_rems4_clicked();

    void on_lightEst_clicked();

    void saveLp(QString fileName);

    void on_saveLp_clicked();

    void on_cx1spin_valueChanged(int arg1);

    void on_cy1spin_valueChanged(int arg1);

    void on_r1spin_valueChanged(int arg1);

    void on_cx2spin_valueChanged(int arg1);

    void on_cy2spin_valueChanged(int arg1);

    void on_r2spin_valueChanged(int arg1);

    void on_cx3spin_valueChanged(int arg1);

    void on_cy3spin_valueChanged(int arg1);

    void on_r3spin_valueChanged(int arg1);

    void on_cx4spin_valueChanged(int arg1);

    void on_cy4spin_valueChanged(int arg1);

    void on_r4spin_valueChanged(int arg1);

    void on_remw1_clicked();

    void on_w1but_clicked();

    void on_w2but_clicked();

    void on_w3but_clicked();

    void on_w4but_clicked();

    void on_whiteEst_clicked();

    void on_saveCorrImages_clicked();

    void on_interpDir_clicked();

    void on_pushButton_clicked();

    void on_saveAPA_clicked();

    void on_remw2_clicked();

    void on_remw3_clicked();

    void on_remw4_clicked();

    void on_pushButton_2_clicked();

    void on_action_lp_file_triggered();

    void on_cropBut_clicked();

    void on_remCrop_clicked();

    void on_box8Bit_clicked(bool checked);

    void on_box8Bit_stateChanged(int arg1);

    void on_box8Bit_toggled(bool checked);

   // void on_pushButton_3_clicked();

    void on_loadListButton_clicked();

    void loadList(QString fileName);

    void loadLp(QString fileName);

    void loadDirFromLp(QString fileName);

    void on_loadLpButton_clicked();

    void loadCalib(QString fileName);

    void on_loadCalibButton_clicked();




    void on_angle1spin_valueChanged(double arg1);

    void on_r2_2Spin_valueChanged(int arg1);

    void on_r2_1Spin_valueChanged(int arg1);

    void on_angle2spin_valueChanged(double arg1);


    void on_angle3spin_valueChanged(double arg1);


    void on_r2_3Spin_valueChanged(int arg1);

    void on_r2_4Spin_valueChanged(int arg1);

    void on_angle4spin_valueChanged(double arg1);

    void clearParams();

  //  void on_apaButton_clicked();

    void on_undistortImagesButton_clicked();
    void on_removeAmbientButton_clicked();


    void on_prevButton_clicked();

    void on_nextButton_clicked();

    void saveId(QString fileName);
    void on_saveIdButton_clicked();

    void on_loadIdButton_clicked();

    void on_loadCalimButton_clicked();

    void on_corrBackimgBut_clicked();

    void on_createProjectButton_clicked();

    void on_openProjectButton_clicked();

    void on_findImagesButton_clicked();

    void on_copyFolderButton_clicked();

    QString getFilename();


    QString getFoldername();

    void loadCorrim(QString fileName);

    void loadCorrData(QString fileName);

    void on_importCalimButton_clicked();

    void importCalim(QString sourceFolder);

    void findCalim();

    void on_lpDirButton_clicked();

    //void on_pushButton_3_clicked();

    void on_undistortCalibBut_clicked();

private:
    Ui::RTITool *ui;
};

#endif // RTITool_H
