#include "rtitool.h"
#include "ui_rtitool.h"
#include <iostream>
#include <fstream>
#include <QFileDialog>
#include <QString>
#include <QStringList>
#include <QTextStream>
#include <QInputDialog>
#include <QDebug>
#include <QPixmap>
#include <QPainter>
#include <QFile>
#include <QTime>
#include <QProgressDialog>
#include <QMessageBox>
#include <QtCore/qtextstream.h>
#include "s_hull_pro.h"


using namespace std;
using namespace cv;




bool pointSortPredicate(const Shx& a, const Shx& b)
{
    if (a.r < b.r)
        return true;
    else if (a.r > b.r)
        return false;
    else if (a.c < b.c)
        return true;
    else
        return false;
};

bool pointComparisonPredicate(const Shx& a, const Shx& b)
{
    return a.r == b.r && a.c == b.c;
}

template<class T>struct compare_index
{
    const T base_arr;
    compare_index (const T arr) : base_arr (arr) {}

    bool operator () (int a, int b) const
    {
        return (base_arr[a] < base_arr[b]);
    }
};


void savePTM_LRGB(QString filename, int W, int H, QString chroma_img){
    //save the result in the LRGB format

    double max[6], min[6];
    int bias[6];
    char* files[6] = {"file.c1","file.c2","file.c3","file.c4","file.c5","file.c6"};
    float scale[6];
    float* pBuff;
    for (int i = 0; i <= 5; i++){
        ifstream coef (files[i], ios::in | ios::binary);
        pBuff = new float[W*H];
        coef.read((char*)pBuff,W*H*sizeof(float));
        min[i] = 99999999999;
        max[i] = -99999999999;

        for (int x= 0; x < W*H; x++)
        {
            if(pBuff[x] > max[i]) max[i]=pBuff[x];
            if(pBuff[x] < min[i]) min[i]=pBuff[x];
        }

        scale[i]=(float) 1.0+floor((max[i]-min[i]-1)/256);
        bias[i]=(int)(-min[i]/scale[i]);// you can change this value
        delete(pBuff);
        qDebug() <<"minmax "<<min[i]<< ' ' << max[i] <<'\n';
        qDebug() <<"scale "<<int(scale[i])<<'\n';
        qDebug() <<"bias "<<int(bias[i])<<'\n';
        coef.close();
    }

    unsigned char* scaledc = new unsigned char[W*H*6];

    for (int i = 0; i <= 5; i++){
        ifstream coef (files[i], ios::in | ios::binary);
        pBuff = new float[W*H];
        coef.read((char*)pBuff,W*H*sizeof(float));
        coef.close();

        for (int x = 0;x<W; x++)
            for (int y = 0; y <H; y++)
            {

                scaledc[x+y*W+W*H*i] = (unsigned char)((pBuff[y*W+x]/scale[i])+(float)bias[i]);
            }
        delete(pBuff);
    }



    ofstream outfile;
    outfile.open("test.ptm",ios::binary);
    outfile <<  "PTM_1.2\n";
    outfile <<  "PTM_FORMAT_LRGB\n";
    outfile << W <<"\n";
    outfile << H <<"\n";
    QString num;
    for (int i = 0; i < 6; i++){
        num=QString::number((float)scale[i]);
        outfile << num.toStdString()<<" ";

    } outfile <<'\n';
    for (int i = 0; i < 6; i++){
        num=QString::number((int)bias[i]);
        outfile << num.toStdString()<<' ';
    }outfile <<'\n';

    int offset;
    unsigned char c;

    cv::Mat image;
    image = cv::imread(chroma_img.toStdString(), CV_LOAD_IMAGE_COLOR);
    // image = cv::imread("/home/giach/Data/Repos/scan4reco/playground/rtitool2/CROPPEDRoughCorrSing/cropped0.tif", CV_LOAD_IMAGE_COLOR);


    for (int y = H-1; y >=0; y--)
        for (int x = 0;x<W; x++)
            for (int i = 0; i < 6; i++)
            {
                c=scaledc[x+y*W+W*H*i] ;//scaledc[i].at<unsigned char>(x,y);
                outfile.write(( char *)&c,1);

            }

    for (int y = H-1; y >=0; y--)
        for (int x = 0;x<W; x++){
            Vec3b val=image.at<Vec3b>(y,x);

            for (int i = 0; i < 3; i++)
                outfile.write(( char *)&val[i],1);
        }


    delete(scaledc);

    outfile.close();


}



string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}


cv::Mat processImageGrabCut(cv::Mat image ){
    cv::Mat bimage;
    //cv::Mat mask = imread("mask.png");
    // cv::threshold(mask, mask, 100, 150, cv::THRESH_BINARY_INV);
    //mask.convertTo(mask,CV_8UC1);
    image.convertTo(bimage, CV_8UC3);
    // define bounding rectangle
    //cv::Rect rectangle(40,90,bimage.cols-80,bimage.rows-170);
    cv::Rect rectangle(10,10,bimage.cols-10,bimage.rows-10);

    cv::Mat result; // segmentation result (4 possible values)
    cv::Mat bgModel,fgModel; // the models (internally used)


    //contours = mask.clone();
    //GrabCut segmentation
    cv::grabCut(bimage,    // input image
                result,   // segmentation result
                rectangle,// rectangle containing foreground
                bgModel,fgModel, // models
                1,        // number of iterations
                cv::GC_INIT_WITH_RECT); // use rectangle

    //cv::Mat cimage = Mat::zeros(image.size(), CV_8UC3);
    //cv::grabCut(contours,mask, rectangle, bgModel, fgModel,1,cv::GC_INIT_WITH_RECT);

    cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
    // Generate output image
    cv::Mat foreground(bimage.size(),CV_8UC3,cv::Scalar(255,255,255));
    cv::Mat background(bimage.size(),CV_8UC3,cv::Scalar(255,255,255));
    bimage.copyTo(foreground,result);

    return foreground;
}



cv::RotatedRect segmentAndFitEllipse(cv::Mat& image){

    cv::Mat cloneim=image.clone();
    cv::Mat gray, segmentedOtsu;
    cv::cvtColor(cloneim,gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, segmentedOtsu, 100, 150,cv::THRESH_OTSU);
    cv::threshold(segmentedOtsu, segmentedOtsu, 0, 255,cv::THRESH_BINARY_INV);
    imshow("segmentedOtsu",segmentedOtsu);
    waitKey();
    cv::Mat sel = cv::getStructuringElement(cv::MorphShapes (MORPH_ELLIPSE),cv::Size(30,30));
    cv::morphologyEx(segmentedOtsu, segmentedOtsu, cv::MorphShapes (MORPH_CLOSE),sel);
    cv::imwrite("mask.png",segmentedOtsu);

    vector<vector<Point> > contours;
    findContours(segmentedOtsu, contours, RETR_LIST, CHAIN_APPROX_NONE);
    cv::RotatedRect box;
    cv::Mat cimage = Mat::zeros(segmentedOtsu.size(), CV_8UC3);

    for(std::size_t i = 0; i < contours.size(); i++){
        size_t count = contours[i].size();
        cout<<"\nCount "<<count<<endl;
        cv::Mat pointsf;
        Mat(contours[i]).convertTo(pointsf, CV_32F);
        box = cv::fitEllipse(pointsf);
        if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 )
            continue;
        cv::drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);
        cv::ellipse(cimage, box, Scalar(0,0,255), 1, LINE_AA);
        cv::ellipse(cimage, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1, LINE_AA);
        cout<<"\nEllipse features-Center: "<<box.center<<" Angle "<<box.angle<<" Width "<<box.size.width<<" Height "<<box.size.height<<endl;
        Point2f vtx[4];
        box.points(vtx);
        for( int j = 0; j < 4; j++ )
            line(cimage, vtx[j], vtx[(j+1)%4], Scalar(0,255,0), 1, LINE_AA);
    }

    imshow("Fit Ellipse", cimage);
    waitKey();
    return box;
}

cv::RotatedRect segmentAndFitEllipseMax(cv::Mat& image){

    cv::Mat cloneim=image.clone();
    cv::Mat gray, segmentedOtsu, segmentedGC;
    cv::cvtColor(cloneim,gray, cv::COLOR_BGR2GRAY);
    cv::Mat fgd, bgd;

    cv::threshold(gray, segmentedOtsu, 0, 255,CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
    //  segmentedGC = segmentedOtsu*cv::GC_PR_BGD;
    // imshow("segmentedOtsu",gray);
    // waitKey();
    /* cv::Rect rectangle(50,50,image.cols-100,image.rows-100);
    cv::grabCut(cloneim,    // input image
                segmentedGC,   // segmentation result
                rectangle,// rectangle containing foreground
               bgd,fgd, // models
                1,       // number of iterations
                 cv::GC_INIT_WITH_RECT); // use mask
  //  cv::threshold(segmentedOtsu, segmentedOtsu, 0, 255,cv::THRESH_BINARY_INV);
      cv::compare(segmentedGC,cv::GC_PR_FGD,segmentedGC,cv::CMP_EQ);
      cv::Mat foreground(cloneim.size(),CV_8UC3,cv::Scalar(255,255,255));
      cloneim.copyTo(foreground,segmentedGC); // bg pixels not copied
*/
    //imshow("segmentedGC",foreground);
    //waitKey();
    /* cv::Mat sel = cv::getStructuringElement(cv::MorphShapes (MORPH_ELLIPSE),cv::Size(20,20));
    cv::morphologyEx(segmentedOtsu, segmentedOtsu, cv::MorphShapes (MORPH_OPEN),sel);
    sel = cv::getStructuringElement(cv::MorphShapes (MORPH_ELLIPSE),cv::Size(40,40));
    cv::morphologyEx(segmentedOtsu, segmentedOtsu, cv::MorphShapes (MORPH_CLOSE),sel);*/
    //    cv::imwrite("mask.png",segmentedOtsu);
    //  imshow("segmentedOtsu",segmentedOtsu);
    //  waitKey();

    vector<vector<Point> > contours;
    findContours(segmentedOtsu, contours, RETR_LIST, CHAIN_APPROX_NONE);
    //cout<<"\nContours Size"<<contours.size()<<endl;

    std::vector<unsigned int> contoursDim(contours.size());
    for(size_t i = 0; i < contours.size(); i++){
        unsigned int count = contours[i].size();
        contoursDim[i] = count;
        //  cout<<"\nContours Dim"<<contoursDim[i]<<endl;
    }

    Mat cimage = Mat::zeros(segmentedOtsu.size(), CV_8UC3);
    RotatedRect box;
    std::vector<unsigned int>::iterator result = std::max_element(contoursDim.begin(), contoursDim.end());
    unsigned int maxIndex = std::distance(contoursDim.begin(),result);
    //cout<<"\nIndex of maximum contour: "<<maxIndex<<endl;
    Mat pointsf;
    Mat(contours[maxIndex]).convertTo(pointsf, CV_32F);
    box = cv::fitEllipse(pointsf);
    if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 ){
        qDebug() <<"\nFitting went wrong!";
    }
    else
    {
        cv::drawContours(cimage, contours, (int)maxIndex, Scalar::all(255), 1, 8);
        cv::ellipse(cimage, box, Scalar(0,0,255), 1, LINE_AA);
        cv::ellipse(cimage, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1, LINE_AA);
        qDebug() <<"\nEllipse features-Center: "<<box.center.x <<" Angle "<<box.angle<<" Width "<<box.size.width<<" Height "<<box.size.height<<endl;
        Point2f vtx[4];
        box.points(vtx);
        for( int j = 0; j < 4; j++ )
            line(cimage, vtx[j], vtx[(j+1)%4], Scalar(0,255,0), 1, LINE_AA);
    }
    // imshow("Fit Max Ellipse", cimage);
    // waitKey();
    return box;
}




cv::Point2f getHighlightPosition(cv::Mat& image, cv::RotatedRect& boxEllipse){

    Point2f highlightCoordinates = Point2f(0,0);

    Mat gray;
    cv::cvtColor(image,gray, cv::COLOR_BGR2GRAY);



    Mat mask = Mat::zeros(image.size(), CV_8UC1);

    cv::ellipse(mask, boxEllipse, 255, -1, LINE_AA);

    Mat area;
    gray.copyTo(area,mask);



    Point centroide;
    Mat dst, copia;
    Scalar color(255,39,0);
    Scalar color2(0,0,255);

    vector< vector<Point> > contorni;

    double la= 0.0;
    int li= -1;

    /* Method: starting from 255 (gray level and 8 bit image) and decreasing we threshold at this value the image and get the regions extracted
     * stopping when a bright region is found inside the circle. If more regions are found at the threshold, the largest is preferred to locate the highlight
     * the highlight is finally placed estimating the elliptic approximation of the region and taking the center.
     *
     * The light direction is computed then assuming ortographic view like done in the literature
     *
     * However, to change it we must do differently. This is not obvious as we don't have the information in the cropped region. We need to know the real pixels
     * coordinates. The best solution may be to return the highlight position in the ellipse then process outside
     *
     * */


    // loop over gray level

    // for (int valore = 255; (la>0 == 0) && valore>5; valore=valore-1){
    for (int valore = 255; (contorni.size() == 0) && valore>5; valore=valore-3){
        threshold( area , dst, valore, 255,0);
        findContours(dst, contorni, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0));

        la = 0; li= -1;
        for( int i = 0; i< contorni.size(); i++ ) { // if there is a region
            //  Find the area of contour
            double a=contourArea(contorni[i],false);
            qDebug() << "--" << contorni.size() << " " << valore << " "  << i << " " << a;
            if (a>la){
                copia = Mat::zeros( dst.size(), CV_8U );
                cv::drawContours(copia, contorni, i, 255, -1,8, noArray(), 0,Point(0,0));

                Moments momento = moments(copia, true);
                Point centroid_i;
                centroid_i.x = momento.m10/momento.m00;
                centroid_i.y = momento.m01/momento.m00;
                la=a;
                li=i;
                centroide = centroid_i;
            }

            if(li==-1){
                li=0;
                la=0;
                centroide.x = contorni[0][0].x;
                centroide.y = contorni[0][0].y;
            }

        }
        qDebug() << "val "<<  valore << " " << centroide.x << " " << centroide.y;

    }
    if (li<0) {
        std::cerr << "############ NO HIGHLIGHT!!!############" << std::endl;

        abort();
    } else {
        drawContours(image, contorni, li, color, -1,8, noArray(), 0,Point(0,0));

        circle(image ,centroide, 1, color2, -1, 8,0);
        /*   imshow("region",image);
        waitKey();
        */
    }
    return(centroide);
}

double* getLightDirection(cv::RotatedRect ellipse, cv::Point2d hlPos, cv::Mat CameraMatrix, double radiusSphere, QTextStream& stream){
    ////Method that estimates light direction based on the parameters of the ellipse, the coordinates of the light position,
    ////and the intrinsics of the camera. The rationale behind is https://www.overleaf.com/5539544ykcdsw#/17814043/.

    //write sphre position



    double ox, oy, alphax, alphay,sx, sy,f;

    alphax = CameraMatrix.at<double>(0,0); //11513.914929;
    alphay = CameraMatrix.at<double>(1,1);//11521.026366;

    f = alphax;//focal length in centimeters

    ox = CameraMatrix.at<double>(0,2);//3768.398316;
    oy = CameraMatrix.at<double>(1,2);//2257.433392;


    sx = 1;
    sy = alphay/alphax;


    double vecx = sx*(ellipse.center.x-ox);
    double vecy = sy*(ellipse.center.y-oy);
    double nor = sqrt(vecx*vecx+vecy*vecy);
    double dx=vecx/nor;
    double dy=vecy/nor;

    double majorAxis = ellipse.size.height;
    double minorAxis = ellipse.size.width;

    double px,py,qx,qy;
    ////the extreme of the major axis computed with the ellipse equation.
    px = ellipse.center.x - majorAxis*0.5f*dx;
    py = ellipse.center.y - majorAxis*0.5f*dy;

    qx = ellipse.center.x + majorAxis*0.5f*dx;
    qy = ellipse.center.y + majorAxis*0.5f*dy;


    cv::Point3d p((px-ox)*sx, (py-oy)*sy,f);
    cv::Point3d q((qx-ox)*sx, (qy-oy)*sy,f);
    cv::Point3d centerProjection(ox,oy,0);
    cv::Point3d o(ox,oy,f);
    cv::Point3d a,b,w,v;
    a = (p)/norm(p);
    b = (q)/norm(p);
    w = (a+b)/norm(a+b);

    //radius of the sphere in cm

    //radius of the sphere in pixels;
    //float radiusSphere = 151.181102;
    ////compute angles of triangle OCP;
    float angleOCP, anglePOC, OP, cameraDistance;
    anglePOC = acos((a.x*w.x +a.y*w.y +a.z*w.z));
    angleOCP = asin((a.x*w.x +a.y*w.y +a.z*w.z));
    OP = tan(angleOCP)*radiusSphere;
    cameraDistance = sqrt(pow(OP,2)+pow(radiusSphere,2));
    qDebug() <<M_PI<<"Test: "<<OP<<" camera distance "<<cameraDistance<<"\t AnglePOC\t"<<anglePOC*180/M_PI<<"\tAngleOCP\t"<<angleOCP*180/M_PI;

    //compute the coordinates of the sphere's center in camera coordinates.
    cv::Point3d centerSphere = w*cameraDistance;


    stream<<centerSphere.x<<" "<< centerSphere.y<<" " <<centerSphere.z<<"\n";



    //highlight coordinates in camera coordinates.
    cv::Point3d imageHighlight((hlPos.x -ox)*sx, (hlPos.y-oy)*sy,f);
    //view direction vector
    cv::Point3d viewDirection = (imageHighlight)/norm(imageHighlight);

    //solve quadratic sphere equation
    double sol1,sol2, discriminant, coefa, coefb, coefc;
    cv::Point3d sphereHighlight,sphereHighlight1, sphereHighlight2;
    coefa = viewDirection.x*viewDirection.x+viewDirection.y*viewDirection.y+viewDirection.z*viewDirection.z;
    coefb = -2*(viewDirection.x*centerSphere.x + viewDirection.y*centerSphere.y+viewDirection.z*centerSphere.z);
    coefc = centerSphere.x*centerSphere.x +centerSphere.y*centerSphere.y +centerSphere.z*centerSphere.z-radiusSphere*radiusSphere;
    discriminant = coefb*coefb -4*coefa*coefc;
    sol1 = (-coefb + sqrt(discriminant))/(2*coefa);
    sol2 = (-coefb - sqrt(discriminant))/(2*coefa);

    //choose the solution closest to the origin.
    sphereHighlight1 = sol1*viewDirection;
    sphereHighlight2 = sol2*viewDirection;
    if(abs(sphereHighlight1.z) < abs(sphereHighlight2.z))
        sphereHighlight = sphereHighlight1;
    else
        sphereHighlight = sphereHighlight2;

    //normal vector
    cv::Point3d normal = (sphereHighlight-centerSphere)/norm(sphereHighlight-centerSphere);


    //estimate light direction
    sphereHighlight = sphereHighlight/norm(sphereHighlight);
    double dotProductNormalHiglight = sphereHighlight.x*normal.x + sphereHighlight.y*normal.y + sphereHighlight.z*normal.z;
    qDebug() <<"Solution"<<dotProductNormalHiglight<<"\n";
    cv::Point3d lightDirection = sphereHighlight - 2*dotProductNormalHiglight*normal;
    lightDirection = lightDirection/norm(lightDirection);
    qDebug()  <<"\nLight direction vector:\t"<<lightDirection.x<<"\t"<<lightDirection.y<<"\t"<<lightDirection.z;
    double dotProductNormalIncident = lightDirection.x*normal.x + lightDirection.y*normal.y + lightDirection.z*normal.z;
    double ang = acos(dotProductNormalIncident);
    qDebug()  <<"\nHighlight angle\t"<<ang<<"\t"<<ang*180/M_PI;

    double* ld=new double[3];
    ld[0]=lightDirection.x;
    ld[1]=-lightDirection.y;
    ld[2]=-lightDirection.z;


    return ld;


}




/**
 * La funzione viene usata per permettere all'utente di modificare posizione del centro e raggio del cerchio trovato
 * con la funzione detectSphere.
 */
void resizeArea(int event, int x, int y, int flags, void* userdata){
    Mat* image = static_cast< Mat* > (userdata);
    Mat img = (*image).clone();
    Mat imgmod = img.clone();
    img.copyTo(imgmod);
    Scalar color(155,239,98);
    bool selezioneL, selezioneR;
    int posizione;
    Point centro;
    double raggio;

    if ( event == CV_EVENT_LBUTTONDOWN){
        selezioneL = true;
    }

    if ( event == CV_EVENT_RBUTTONDOWN){
        posizione = x;
        selezioneR = true;
    }

    if ( event == CV_EVENT_LBUTTONUP){
        selezioneL = false;
    }

    if ( event == CV_EVENT_RBUTTONUP){
        selezioneR = false;
    }

    if ( event == CV_EVENT_MOUSEMOVE && selezioneL){
        centro.x = x;
        centro.y = y;
        circle(imgmod, centro, raggio, color, 1, 8, 0);
        circle(imgmod, centro, 1, color, CV_FILLED, 8,0);
        imshow("ROI", imgmod);
    }

    if ( event == CV_EVENT_MOUSEMOVE && selezioneR){
        if(x >= posizione)
            raggio++;
        else
            raggio--;

        posizione = x;

        if(raggio >= 3)
            circle(imgmod, centro, raggio, color, 1, 8, 0);

        circle(imgmod, centro, 1, color, CV_FILLED, 8,0);
        imshow("ROI", imgmod);
    }
}



// This function takes the cropped ROI of the full-size image (OpenCV) and fit a circle over edges
// using the standard opencv hough function. It is not optimal and actually I wanted to do differently
// in any case now we will change

void fitCircle(Mat image, float* cx, float* cy, float* r){

    float dx[16], dy[16];
    for(int k=0;k<16;k++)
    {
        dx[k]=sin(k*M_PI/8);
        dy[k]=cos(k*M_PI/8);
    }

    int diffmax = 0;
    float raggio = *r;
    Point centro(*cx,*cy);
    float val = image.at<uchar>(centro);
    // int center_intensity = Center_intensity.val[0];		// Il primo valore è quello che a noi interessa
    float save_x=centro.x;
    float save_y=centro.y;
    float save_r= raggio;
    int nv[16];
    float funct, maxf=0;
    vector<Vec3f> circles;
    qDebug() << "inte " << val;

    Mat contours;
    //Canny(image,contours,50,200);
    //        cv::namedWindow("Canny");
    //  cv::imshow("Canny",contours);
    //cv::waitKey(0);

    //GaussianBlur( image, image, Size(3, 3), 2, 2 );
    HoughCircles( image, circles, CV_HOUGH_GRADIENT, 1, 10, 60, 20, *r/2, *r+2);
    for( size_t i = 0; i < circles.size(); i++ )
    {
        qDebug() << circles[i][2];

    }
    if(circles.size()>0){
        qDebug() << "Fit OK";
        *cx= (circles[0][0]);
        *cy= (circles[0][1]);
        *r= (circles[0][2]);

    }

    // display for debug
    //namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    //   imshow( "Hough Circle Transform Demo", image );
    //waitKey(0);

    // old trials to do a different circle detection
    /* for(float x=centro.x-10; x<centro.x+10; x++)
     for(float y=centro.y-10; y<centro.y+10; y++)
     for(float ro=raggio-10; ro< raggio+1; ro++){
     int nu=0;
     float val=0;
     for(int k=0;k<16;k++){
     if(x+dx[k]*(ro+1) >= 0 && x+dx[k]*(ro+1) < image.size().width)
     if(y+dy[k]*(ro+1) >= 0 && y+dy[k]*(ro+1) < image.size().height){
     val = val+ (ro-20)*max(10,image.at<uchar>(x+dx[k]*(float)(ro+1),y+dy[k]*(float)(ro+1))-image.at<uchar>(x+dx[k]*(float)(ro-1),y+dy[k]*(float)(ro-1)));
     nu++;
     }
     }

     if(val > maxf) {
     maxf =val;
     save_x=x;
     save_y=y;
     save_r=ro;

     qDebug() << "inte " << maxf << " " << save_x << " " << save_y << " " << save_r;
     }
     }


     //float raggio = sqrt( double(pow(double((centro.x- radius.x)),2.0) + pow(double((centro.y - radius.y)),2.0))); // Calcolo il raggio tramite il Teorema di Pitagora
     *cx=save_x;
     *cy=save_y;
     *r=save_r;
     */
    //  circle(image,Point(*cx,*cy),*r,Scalar(255,255,255),1,8,0);

    // namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    // imshow( "Hough Circle Transform Demo", image );

}

/**
 * This function is used both to estimate the highlight position and to estimate the light direction from the highlight and the sphere
 *
 */

double* Highlight(Mat region, int cx, int cy, int raggio){
    //qDebug() << "Hightlight...";

    Point centroide;
    Mat area, dst;
    Scalar color(255,39,0);
    Scalar color2(0,0,255);

    vector< vector<Point> > contorni;

    double Sx, Sy, phiL, tetaL, vectorX, vectorY, vectorZ;
    cvtColor( region, area, CV_RGB2GRAY );
    Mat copia;


    double la= 0.0;
    int li= -1;

    /* Method: starting from 255 (gray level and 8 bit image) and decreasing we threshold at this value the image and get the regions extracted
   * stopping when a bright region is found inside the circle. If more regions are found at the threshold, the largest is preferred to locate the highlight
   * the highlight is finally placed estimating the elliptic approximation of the region and taking the center.
   *
   * The light direction is computed then assuming ortographic view like done in the literature
   *
   * However, to change it we must do differently. This is not obvious as we don't have the information in the cropped region. We need to know the real pixels
   * coordinates. The best solution may be to return the highlight position in the ellipse then process outside
   *
   * */


    // loop over gray level

    qDebug() << "h search";
    for (int valore = 255; (contorni.size() == 0) && valore>5; valore=valore-5){
        threshold( area , dst, valore, 255,0);
        findContours(dst, contorni, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0));

        la = 0.0; li= -1;
        for( int i = 0; i< contorni.size(); i++ ) { // if there is a region

            //  Find the area of contour
            double a=contourArea(contorni[i],false);

            qDebug() << "reg found" << a;


            if (a>la){
                copia = Mat::zeros( dst.size(), CV_8U );
                cv::drawContours(copia, contorni, i, 255, -1,8, noArray(), 0,Point(0,0));

                Moments momento = moments(copia, true);
                Point centroid_i;
                centroid_i.x = momento.m10/momento.m00;
                centroid_i.y = momento.m01/momento.m00;

                centroid_i.x = cx -raggio + centroid_i.x;
                centroid_i.y = cy -raggio + centroid_i.y;

                double Sx_i = double((centroid_i.x - cx)) / raggio;
                double Sy_i = -1.0 *  double((centroid_i.y - cy)) / raggio;

                if (1.0 - pow(Sx_i,2.0) - pow(Sy_i,2.0) > 0.0) {
                    // Good candidate
                    la=a;
                    li=i;

                    centroide = centroid_i;
                    Sx = Sx_i;
                    Sy = Sy_i;
                }
            }
            if(li==-1){
                li=0;
                la=0;
                centroide.x = cx -raggio + contorni[0][0].x;
                centroide.y = cy -raggio + contorni[0][0].y;
                Sx = double((centroide.x - cx)) / raggio;
                Sy = -1.0 *  double((centroide.y - cy)) / raggio;
            }

        } // Look for centroid at current level

        if (li>=0) {
            // Found!
        } else {
            // Continue at next gray level
            contorni.clear();
        }
    }

    if (li<0) {
        std::cerr << "############ NO HIGHLIGHT!!!############" << std::endl;
        vectorX=0.0; vectorY=0.0; vectorZ=1.0;
        abort();
    } else {
        drawContours(region, contorni, li, color, -1,8, noArray(), 0,Point(0,0));
        Point lc;
        lc.x = centroide.x+raggio-cx;
        lc.y = centroide.y+raggio-cy;
        circle(region ,lc, 1, color2, -1, 8,0);
        //imshow("region",region);
        //waitKey();

        // Compute light vector
        phiL=0;
        if(1.0 - pow(Sx,2.0) - pow(Sy,2.0)>0)
            phiL = 2.0 *acos (double(sqrt(1.0 - pow(Sx,2.0) - pow(Sy,2.0))));
        //tetaL = atan (double(cy - centroide.y)/ double((cx - centroide.x)));

        tetaL = atan2 (double(cy - centroide.y),double((cx - centroide.x)));
        vectorX = (centroide.x > cx ? abs(sin(phiL) * cos(tetaL)) : -1*abs(sin(phiL) * cos(tetaL)));
        vectorY = (centroide.y < cy ? abs(sin(phiL) * sin(tetaL)) : -1*abs(sin(phiL) * sin(tetaL)));
        vectorZ = abs(cos(phiL));
    }

    double* lightVector = new double(3);
    double norm = sqrt(vectorX*vectorX+vectorY*vectorY+vectorZ*vectorZ);
    lightVector[0] = vectorX/norm;
    lightVector[1] = vectorY/norm;
    lightVector[2] = vectorZ/norm;

    //qDebug() << lightVector[0] << " " << lightVector[1] << " " << lightVector[2] << endl;
    return(lightVector);
}


RTITool::RTITool(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::RTITool)
{
    ui->setupUi(this);

    iw=new ImageView(this);
    ui->weightDir->setChecked(true);
    ui->autoFit->setChecked(true);
    ui->onMax->setChecked(true);
    ui->projBox->setChecked(true);
    ui->cx1spin->setVisible(false);
    ui->cx2spin->setVisible(false);
    ui->cx3spin->setVisible(false);
    ui->cx4spin->setVisible(false);
    ui->cy1spin->setVisible(false);
    ui->cy2spin->setVisible(false);
    ui->cy3spin->setVisible(false);
    ui->cy4spin->setVisible(false);
    ui->r1spin->setVisible(false);
    ui->r2spin->setVisible(false);
    ui->r3spin->setVisible(false);
    ui->r4spin->setVisible(false);
    ui->r2_1Spin->setVisible(false);
    ui->r2_2Spin->setVisible(false);
    ui->r2_3Spin->setVisible(false);
    ui->r2_4Spin->setVisible(false);
    ui->angle1spin->setVisible(false);
    ui->angle2spin->setVisible(false);
    ui->angle3spin->setVisible(false);
    ui->angle4spin->setVisible(false);

    ui->tabDir->setTabEnabled(2, false);
    ui->tabDir->setTabEnabled(3, false);
    ui->tabDir->setTabEnabled(4, false);
    ui->tabDir->setTabEnabled(1, false);

    ui->folderName->setReadOnly(true);
}

RTITool::~RTITool()
{
    delete ui;
}

void RTITool::clearParams(){

    for(int i=0;i<4;i++){
        iw->ls[i]->setGeometry(0,0,0,0);
        iw->ls[i]->setVisible(true);
        iw->cx[i]=0;
        iw->cy[i]=0;
        iw->rx[i]=0;
        iw->ry[i]=0;
        iw->angle[i]=0;
        iw->radius[i]=0;
        iw->origins[i]=QPoint(0,0);

    }
    iw->sphere1->setGeometry(QRect(0,0,0,0));
    ui->sph1->setText(QString("-"));
    ui->cx1spin->setVisible(false);
    ui->cy1spin->setVisible(false);
    ui->r1spin->setVisible(false);
    ui->r1spin->setVisible(false);
    ui->r2_1Spin->setVisible(false);
    ui->angle1spin->setVisible(false);
    iw->sphere2->setGeometry(QRect(0,0,0,0));
    ui->sph2->setText(QString("-"));
    ui->cx2spin->setVisible(false);
    ui->cy2spin->setVisible(false);
    ui->r2spin->setVisible(false);
    ui->r2spin->setVisible(false);
    ui->r2_2Spin->setVisible(false);
    ui->angle2spin->setVisible(false);
    iw->sphere3->setGeometry(QRect(0,0,0,0));
    ui->sph3->setText(QString("-"));
    ui->cx3spin->setVisible(false);
    ui->cy3spin->setVisible(false);
    ui->r3spin->setVisible(false);
    ui->r3spin->setVisible(false);
    ui->r2_3Spin->setVisible(false);
    ui->angle3spin->setVisible(false);
    iw->sphere4->setGeometry(QRect(0,0,0,0));
    ui->sph4->setText(QString("-"));
    ui->cx4spin->setVisible(false);
    ui->cy4spin->setVisible(false);
    ui->r4spin->setVisible(false);
    ui->r4spin->setVisible(false);
    ui->r2_4Spin->setVisible(false);
    ui->angle4spin->setVisible(false);
}

void RTITool::on_actionImage_list_triggered()
{
    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open image list"));

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    qDebug(fileName.toLatin1());
    QStringList  nList;

    QTextStream textStream(&file);
    while (true)
    {
        QString line = textStream.readLine();

        qDebug(line.toLatin1());
        if (line.isNull())
            break;
        else
            nList.append(line);

    }
    file.close();
    ui->listWidget->clear();
    ui->listWidget->addItems( nList );

    QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + ui->listWidget->item(0)->text();
    iw->load(filen);
    //iw->load(ui->listWidget->item(0)->text());
    iw->show();

    for(int i=0;i<4;i++){
        iw->ls[i]->setGeometry(0,0,0,0);
        iw->ls[i]->setVisible(true);
        iw->cx[i]=0;
        iw->cy[i]=0;
        iw->rx[i]=0;
        iw->ry[i]=0;
        iw->angle[i]=0;
        iw->radius[i]=0;
        iw->origins[i]=QPoint(0,0);

    }
    iw->sphere1->setGeometry(QRect(0,0,0,0));
    ui->sph1->setText(QString("-"));
    ui->cx1spin->setVisible(false);
    ui->cy1spin->setVisible(false);
    ui->r1spin->setVisible(false);
    ui->r1spin->setVisible(false);
    ui->r2_1Spin->setVisible(false);
    ui->angle1spin->setVisible(false);
    iw->sphere2->setGeometry(QRect(0,0,0,0));
    ui->sph2->setText(QString("-"));
    ui->cx2spin->setVisible(false);
    ui->cy2spin->setVisible(false);
    ui->r2spin->setVisible(false);
    ui->r2spin->setVisible(false);
    ui->r2_2Spin->setVisible(false);
    ui->angle2spin->setVisible(false);
    iw->sphere3->setGeometry(QRect(0,0,0,0));
    ui->sph3->setText(QString("-"));
    ui->cx3spin->setVisible(false);
    ui->cy3spin->setVisible(false);
    ui->r3spin->setVisible(false);
    ui->r3spin->setVisible(false);
    ui->r2_3Spin->setVisible(false);
    ui->angle3spin->setVisible(false);
    iw->sphere4->setGeometry(QRect(0,0,0,0));
    ui->sph4->setText(QString("-"));
    ui->cx4spin->setVisible(false);
    ui->cy4spin->setVisible(false);
    ui->r4spin->setVisible(false);
    ui->r4spin->setVisible(false);
    ui->r2_4Spin->setVisible(false);
    ui->angle4spin->setVisible(false);

    // qDebug(nList);
    QApplication::processEvents();
}

void RTITool::on_listWidget_itemDoubleClicked(QListWidgetItem *item)
{
    // Load an image
    Mat image;
    /*  QString name = ui->folderName->text()  + QDir::separator() + "images" + QDir::separator()  + item->text();
    image = imread(name.toStdString());   // Read the file
    qDebug(item->text().toLatin1());
    if(! image.data )         // Check for invalid input
    {

        return;
    }
*/
    iw->load(ui->folderName->text()  + QDir::separator() + "images" + QDir::separator()  + item->text());
    /*  // Create a window for display.
      namedWindow( "Display window", CV_WINDOW_AUTOSIZE );

      // Show our image inside it.
      imshow( "Display window", image );

      waitKey(0);              // Wait for a keystroke in the window*/
}

void RTITool::on_actionEstimate_triggered()
{
    int ax,ay,bx,by;
    Mat rois1, groi;
    Mat image;
    int cs=0;


    for(int ns=0;ns<4;ns++)
        iw->lights[ns].clear();


    Rect rc[4];

    // loop over spheres
    for(int ns=0;ns<4;ns++)
        if(iw->radius[ns]>0 && iw->cx[ns] >0 && iw->cy[ns] >0)
        {

            ax = (int) (iw->origins[ns].x() / iw->scaleFactor);
            ay = (int) (iw->origins[ns].y() / iw->scaleFactor);

            rc[ns] = Rect(ax+iw->cx[ns]-iw->radius[ns],ay+iw->cy[ns]-iw->radius[ns],2*iw->radius[ns]+1,2*iw->radius[ns]+1);


            // loop over all the images of the list

            for(int row = 0; row < ui->listWidget->count(); row++)
            {
                QListWidgetItem *item = ui->listWidget->item(row);
                QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
                image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
                image(rc[ns]).copyTo(rois1);

                double* lv=Highlight(rois1,iw->cx[ns],iw->cy[ns],iw->radius[ns]);
                qDebug() << lv[0] << " " << lv[1] << " " << lv[2] << endl;
                image.release();
                iw->lights[cs].push_back(lv);
            }
            cs++;
        }
    ui->dirlab->setText("OK");
    iw->setVisible(false);
}


void RTITool::toggleSph1()
{

    ui->sph1->setText(QString("OK"));
    Mat image;
    Mat rois1, groi;
    int ax,ay,bx,by;
    int cs=0;

    iw->sphere1->frameGeometry().getCoords(&ax,&ay,&bx,&by);
    //qDebug() << iw->scaleFactor << "mmm";
    if(bx>0 && by>0){

        ax = (int) ((double)ax / iw->scaleFactor);
        ay = (int) ((double)ay / iw->scaleFactor);
        bx = (int) ((double)bx / iw->scaleFactor);
        by = (int) ((double)by / iw->scaleFactor);
        Rect r(ax,ay,bx-ax,by-ay);


        QListWidgetItem *item;
        if(ui->onMax->isChecked()){
            if(!maxi.size().area()>0)
                on_pushButton_clicked();
            image=maxi;
        }
        else{
            item= ui->listWidget->currentItem();

            // image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);
            QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
            image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
        }
        image(r).copyTo(rois1);
        cv::cvtColor(rois1,groi,CV_BGR2GRAY);
        float cx,cy,radius;

        cx = groi.cols/2;
        cy = groi.rows/2;
        radius = min(cx,cy)-1;
        RotatedRect box;

        if(ui->autoFit->isChecked()){
            fitCircle(groi,&cx,&cy,&radius);

            box = segmentAndFitEllipseMax(rois1);
            qDebug() <<"Returned features-Center: " << box.center.x << " "  << box.center.y << " Angle " << box.angle <<" Width "<<box.size.width<<" Height "<<box.size.height;
            iw->boxE[0]=box;
            QFile bfile(ui->folderName->text() + QDir::separator() + "box0.txt");
            if (!bfile.open(QFile::WriteOnly | QFile::Text)) {
                qDebug() << "error";
            }
            else{
                QTextStream stream( &bfile );
                stream << box.center.x << " "  << box.center.y << " Angle " << box.angle <<" Width "<<box.size.width<<" Height "<<box.size.height;
                bfile.close();
            }
        }

        qDebug() <<"Returned features-Center: " << cx << " "  << cy << "  "<< radius;

        iw->ls[0]->setGeometry((ax)*iw->scaleFactor,(ay)*iw->scaleFactor,(bx-ax)*iw->scaleFactor,(by-ay)*iw->scaleFactor);
        iw->ls[0]->setScaledContents(true);


        //   circle(rois1, Point(cx,cy), radius, color, 1, 8, 0);		// Disegno il cerchio effettivo
        // namedWindow("ROI", WINDOW_AUTOSIZE);
        // imshow("ROI", rois1);
        image.release();
        groi.release();
        rois1.release();

        if(ui->projBox->isChecked())
        {
            iw->cx[0]=box.center.x;
            iw->cy[0]=box.center.y;
            iw->rx[0]=0.5*box.size.width;
            iw->ry[0]=0.5*box.size.height;
            iw->radius[0]=max(box.size.width,box.size.height);
            iw->angle[0]=box.angle;

            ui->r1spin->setValue(iw->rx[0]);
            ui->r2_1Spin->setValue(iw->ry[0]);
            ui->angle1spin->setValue(iw->angle[0]);
        }
        else
        {
            iw->cx[0]=cx;
            iw->cy[0]=cy;
            iw->radius[0]=radius;
            ui->r1spin->setValue(radius);
        }
        ui->cx1spin->setValue(cx+ax);
        ui->cy1spin->setValue(cy+ay);


        ui->cx1spin->setVisible(true);
        ui->cy1spin->setVisible(true);
        ui->r1spin->setVisible(true);

        if(ui->projBox->isChecked())
        {
            ui->r2_1Spin->setVisible(true);
            ui->angle1spin->setVisible(true);
        }

        QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
        pixmap.fill(QColor("transparent"));

        QPainter painter(&pixmap);
        if(ui->projBox->isChecked())
        {
            //  painter.setPen(QPen(Qt::blue));
            // painter.drawEllipse(QPointF(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor),(0.5*box.size.width*iw->scaleFactor),(0.5*box.size.height*iw->scaleFactor));

            painter.setPen(QPen(Qt::green));
            painter.save();
            QTransform trans;
            // Move to the center of the widget
            //trans.translate(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor);
            trans.translate(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor);
            // Do the rotation
            trans.rotate(iw->angle[0]);
            // Move to the center of the image
            painter.setTransform(trans);
            painter.drawEllipse(QPointF(0,0),(iw->rx[0]*iw->scaleFactor),(iw->ry[0]*iw->scaleFactor));
            painter.restore();
        }
        else
        {
            painter.setPen(QPen(Qt::red));
            painter.drawEllipse(QPointF(cx*iw->scaleFactor,cy*iw->scaleFactor),(radius*iw->scaleFactor),(radius*iw->scaleFactor));
        }


        iw->ls[0]->setPixmap(pixmap);
        iw->ls[0]->setVisible(true);
    }


}

void RTITool::toggleSph2()
{

    ui->sph2->setText(QString("OK"));
    Mat image;
    Mat rois1, groi;
    int ax,ay,bx,by;
    int cs=0;

    iw->sphere2->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    if(bx>0 && by>0){

        ax = (int) ((double)ax / iw->scaleFactor);
        ay = (int) ((double)ay / iw->scaleFactor);
        bx = (int) ((double)bx / iw->scaleFactor);
        by = (int) ((double)by / iw->scaleFactor);
        Rect r(ax,ay,bx-ax,by-ay);

        QListWidgetItem *item;

        if(ui->onMax->isChecked()){
            if(!maxi.size().area()>0)
                on_pushButton_clicked();
            image=maxi;
        }
        else{
            item= ui->listWidget->currentItem();
            QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
            image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
            //image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);
        }
        //QListWidgetItem *item = ui->listWidget->currentItem();
        //image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);

        image(r).copyTo(rois1);
        cv::cvtColor(rois1,groi,CV_BGR2GRAY);
        float cx,cy,radius;

        cx = groi.cols/2;
        cy = groi.rows/2;
        radius = min(cx,cy)-1;
        RotatedRect box;

        if(ui->autoFit->isChecked()){
            fitCircle(groi,&cx,&cy,&radius);

            box = segmentAndFitEllipseMax(rois1);
            iw->boxE[1]=box;
            QFile bfile(ui->folderName->text() + QDir::separator() + "box1.txt");
            if (!bfile.open(QFile::WriteOnly | QFile::Text)) {
                qDebug() << "error";
            }
            else{
                QTextStream stream( &bfile );
                stream << box.center.x << " "  << box.center.y << " Angle " << box.angle <<" Width "<<box.size.width<<" Height "<<box.size.height;
                bfile.close();
            }
            //   qDebug() <<"Returned features-Center: " << box.center.x << " "  << box.center.y << " Angle " << box.angle <<" Width "<<box.size.width<<" Height "<<box.size.height;
        }

        //        qDebug() <<"Returned features-Center: " << cx << " "  << cy << "  "<< radius;

        iw->ls[1]->setGeometry((ax)*iw->scaleFactor,(ay)*iw->scaleFactor,(bx-ax)*iw->scaleFactor,(by-ay)*iw->scaleFactor);
        iw->ls[1]->setScaledContents(true);;;


        //   circle(rois1, Point(cx,cy), radius, color, 1, 8, 0);		// Disegno il cerchio effettivo
        // namedWindow("ROI", WINDOW_AUTOSIZE);
        // imshow("ROI", rois1);
        image.release();
        groi.release();
        rois1.release();

        if(ui->projBox->isChecked())
        {
            iw->cx[1]=box.center.x;
            iw->cy[1]=box.center.y;
            iw->rx[1]=0.5*box.size.width;
            iw->ry[1]=0.5*box.size.height;
            iw->radius[1]=max(box.size.width,box.size.height);
            iw->angle[1]=box.angle;

            ui->r2spin->setValue(iw->rx[1]);
            ui->r2_2Spin->setValue(iw->ry[1]);
            ui->angle2spin->setValue(iw->angle[1]);
        }
        else
        {
            iw->cx[1]=cx;
            iw->cy[1]=cy;
            iw->radius[1]=radius;
            ui->r2spin->setValue(radius);
        }
        ui->cx2spin->setValue(cx+ax);
        ui->cy2spin->setValue(cy+ay);


        ui->cx2spin->setVisible(true);
        ui->cy2spin->setVisible(true);
        ui->r2spin->setVisible(true);

        if(ui->projBox->isChecked())
        {
            ui->r2_2Spin->setVisible(true);
            ui->angle2spin->setVisible(true);
        }

        QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
        pixmap.fill(QColor("transparent"));

        QPainter painter(&pixmap);
        if(ui->projBox->isChecked())
        {
            //  painter.setPen(QPen(Qt::blue));
            // painter.drawEllipse(QPointF(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor),(0.5*box.size.width*iw->scaleFactor),(0.5*box.size.height*iw->scaleFactor));

            painter.setPen(QPen(Qt::green));
            painter.save();
            QTransform trans;
            // Move to the center of the widget
            //trans.translate(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor);
            trans.translate(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor);
            // Do the rotation
            trans.rotate(iw->angle[1]);
            // Move to the center of the image
            painter.setTransform(trans);
            painter.drawEllipse(QPointF(0,0),(iw->rx[1]*iw->scaleFactor),(iw->ry[1]*iw->scaleFactor));
            painter.restore();
        }
        else
        {
            painter.setPen(QPen(Qt::red));
            painter.drawEllipse(QPointF(cx*iw->scaleFactor,cy*iw->scaleFactor),(radius*iw->scaleFactor),(radius*iw->scaleFactor));
        }


        iw->ls[1]->setPixmap(pixmap);
        iw->ls[1]->setVisible(true);
    }


}

void RTITool::toggleSph3()
{

    ui->sph3->setText(QString("OK"));

    Mat image;
    Mat rois1, groi;
    int ax,ay,bx,by;
    int cs=0;

    iw->sphere3->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    if(bx>0 && by>0){

        ax = (int) ((double)ax / iw->scaleFactor);
        ay = (int) ((double)ay / iw->scaleFactor);
        bx = (int) ((double)bx / iw->scaleFactor);
        by = (int) ((double)by / iw->scaleFactor);
        Rect r(ax,ay,bx-ax,by-ay);

        QListWidgetItem *item ;
        //QListWidgetItem *item = ui->listWidget->currentItem();
        //image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);

        if(ui->onMax->isChecked()){
            if(!maxi.size().area()>0)
                on_pushButton_clicked();
            image=maxi;
        }
        else{
            item= ui->listWidget->currentItem();
            //image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);
            QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
            image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
        }

        image(r).copyTo(rois1);
        cv::cvtColor(rois1,groi,CV_BGR2GRAY);
        float cx,cy,radius;

        cx = groi.cols/2;
        cy = groi.rows/2;
        radius = min(cx,cy)-1;
        RotatedRect box;

        if(ui->autoFit->isChecked()){
            fitCircle(groi,&cx,&cy,&radius);

            box = segmentAndFitEllipseMax(rois1);
            iw->boxE[2]=box;
            QFile bfile(ui->folderName->text() + QDir::separator() + "box2.txt");
            if (!bfile.open(QFile::WriteOnly | QFile::Text)) {
                qDebug() << "error";
            }
            else{
                QTextStream stream( &bfile );
                stream << box.center.x << " "  << box.center.y << " Angle " << box.angle <<" Width "<<box.size.width<<" Height "<<box.size.height;
                bfile.close();
            }

            qDebug() <<"Returned features-Center: " << box.center.x << " "  << box.center.y << " Angle " << box.angle <<" Width "<<box.size.width<<" Height "<<box.size.height;
        }

        qDebug() <<"Returned features-Center: " << cx << " "  << cy << "  "<< radius;

        iw->ls[2]->setGeometry((ax)*iw->scaleFactor,(ay)*iw->scaleFactor,(bx-ax)*iw->scaleFactor,(by-ay)*iw->scaleFactor);
        iw->ls[2]->setScaledContents(true);;;


        //   circle(rois1, Point(cx,cy), radius, color, 1, 8, 0);		// Disegno il cerchio effettivo
        // namedWindow("ROI", WINDOW_AUTOSIZE);
        // imshow("ROI", rois1);
        image.release();
        groi.release();
        rois1.release();

        if(ui->projBox->isChecked())
        {
            iw->cx[2]=box.center.x;
            iw->cy[2]=box.center.y;
            iw->rx[2]=0.5*box.size.width;
            iw->ry[2]=0.5*box.size.height;
            iw->radius[2]=max(box.size.width,box.size.height);
            iw->angle[2]=box.angle;

            ui->r3spin->setValue(iw->rx[2]);
            ui->r2_3Spin->setValue(iw->ry[2]);
            ui->angle3spin->setValue(iw->angle[2]);
        }
        else
        {
            iw->cx[2]=cx;
            iw->cy[2]=cy;
            iw->radius[2]=radius;
            ui->r3spin->setValue(radius);
        }
        ui->cx3spin->setValue(cx+ax);
        ui->cy3spin->setValue(cy+ay);


        ui->cx3spin->setVisible(true);
        ui->cy3spin->setVisible(true);
        ui->r3spin->setVisible(true);

        if(ui->projBox->isChecked())
        {
            ui->r2_3Spin->setVisible(true);
            ui->angle3spin->setVisible(true);
        }

        QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
        pixmap.fill(QColor("transparent"));

        QPainter painter(&pixmap);
        if(ui->projBox->isChecked())
        {
            painter.setPen(QPen(Qt::green));
            painter.save();
            QTransform trans;
            // Move to the center of the widget
            trans.translate(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor);
            // Do the rotation
            trans.rotate(iw->angle[2]);
            // Move to the center of the image
            painter.setTransform(trans);
            painter.drawEllipse(QPointF(0,0),(iw->rx[2]*iw->scaleFactor),(iw->ry[2]*iw->scaleFactor));
            painter.restore();
        }
        else
        {
            painter.setPen(QPen(Qt::red));
            painter.drawEllipse(QPointF(cx*iw->scaleFactor,cy*iw->scaleFactor),(radius*iw->scaleFactor),(radius*iw->scaleFactor));
        }


        iw->ls[2]->setPixmap(pixmap);
        iw->ls[2]->setVisible(true);
    }


}
void RTITool::toggleSph4()
{

    ui->sph4->setText(QString("OK"));
    Mat image;
    Mat rois1, groi;
    int ax,ay,bx,by;
    int cs=0;

    iw->sphere4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    if(bx>0 && by>0){

        ax = (int) ((double)ax / iw->scaleFactor);
        ay = (int) ((double)ay / iw->scaleFactor);
        bx = (int) ((double)bx / iw->scaleFactor);
        by = (int) ((double)by / iw->scaleFactor);
        Rect r(ax,ay,bx-ax,by-ay);

        QListWidgetItem *item ;

        if(ui->onMax->isChecked()){
            if(!maxi.size().area()>0)
                on_pushButton_clicked();
            image=maxi;
        }
        else{
            item= ui->listWidget->currentItem();
            QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
            image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
            //image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);
        }


        image(r).copyTo(rois1);
        cv::cvtColor(rois1,groi,CV_BGR2GRAY);
        float cx,cy,radius;

        cx = groi.cols/2;
        cy = groi.rows/2;
        radius = min(cx,cy)-1;

        RotatedRect box;

        if(ui->autoFit->isChecked()){
            fitCircle(groi,&cx,&cy,&radius);

            box = segmentAndFitEllipseMax(rois1);
            iw->boxE[3]=box;

           // qDebug() <<"Returned features-Center: " << box.center.x << " "  << box.center.y << " Angle " << box.angle <<" Width "<<box.size.width<<" Height "<<box.size.height;
            QFile bfile(ui->folderName->text() + QDir::separator() + "box3.txt");
            if (!bfile.open(QFile::WriteOnly | QFile::Text)) {
                qDebug() << "error";
            }
            else{
                QTextStream stream( &bfile );
                stream << box.center.x << " "  << box.center.y << " Angle " << box.angle <<" Width "<<box.size.width<<" Height "<<box.size.height;
                bfile.close();
            }
        }


        iw->ls[3]->setGeometry((ax)*iw->scaleFactor,(ay)*iw->scaleFactor,(bx-ax)*iw->scaleFactor,(by-ay)*iw->scaleFactor);
        iw->ls[3]->setScaledContents(true);;;

        //   circle(rois1, Point(cx,cy), radius, color, 1, 8, 0);		// Disegno il cerchio effettivo
        // namedWindow("ROI", WINDOW_AUTOSIZE);
        // imshow("ROI", rois1);

        image.release();
        groi.release();
        rois1.release();

        if(ui->projBox->isChecked())
        {
            iw->cx[3]=box.center.x;
            iw->cy[3]=box.center.y;
            iw->rx[3]=0.5*box.size.width;
            iw->ry[3]=0.5*box.size.height;
            iw->radius[3]=max(box.size.width,box.size.height);
            iw->angle[3]=box.angle;

            ui->r4spin->setValue(iw->rx[3]);
            ui->r2_4Spin->setValue(iw->ry[3]);
            ui->angle4spin->setValue(iw->angle[3]);
        }
        else
        {
            iw->cx[3]=cx;
            iw->cy[3]=cy;
            iw->radius[3]=radius;
            ui->r4spin->setValue(radius);
        }
        ui->cx4spin->setValue(cx+ax);
        ui->cy4spin->setValue(cy+ay);


        ui->cx4spin->setVisible(true);
        ui->cy4spin->setVisible(true);
        ui->r4spin->setVisible(true);

        if(ui->projBox->isChecked())
        {
            ui->r2_4Spin->setVisible(true);
            ui->angle4spin->setVisible(true);
        }

        QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
        pixmap.fill(QColor("transparent"));

        QPainter painter(&pixmap);
        if(ui->projBox->isChecked())
        {
            painter.setPen(QPen(Qt::green));
            painter.save();
            QTransform trans;
            // Move to the center of the widget
            //trans.translate(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor);
            trans.translate(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor);
            // Do the rotation
            trans.rotate(iw->angle[3]);
            // Move to the center of the image
            painter.setTransform(trans);
            painter.drawEllipse(QPointF(0,0),(iw->rx[3]*iw->scaleFactor),(iw->ry[3]*iw->scaleFactor));
            painter.restore();
        }
        else
        {
            painter.setPen(QPen(Qt::red));
            painter.drawEllipse(QPointF(cx*iw->scaleFactor,cy*iw->scaleFactor),(radius*iw->scaleFactor),(radius*iw->scaleFactor));
        }


        iw->ls[3]->setPixmap(pixmap);
        iw->ls[3]->setVisible(true);
    }


}

void RTITool::toggleW1()
{

    ui->w1l->setText(QString("OK"));
}
void RTITool::toggleW2()
{

    ui->w2l->setText(QString("OK"));
}
void RTITool::toggleW3()
{

    ui->w3l->setText(QString("OK"));
}
void RTITool::toggleW4()
{

    ui->w4l->setText(QString("OK"));
}

void RTITool::areaCrop()
{
    int a,b,c,d;
    iw->cropArea->geometry().getCoords(&a,&b,&c,&d);

    ui->spinOx->setValue(int(a/iw->scaleFactor));
    ui->spinOy->setValue(int(b/iw->scaleFactor));
    ui->spinSx->setValue(int((c-a)/iw->scaleFactor));
    ui->spinSy->setValue(int((d-b)/iw->scaleFactor));
}

void RTITool::on_sph1but_clicked()
{
    iw->active=5;
}

void RTITool::on_sph2but_clicked()
{
    iw->active=6;
}

void RTITool::on_sph3but_clicked()
{
    iw->active=7;
}

void RTITool::on_sph4but_clicked()
{
    iw->active=8;
}



void RTITool::on_rems1_clicked()
{
    iw->sphere1->setGeometry(QRect(0,0,0,0));
    iw->ls[0]->setGeometry(0,0,0,0);
    iw->ls[0]->setVisible(true);
    iw->cx[0]=0;
    iw->cy[0]=0;
    iw->radius[0]=0;
    iw->origins[0]=QPoint(0,0);
    ui->sph1->setText(QString("-"));
    ui->cx1spin->setVisible(false);
    ui->cy1spin->setVisible(false);
    ui->r1spin->setVisible(false);


    iw->rx[0]=0;
    iw->ry[0]=0;
    iw->angle[0]=0;

    ui->r1spin->setVisible(false);
    ui->r2_1Spin->setVisible(false);
    ui->angle1spin->setVisible(false);

}

void RTITool::on_rems2_clicked()
{
    iw->sphere2->setGeometry(QRect(0,0,0,0));
    iw->ls[1]->setGeometry(0,0,0,0);
    iw->ls[1]->setVisible(true);
    iw->cx[1]=0;
    iw->cy[1]=0;
    iw->origins[1]=QPoint(0,0);
    iw->radius[1]=0;
    ui->sph2->setText(QString("-"));
    ui->cx2spin->setVisible(false);
    ui->cy2spin->setVisible(false);
    ui->r2spin->setVisible(false);

    iw->rx[1]=0;
    iw->ry[1]=0;
    iw->angle[1]=0;

    ui->r2spin->setVisible(false);
    ui->r2_2Spin->setVisible(false);
    ui->angle2spin->setVisible(false);
}

void RTITool::on_rems3_clicked()
{
    iw->sphere3->setGeometry(QRect(0,0,0,0));
    iw->ls[2]->setGeometry(0,0,0,0);
    iw->ls[2]->setVisible(true);
    iw->cx[2]=0;
    iw->cy[2]=0;
    iw->origins[2]=QPoint(0,0);
    iw->radius[2]=0;
    ui->sph3->setText(QString("-"));
    ui->cx3spin->setVisible(false);
    ui->cy3spin->setVisible(false);
    ui->r3spin->setVisible(false);

    iw->rx[2]=0;
    iw->ry[2]=0;
    iw->angle[2]=0;

    ui->r3spin->setVisible(false);
    ui->r2_3Spin->setVisible(false);
    ui->angle3spin->setVisible(false);
}

void RTITool::on_rems4_clicked()
{
    iw->sphere4->setGeometry(QRect(0,0,0,0));
    iw->ls[3]->setGeometry(0,0,0,0);
    iw->ls[3]->setVisible(true);
    iw->cx[3]=0;
    iw->cy[3]=0;
    iw->radius[3]=0;
    iw->origins[3]=QPoint(0,0);
    ui->sph4->setText(QString("-"));
    ui->cx4spin->setVisible(false);
    ui->cy4spin->setVisible(false);
    ui->r4spin->setVisible(false);
    iw->rx[3]=0;
    iw->ry[3]=0;
    iw->angle[3]=0;

    ui->r4spin->setVisible(false);
    ui->r2_4Spin->setVisible(false);
    ui->angle4spin->setVisible(false);
}

void RTITool::on_lightEst_clicked()
{
    int ax,ay,bx,by;
    Mat rois1;
    Mat image;
    int cs=0;
    QListWidgetItem *item;

    if(ui->projBox->isChecked() && iw->cameraMatrix.at<double>(0,2) == 0){
        ui->msgBox->append("ERROR: no calibration parameters loaded");
        return;
    }

    for (int i=0; i<4; i++)
        iw->lights[i].clear();

    int steps=4*ui->listWidget->count();
    int sp=0;
    QProgressDialog progress("Estimating directions...", "", 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);
    progress.setCancelButton(0);
    progress.setWindowTitle("Progress Dialog");
    progress.show( );
    QCoreApplication::processEvents();


    QString  sphFilename="spherecentersRtitool.sph";
    QFile sphfile(sphFilename);
    QTextStream stream( &sphfile );
    if (!sphfile.open(QFile::WriteOnly | QFile::Text)) {
        qDebug() << "error";
    }

    else{
        stream<< "Sphere centers positions: x y z"<<"\n";
    }
    for(int ns=0;ns<4;ns++)
        if(iw->radius[ns]>0 && iw->cx[ns] >0 && iw->cy[ns] >0)
        {
            qDebug() <<"\nNumberofsphere "<<ns;
            // we have the ns th sphere. we could/should handle missing ones

            qDebug() << "Scale factor " << iw->scaleFactor;
            //    iw->sphere1->frameGeometry().getCoords(&ax,&ay,&bx,&by);
            //if (ns == 0) iw->sphere1->frameGeometry().getCoords(&ax,&ay,&bx,&by);
            //if (ns == 1) iw->sphere2->frameGeometry().getCoords(&ax,&ay,&bx,&by);

            //estimate annotated sphere region in original image coordinates
            ax = (iw->origins[ns].x() / iw->scaleFactor);
            ay = (iw->origins[ns].y() / iw->scaleFactor);

            bx = (int) (iw->ends[ns].x()/ iw->scaleFactor);
            by = (int) (iw->ends[ns].y() / iw->scaleFactor);
            // bounding box of sphere region



            for(int row = 0; row < ui->listWidget->count(); row++)
            {
                item = ui->listWidget->item(row);
                //    image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);
                QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
                image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);

                //qDebug() <<  item->text();
                //qDebug() << iw->cx[ns] << " " << iw->radius[ns] << iw->cy[ns] << endl;

                double* lv;
                cv::Point2f hpos;

                if(ui->projBox->isChecked())
                {
                    Rect rc2(ax, ay, 2*iw->ry[ns] + 100, 2*iw->rx[ns]+100);

                    image(rc2).copyTo(rois1);

                    hpos = getHighlightPosition(rois1, iw->boxE[ns]);

                    hpos.x = hpos.x+ax;
                    hpos.y = hpos.y+ay;
                    RotatedRect box = iw->boxE[ns];
                    box.center.x =box.center.x+ax;
                    box.center.y =box.center.y+ay;

                    double sphereRad = ui->spinSphere->value();

                    QString radFilename="radius.txt";
                    QFile radFile(radFilename);
                    QTextStream stream( &radFile );
                    if (!sphfile.open(QFile::WriteOnly | QFile::Text)) {
                        qDebug() << "error";
                    }

                    else{
                        stream << sphereRad <<"\n";
                    }
                    radFile.close();


                    lv = getLightDirection(box, hpos, iw->cameraMatrix, sphereRad, stream);


                }

                else{
                    Rect rc(ax+iw->cx[ns] - iw->radius[ns],ay+iw->cy[ns]-iw->radius[ns],2*iw->radius[ns]+1,2*iw->radius[ns]+1);

                    qDebug() << ns << "---" << ax << " " << iw->cx[ns];
                    qDebug() << "---" << ay << " " << iw->cy[ns];
                    image(rc).copyTo(rois1);

                    //                   imshow("region",rois1);
                    //                   waitKey();
                    lv=Highlight(rois1,iw->cx[ns],iw->cy[ns],iw->radius[ns]);

                }

                qDebug() << "Light " << row;
                qDebug() << lv[0] << " " << lv[1] << " " << lv[2] << endl;
                image.release();
                rois1.release();
                iw->lights[cs].push_back(lv);

                progress.setValue(100*ns*row/(ui->listWidget->count()*4));
                QApplication::processEvents();
            }
            cs++;
        }

    if(ui->projBox->isChecked())
    {

        QString str = QString("Persp. %1 spheres").arg(cs);
        ui->direLab->setText(str);
        ui->msgBox->append("Light directions estimated with perspective projection");
    }
    else {
        QString str = QString("Ortho %1 spheres").arg(cs);
        ui->direLab->setText(str);
        ui->msgBox->append("Light directions estimated with orthographic projection");
    }
    sphfile.close();


    saveLp(ui->folderName->text()+  QDir::separator() + "file.lp");

    if(cs>2)
        RTITool::on_interpDir_clicked();

    progress.setValue(100);
}

void RTITool::saveLp(QString fileName){

    int ns=0;
    int sn[4]={0,0,0,0};
    for(int i=0;i<4;i++)
        if( iw->radius[i]>0){
            sn[ns]=i;
            ns=ns+1;
        }


    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug() << "error";
    }
    else{
        QTextStream stream( &file );
        stream << ui->listWidget->count() << " " << ns << " ";

        if ( ui->projBox->isChecked() )
            stream << "2 ";
        else
            stream << "1 ";

        for(int i=0;i<ns;i++){
            if ( ui->projBox->isChecked() ){
                double cxx = iw->cx[i] + (iw->origins[i].x() / iw->scaleFactor);
                double cyy = iw->cy[i] + (iw->origins[i].y() / iw->scaleFactor);
                stream <<  iw->angle[sn[i]] << " " << iw->rx[sn[i]] << " " <<  iw->ry[sn[i]] << " " <<  cxx << " "<<  cyy << " ";
            }
            else{
                double cxx = iw->cx[i] + (iw->origins[i].x() / iw->scaleFactor);
                double cyy = iw->cy[i] + (iw->origins[i].y() / iw->scaleFactor);
                stream <<  iw->radius[sn[i]] << " "<<  cxx << " "<<  cyy << " ";
            }
        }
        stream << "\n";
        for(int i=0;i<ui->listWidget->count();i++){
            stream  << ui->listWidget->item(i)->text() ;
            for(int k=0;k<ns;k++)
                stream << " " << iw->lights[k].at(i)[0] <<  " " << iw->lights[k].at(i)[1] << " " << iw->lights[k].at(i)[2];

            stream << "\n";
        }
    }
    file.close();
}




void RTITool::saveOLp(QString fileName){

    int ns=0;
    int sn[4]={0,0,0,0};
    for(int i=0;i<4;i++)
        if( iw->radius[i]>0){
            sn[ns]=i;
            ns=ns+1;
        }


    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug() << "error";
    }
    else{
        QTextStream stream( &file );
        stream << ui->listWidget->count() << " " << ns << "\n";


        for(int i=0;i<ui->listWidget->count();i++){
            stream  << ui->listWidget->item(i)->text() ;
            double xx=0,yy=0,zz=0;
            for(int k=0;k<ns;k++){
            xx=xx+iw->lights[k].at(i)[0];
            yy=yy+iw->lights[k].at(i)[1];
            zz=zz+iw->lights[k].at(i)[2];
            }
                xx=xx/ns; yy=yy/ns;zz=zz/ns;

                stream << " " << xx <<  " " << yy << " " << zz << "\n";
        }
    }
    file.close();
}


void RTITool::on_saveLp_clicked()
{

    QString filename = QFileDialog::getSaveFileName(
                this,
                tr("File name"),
                QDir::currentPath(),
                tr("*.lp") );

    saveOLp(filename);


}



void RTITool::on_cx1spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere1->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->cx[0] = arg1-ax;


    // QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));

    pixmap.fill(QColor("transparent"));
    iw->ls[0]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        //  painter.setPen(QPen(Qt::blue));
        // painter.drawEllipse(QPointF(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor),(0.5*box.size.width*iw->scaleFactor),(0.5*box.size.height*iw->scaleFactor));

        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        //trans.translate(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor);
        trans.translate(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[0]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[0]*iw->scaleFactor),(iw->ry[0]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor));
    }

    iw->ls[0]->setPixmap(pixmap);
    iw->ls[0]->setVisible(true);

}

void RTITool::on_cy1spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere1->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->cy[0] = arg1-ay;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[0]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        //  painter.setPen(QPen(Qt::blue));
        // painter.drawEllipse(QPointF(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor),(0.5*box.size.width*iw->scaleFactor),(0.5*box.size.height*iw->scaleFactor));

        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[0]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[0]*iw->scaleFactor),(iw->ry[0]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor));
    }
    iw->ls[0]->setPixmap(pixmap);
    iw->ls[0]->setVisible(true);
}

void RTITool::on_r1spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere1->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    if(ui->projBox->isChecked())
    {
        iw->rx[0] = arg1;
    }
    else
        iw->radius[0] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[0]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[0]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[0]*iw->scaleFactor),(iw->ry[0]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor));
    }
    iw->ls[0]->setPixmap(pixmap);
    iw->ls[0]->setVisible(true);
}

void RTITool::on_cx2spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere2->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->cx[1] = arg1-ax;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[1]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[1]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[1]*iw->scaleFactor),(iw->ry[1]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor));
    }
    iw->ls[1]->setPixmap(pixmap);
    iw->ls[1]->setVisible(true);

}

void RTITool::on_cy2spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere2->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->cy[1] = arg1-ay;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[1]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[1]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[1]*iw->scaleFactor),(iw->ry[1]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor));
    }
    iw->ls[1]->setPixmap(pixmap);
    iw->ls[1]->setVisible(true);
}

void RTITool::on_r2spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere2->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    if(ui->projBox->isChecked())
    {
        iw->rx[1] = arg1;
    }
    else
        iw->radius[1] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[1]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        //  painter.setPen(QPen(Qt::blue));
        // painter.drawEllipse(QPointF(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor),(0.5*box.size.width*iw->scaleFactor),(0.5*box.size.height*iw->scaleFactor));

        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        //trans.translate(box.center.x*iw->scaleFactor,box.center.y*iw->scaleFactor);
        trans.translate(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[1]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[1]*iw->scaleFactor),(iw->ry[1]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor));
    }
    iw->ls[1]->setPixmap(pixmap);
    iw->ls[1]->setVisible(true);
}

void RTITool::on_cx3spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere3->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->cx[2] = arg1-ax;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[2]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[2]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[2]*iw->scaleFactor),(iw->ry[2]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor));
    }
    iw->ls[2]->setPixmap(pixmap);
    iw->ls[2]->setVisible(true);

}

void RTITool::on_cy3spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere3->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->cy[2] = arg1-ay;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[2]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[2]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[2]*iw->scaleFactor),(iw->ry[2]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor));
    }
    iw->ls[2]->setPixmap(pixmap);
    iw->ls[2]->setVisible(true);
}

void RTITool::on_r3spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere3->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    if(ui->projBox->isChecked())
    {
        iw->rx[2] = arg1;
    }
    else iw->radius[2] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[2]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[2]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[2]*iw->scaleFactor),(iw->ry[2]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor));
    }
    iw->ls[2]->setPixmap(pixmap);
    iw->ls[2]->setVisible(true);
}

void RTITool::on_cx4spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->cx[3] = arg1-ax;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[3]->setScaledContents(true);;;
    QPainter painter(&pixmap);
    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[3]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[3]*iw->scaleFactor),(iw->ry[3]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor));
    }
    iw->ls[3]->setPixmap(pixmap);
    iw->ls[3]->setVisible(true);

}

void RTITool::on_cy4spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->cy[3] = arg1-ay;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[3]->setScaledContents(true);;;
    QPainter painter(&pixmap); if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[3]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[3]*iw->scaleFactor),(iw->ry[3]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor));
    }
    iw->ls[3]->setPixmap(pixmap);
    iw->ls[3]->setVisible(true);
}

void RTITool::on_r4spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    if(ui->projBox->isChecked())
    {
        iw->rx[3] = arg1;
    }
    else
        iw->radius[3] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[3]->setScaledContents(true);;;
    QPainter painter(&pixmap); if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        // Move to the center of the widget
        trans.translate(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor);
        // Do the rotation
        trans.rotate(iw->angle[3]);
        // Move to the center of the image
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[3]*iw->scaleFactor),(iw->ry[3]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor));
    }
    iw->ls[3]->setPixmap(pixmap);
    iw->ls[3]->setVisible(true);
}

void RTITool::on_w1but_clicked()
{
    iw->active=1; // activate white box 1
}

void RTITool::on_w2but_clicked()
{
    iw->active=2;// activate white box 2
}

void RTITool::on_w3but_clicked()
{
    iw->active=3; // activate white box 3
}

void RTITool::on_w4but_clicked()
{
    iw->active=4; // activate white box 4
}


void RTITool::on_remw1_clicked()
{
    // remove white box 1
    iw->white1->setGeometry(QRect(0,0,0,0));
    iw->originw[0]=QPoint(0,0);
    ui->w1l->setText(QString("-"));

}


void RTITool::on_remw2_clicked()
{
     // remove white box 2
    iw->white2->setGeometry(QRect(0,0,0,0));
    iw->originw[1]=QPoint(0,0);
    ui->w2l->setText(QString("-"));
}

void RTITool::on_remw3_clicked()
{
     // remove white box 3
    iw->white3->setGeometry(QRect(0,0,0,0));
    iw->originw[2]=QPoint(0,0);
    ui->w3l->setText(QString("-"));
}

void RTITool::on_remw4_clicked()
{
     // remove white box 4
    iw->white4->setGeometry(QRect(0,0,0,0));
    ui->w4l->setText(QString("-"));
    iw->originw[3]=QPoint(0,0);
}

void RTITool::on_whiteEst_clicked()
{
    int ax,ay,bx,by;
    Mat image;
    Mat gim;
    //Mat rois1, groi;
    Mat A;
    Mat B;
    Mat X;
    Mat WC;

    QFile cofile(ui->folderName->text() + QDir::separator() + "corr.txt");

    if (!cofile.open(QIODevice::WriteOnly | QIODevice::Text))
        qDebug() << "error";

    QTextStream out(&cofile);

    out << "CORR_TYPE frame" << endl;

    for(int k=0; k<4; k++) {
        if(k==0) iw->white1->frameGeometry().getCoords(&ax,&ay,&bx,&by);
        else if(k==1) iw->white2->frameGeometry().getCoords(&ax,&ay,&bx,&by);
        else if(k==2)  iw->white3->frameGeometry().getCoords(&ax,&ay,&bx,&by);
        else  iw->white4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

        if(bx>0 && by>0){
            ax = (int) ((double)ax / iw->scaleFactor);
            ay = (int) ((double)ay / iw->scaleFactor);
            bx = (int) ((double)bx / iw->scaleFactor);
            by = (int) ((double)by / iw->scaleFactor);

            out << ax << " " << ay << " " << bx << " " << by << endl;
  // check coordinate (?)
        }
    }


    Mat CI;

    QProgressDialog pdialog("Estimating white correction","",0,100,this);
    pdialog.setWindowModality(Qt::WindowModal);
    pdialog.setCancelButton(0);
    pdialog.setValue(0);
    pdialog.setWindowTitle("Progress Dialog");

    pdialog.show();

    // estimating a second order polynomial fit of the frame color

    for(int ns=0;ns<6;ns++)
        iw->coeffs[ns].clear();

    // o 10
    A=Mat::zeros(100,6,CV_64FC1);
    B=Mat::zeros(100,1,CV_64FC1);

    double cx,cy;
    cx = iw->s.width()/2;
    cy = iw->s.height()/2;
    double lx, ly, lz, elev, azim;
    double tlx, tly, tlz, telev, tazim;
    double plx, ply, plz, pelev, pazim;

    int rmax;
    Mat immax;

    QListWidgetItem *item = ui->listWidget->currentItem();

    /* SAVE CORRECTION
    QString outname;
    if(!QDir(ui->folderName->text() + QDir::separator() +"TMP").exists())
        QDir().mkdir(ui->folderName->text() + QDir::separator() +"TMP");
*/

    elev =0;

    // if light interpolation is on
    if(ui->weightDir->isChecked()){
        if(iw->coilix[0].size()<ui->listWidget->count()) {
            ui->msgBox->append("Error: please estimate interpolated dirs first");
            return;
        }

    }

    double err=0;

    for(int row = 0; row < ui->listWidget->count(); row++)
    {

        pdialog.setValue(100*row/ui->listWidget->count());
        QApplication::processEvents();

        item = ui->listWidget->item(row);

        /* SAVE CORRECTION
        outname = ui->folderName->text() + QDir::separator() + "TMP/correction" + QString::number(row) + ".tif";
*/
        QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();

        if(!iw->force8bit){

            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
         }
        else
            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);

        CI = image.clone();
        cv::cvtColor(image,gim,CV_BGR2GRAY);

        if(ui->weightDir->isChecked()){

            tlx = cx*iw->coilix[0].at(row) +cy*iw->coilix[1].at(row) + iw->coilix[2].at(row);
            tly = cx*iw->coiliy[0].at(row) +cy*iw->coiliy[1].at(row) + iw->coiliy[2].at(row);
            tlz = cx*iw->coiliz[0].at(row) +cy*iw->coiliz[1].at(row) + iw->coiliz[2].at(row);
            double norm = sqrt(tlx*tlx+tly*tly+tlz*tlz);
            tlx=tlx/norm;tly=tly/norm;tlz=tlz/norm;
            telev = tlz;
            tazim = atan(tly/tlx);
            qDebug() <<  cx << cy << " - " << row << "QUA" << (tlz) ;
        }
        else {
            tlx=tly=tlz=0;
            int nl=0;
            for(int i=0; i<4;i++)
                if(! iw->lights[i].size() == 0){
                    tlx+=iw->lights[i].at(row)[0];
                    tly+=iw->lights[i].at(row)[1];
                    tlz+=iw->lights[i].at(row)[2];
                    nl++;
                }
            if(nl==0){
                ui->msgBox->append("Error: please estimate light direction first");
                return;
            }
            tlx/=nl;
            tly/=nl;
            tlz/=nl;
            double norm = sqrt(tlx*tlx+tly*tly+tlz*tlz);
            tlx=tlx/norm;tly=tly/norm;tlz=tlz/norm;
            qDebug() <<  "OK " << tlx << " " << tly << " " << (tlz) ;
            telev = tlz;
            tazim = atan(tly/tlx);
        }

        // here (2722-2757) i see I calculated light dirction in the center of the image in case of interpolated/non interpolated directions

        for(int k=0; k<4; k++) {
            if(k==0) iw->white1->frameGeometry().getCoords(&ax,&ay,&bx,&by);
            else if(k==1) iw->white2->frameGeometry().getCoords(&ax,&ay,&bx,&by);
            else if(k==2)  iw->white3->frameGeometry().getCoords(&ax,&ay,&bx,&by);
            else  iw->white4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

            if(bx>0 && by>0){
                ax = (int) ((double)ax / iw->scaleFactor);
                ay = (int) ((double)ay / iw->scaleFactor);
                bx = (int) ((double)bx / iw->scaleFactor);
                by = (int) ((double)by / iw->scaleFactor);

                // ok here i was sampling points on the rectangle. Skipping too large borders I think
                // and with an error (step). I changed
                // k index for the rectangle i,j indexes for samples on rectangle
                // the sampling before was strangely o_x_x_x_x_x_o_o
                //  now x_x_x_x_x  including first and last point in the rectangle

                int steps=5;

                for(int i=0;i<steps;i++)
                    for(int j=0;j<steps;j++){

                        double vx=ax+(int)((i)*(bx-ax)/(steps-1));
                        double vy=ay+(int)((j)*(by-ay)/(steps-1));

                        if(iw->depth==0) // 8 bit
                            B.at<double>(k*steps*steps+j*steps+i) = gim.at<unsigned char>(Point((int)vx,(int)vy));
                        else // 16 bit
                            B.at<double>(k*steps*steps+j*steps+i) = gim.at<unsigned short>(Point((int)vx,(int)vy));

                        A.at<double>(k*steps*steps+j*steps+i,0) = vx*vx;
                        A.at<double>(k*steps*steps+j*steps+i,1) = vy*vy;
                        A.at<double>(k*steps*steps+j*steps+i,2) = vx*vy;
                        A.at<double>(k*steps*steps+j*steps+i,3) = vx;
                        A.at<double>(k*steps*steps+j*steps+i,4) = vy;
                        A.at<double>(k*steps*steps+j*steps+i,5) = 1;

                    }
            }
            else {
                return;
            }
        }


        solve(A,B,X,DECOMP_SVD);

        for(int i=0;i<6;i++)
            iw->coeffs[i].push_back(X.at<double>(i));

        out << X.at<double>(0) << " " << X.at<double>(1) << " " << X.at<double>(2) << " " << X.at<double>(4) << " " << X.at<double>(4) << " " << X.at<double>(5) << " " << endl;


        for(int k=0; k<4; k++) {
            if(k==0) iw->white1->frameGeometry().getCoords(&ax,&ay,&bx,&by);
            else if(k==1) iw->white2->frameGeometry().getCoords(&ax,&ay,&bx,&by);
            else if(k==2)  iw->white3->frameGeometry().getCoords(&ax,&ay,&bx,&by);
            else  iw->white4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

            if(bx>0 && by>0){
                ax = (int) ((double)ax / iw->scaleFactor);
                ay = (int) ((double)ay / iw->scaleFactor);
                bx = (int) ((double)bx / iw->scaleFactor);
                by = (int) ((double)by / iw->scaleFactor);

                err=0;
                int steps=5;
                for(int i=0;i<steps;i++)
                    for(int j=0;j<steps;j++){


                        double vx=ax+(int)((i)*(bx-ax)/(steps-1));
                        double vy=ay+(int)((j)*(by-ay)/(steps-1));
                        double aa = vx*vx*X.at<double>(0)+ vy*vy*X.at<double>(1)+
                                vx*vy*X.at<double>(2)+ vx*X.at<double>(3)+
                                vy*X.at<double>(4)+ X.at<double>(5) ;

                        double bb;
                        if(iw->depth==0) // 8 bit
                            bb= gim.at<unsigned char>(Point((int)vx,(int)vy));
                        else
                            bb= gim.at<unsigned short>(Point((int)vx,(int)vy));

                        err=err+abs(aa-bb)/100;
                        // compute errors

                    }

            }
        }

        qDebug() << row << "error " << err;

#if 0//  We can save correction images for check
        /* save correction image */
        double aa;
        for(int i=0;i<CI.cols;i++)
            for(int j=0;j<CI.rows;j++){
                double vx =(i);
                double vy =(j);
                /*aa = vx*vx*X.at<double>(0)+ vy*vy*X.at<double>(1)+
                        vx*vy*X.at<double>(2)+ vx*X.at<double>(3)+
                        vy*X.at<double>(4)+ X.at<double>(5) ;*/

                aa = vx*vx*(iw->coeffs[0]).at(row)+ vy*vy*(iw->coeffs[1]).at(row)+
                        vx*vy*(iw->coeffs[2]).at(row)+ vx*(iw->coeffs[3]).at(row)+
                        vy*(iw->coeffs[4]).at(row)+ (iw->coeffs[5]).at(row);

                if(iw->depth==0){
                    Vec3b color = image.at<Vec3b>(Point(i,j));
                    image.at<Vec3b>(Point(i,j)) = color;
                    for(int k=0;k<3;k++)
                        color[k] = MIN(iw->maxval,MAX(0,aa));
                    CI.at<Vec3b>(Point(i,j)) = color;
                }
                else if(iw->depth==2){
                    Vec<unsigned short, 3>  color = image.at<Vec<unsigned short, 3> >(Point(i,j));
                    image.at<Vec<unsigned short, 3> >(Point(i,j)) = color;
                    for(int k=0;k<3;k++)
                        color[k] = MIN(iw->maxval,MAX(0,aa));
                    CI.at<Vec<unsigned short, 3> >(Point(i,j)) = color;
                }
            }
        imwrite( outname.toStdString(), CI );
        CI.release();

#endif
        image.release();


        //  rois1.release();
        //  groi.release();

    }

    ui->corrLab->setText("OK");

    /* grayref = cx*cx*(iw->coeffs[0]).at(rmax)+ cy*cy*(iw->coeffs[1]).at(rmax)+
            cx*cy*(iw->coeffs[2]).at(rmax)+ cx*(iw->coeffs[3]).at(rmax)+
            cy*(iw->coeffs[4]).at(rmax)+ (iw->coeffs[5]).at(rmax);
*/

}

void RTITool::on_saveCorrImages_clicked()
{
    QListWidgetItem *item;
    Mat WC;
    Mat CI;
    Mat image;
    Mat gim;
    double cx,cy;

    double lx, ly, lz, elev, azim, plx,ply,plz;
    double tlx, tly, tlz, telev, tazim;
    int ax,ay,bx,by;
    iw->cropArea->frameGeometry().getCoords(&ax,&ay,&bx,&by);
    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);
    Rect rc(ax,ay,bx-ax,by-ay);

    QFile cofile(ui->folderName->text() + QDir::separator() + "corrim.txt");

    if (!cofile.open(QIODevice::WriteOnly | QIODevice::Text))
        qDebug() << "error";

    QTextStream out(&cofile);

    out << ui->weightDir->isChecked() << endl;
    out << ui->refWhite->value() << endl;
    cofile.close();

    QString outname;
    if(!QDir(ui->folderName->text() + QDir::separator() +"CORR_IMG").exists())
        QDir().mkdir(ui->folderName->text() + QDir::separator() +"CORR_IMG");

    QProgressDialog pdialog("Saving corrected images","",0,100,this);
    pdialog.setWindowModality(Qt::WindowModal);
    pdialog.setCancelButton(0);
    pdialog.setValue(0);
    pdialog.setWindowTitle("Progress Dialog");

    pdialog.show();

    for(int row = 0; row < ui->listWidget->count(); row++)
    {
        if(ui->weightDir->isChecked()){
            tlx = cx*iw->coilix[0].at(row)+cy*iw->coilix[1].at(row)+ iw->coilix[2].at(row);
            tly = cx*iw->coiliy[0].at(row)+cy*iw->coiliy[1].at(row)+ iw->coiliy[2].at(row);
            tlz = cx*iw->coiliz[0].at(row)+cy*iw->coiliz[1].at(row)+ iw->coiliz[2].at(row);
            double norm = sqrt(tlx*tlx+tly*tly+tlz*tlz);
            tlx=tlx/norm;tly=tly/norm;tlz=tlz/norm;
        }
        else {
            tlx=tly=tlz=0;
            int nl=0;
            for(int i=0; i<4; i++)
                if(! iw->lights[i].size() == 0){
                    tlx+=iw->lights[i].at(row)[0];
                    tly+=iw->lights[i].at(row)[1];
                    tlz+=iw->lights[i].at(row)[2];
                    nl++;
                }
            if(nl==0){
                ui->msgBox->append("Error: please estimate light direction first");
                return;
            }
            tlx/=nl;
            tly/=nl;
            tlz/=nl;
            double norm = sqrt(tlx*tlx+tly*tly+tlz*tlz);
            tlx=tlx/norm;tly=tly/norm;tlz=tlz/norm;
            qDebug() <<  "OK " << tlx << " " << tly << " " << (tlz) ;
        }

        pdialog.setValue(100*row/ui->listWidget->count());
        QApplication::processEvents();


        item = ui->listWidget->item(row);

        outname = ui->folderName->text() + QDir::separator() +"CORR_IMG" + QDir::separator() + "corrected" + QString::number(row) + ".tif";

        QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
        if(!iw->force8bit)
            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
        else
            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
        double min, max;
        cv::minMaxLoc(image, &min, &max);
        qDebug() << "minmax " << min << max;

        cx = image.cols/2;
        cy = image.rows/2;


        CI = image.clone();
        cv::cvtColor(image,gim,CV_BGR2GRAY);
        if(ui->weightDir->isChecked()){

        }

        double aa;

        for(int i=0;i<image.cols;i++)
            for(int j=0;j<image.rows;j++){
                double vx =(i);
                double vy =(j);

                aa = vx*vx*(iw->coeffs[0]).at(row)+ vy*vy*(iw->coeffs[1]).at(row)+
                        vx*vy*(iw->coeffs[2]).at(row)+ vx*(iw->coeffs[3]).at(row)+
                        vy*(iw->coeffs[4]).at(row)+ (iw->coeffs[5]).at(row);

                if(iw->depth == 0){
                    Vec3b color = image.at<Vec3b>(Point(i,j));
                    unsigned char gray = gim.at<unsigned char>(Point(i,j));


                    if(ui->weightDir->isChecked()){
                        plx = vx*iw->coilix[0].at(row)+vy*iw->coilix[1].at(row)+ iw->coilix[2].at(row);
                        ply = vx*iw->coiliy[0].at(row)+vy*iw->coiliy[1].at(row)+ iw->coiliy[2].at(row);
                        plz = vx*iw->coiliz[0].at(row)+vy*iw->coiliz[1].at(row)+ iw->coiliz[2].at(row);
                        double norm = sqrt(plx*plx+ply*ply+plz*plz);
                        plx=plx/norm;ply=ply/norm;plz=plz/norm;
                        aa=abs(aa/plz);
                    }
                    else
                        aa=(aa/tlz);

                    double ww = ui->refWhite->value();
                    for(int k=0;k<3;k++)
                        color[k] = MIN(255,MAX(0,color[k]*ww*255.0/aa));

                    CI.at<Vec3b>(Point(i,j)) = color;

                }
                else{
                    Vec<unsigned short, 3>  color = image.at<Vec<unsigned short, 3> >(Point(i,j));
                    unsigned short gray = (unsigned short) gim.at<unsigned short>(Point(i,j));

                    if(ui->weightDir->isChecked()){

                    }
                    if(ui->weightDir->isChecked()){
                        plx = vx*iw->coilix[0].at(row)+vy*iw->coilix[1].at(row)+ iw->coilix[2].at(row);
                        ply = vx*iw->coiliy[0].at(row)+vy*iw->coiliy[1].at(row)+ iw->coiliy[2].at(row);
                        plz = vx*iw->coiliz[0].at(row)+vy*iw->coiliz[1].at(row)+ iw->coiliz[2].at(row);
                        double norm = sqrt(plx*plx+ply*ply+plz*plz);
                        plx=plx/norm;ply=ply/norm;plz=plz/norm;
                        aa=abs(aa/plz);

                    }
                    else
                        aa=abs(aa/tlz);
                    double ww = ui->refWhite->value();

                    for(int k=0;k<3;k++)
                        color[k] = (unsigned short) MIN(65535,MAX(0,(((unsigned short)color[k])*ww*65535.0/aa)));

                    CI.at<Vec<unsigned short, 3> >(Point(i,j)) = color;
                    CI.convertTo(CI,CV_16U);
                }
            }

        /*  if(ax >0 && bx>0)
            CI = CI(rc);*/

        imwrite( outname.toStdString(), CI );
        image.release();
        CI.release();
        gim.release();
    }

    ui->savecLab->setText("OK");

}

void RTITool::on_interpDir_clicked()
{
    Mat A;
    Mat B;
    Mat C;
    Mat D;
    Mat X;
    Mat Y;
    Mat Z;
    Mat IM;

    QString outname;

    for(int i=0;i<4;i++){
        if((iw->lights[i]).size() == 0) return;
    }
    for(int ns=0;ns<3;ns++){
        iw->coilix[ns].clear();
        iw->coiliy[ns].clear();
        iw->coiliz[ns].clear();
    }

    A=Mat::zeros(4,3,CV_64FC1);
    B=Mat::zeros(4,1,CV_64FC1);
    C=Mat::zeros(4,1,CV_64FC1);
    D=Mat::zeros(4,1,CV_64FC1);
    // Mat DX=Mat::zeros(iw->s.width(),iw->s.height(),CV_64FC1);
    // Mat DY=Mat::zeros(iw->s.width(),iw->s.height(),CV_64FC1);
    double dx,dy,dz;

    // if(!QDir("TMP").exists())
    //         QDir().mkdir("TMP");

    for(int row = 0; row < ui->listWidget->count(); row++)
    {
        //qDebug() << "############# Image " << row;
        for(int i=0;i<4;i++){

            double  ax = iw->cx[i] + (iw->origins[i].x() / iw->scaleFactor);
            double  ay = iw->cy[i] + (iw->origins[i].y() / iw->scaleFactor);

            B.at<double>(i) = iw->lights[i].at(row)[0];
            C.at<double>(i) = iw->lights[i].at(row)[1];
            D.at<double>(i) = iw->lights[i].at(row)[2];

            // qDebug() << ax+1+(int)((i+1)*groi.cols/7.0) << " " << ay+1+(int)((j+1)*groi.rows/7.0) << " " << B.at<double>(k*25+j*5+i) << " " << (double)groi.at<unsigned char>((int)((i+1)*groi.cols)/7,(int)((j+1)*groi.rows/7));
            A.at<double>(i,0) = ax;
            A.at<double>(i,1) = ay;
            A.at<double>(i,2) = 1;

            //qDebug() << "### Light " << i << " Hilight at (" << ax << " " << ay << ") dir = (" << iw->lights[i].at(row)[0] << " " << iw->lights[i].at(row)[1] << " " << iw->lights[i].at(row)[2] << ")";
        }

        solve(A,B,X,DECOMP_SVD);
        solve(A,C,Y,DECOMP_SVD);
        solve(A,D,Z,DECOMP_SVD);

        for(int i=0;i<3;i++){
            iw->coilix[i].push_back(X.at<double>(i));
            iw->coiliy[i].push_back(Y.at<double>(i));
            iw->coiliz[i].push_back(Z.at<double>(i));
        }

        for(int j=0;j<4;j++){
            qDebug() << "S1" << iw->lights[j].at(row)[0] << " " <<iw->lights[j].at(row)[1]<< " " <<iw->lights[j].at(row)[2];
            double  ax = iw->cx[j] + (iw->origins[j].x() / iw->scaleFactor);
            double  ay = iw->cy[j] + (iw->origins[j].y() / iw->scaleFactor);

            for(int i=0;i<3;i++){
                qDebug()<< X.at<double>(i);
                qDebug()<< Y.at<double>(i);
                qDebug()<< Z.at<double>(i);
            }
            qDebug() << ax*X.at<double>(0)+ay*X.at<double>(1)+ X.at<double>(2);
            qDebug() << ax*Y.at<double>(0)+ay*Y.at<double>(1)+ Y.at<double>(2);
            qDebug() << ax*Z.at<double>(0)+ay*Z.at<double>(1)+ Z.at<double>(2);

        }
        Vec3b color;
        QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + ui->listWidget->item(0)->text();

        if(row==0){
            //IM = imread((ui->listWidget->item(0))->text().toStdString(), CV_LOAD_IMAGE_COLOR);
            IM = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
            for(int i=0;i<iw->s.width();i++)
                for(int j=0;j<iw->s.height();j++){
                    dx = i*X.at<double>(0)+j*X.at<double>(1)+ X.at<double>(2);
                    dy = i*Y.at<double>(0)+j*Y.at<double>(1)+ Y.at<double>(2);
                    dz = i*Z.at<double>(0)+j*Z.at<double>(1)+ Z.at<double>(2);
                    float norm=sqrt(dx*dx+dy*dy+dz*dz);
                    dx=dx/norm;dy=dy/norm;dz=dz/norm;
                    color[0] = (255*(dx+1)/2);
                    color[1] = (255*(dy+1)/2);
                    color[2] = (255*(dz+1)/2);
                    IM.at<Vec3b>(Point(i,j)) = color;
                }
            //imwrite( "testl.jpg", IM );

            for(int i=0;i<4; i++){
                int  ax = iw->cx[i] + (iw->origins[i].x() / iw->scaleFactor);
                int  ay = iw->cy[i] + (iw->origins[i].y() / iw->scaleFactor);

                dx = ax*X.at<double>(0)+ay*X.at<double>(1)+ X.at<double>(2);
                dy = ax*Y.at<double>(0)+ay*Y.at<double>(1)+ Y.at<double>(2);
                dz = ax*Z.at<double>(0)+ay*Z.at<double>(1)+ Z.at<double>(2);
                float norm=sqrt(dx*dx+dy*dy+dz*dz);
                dx=dx/norm;dy=dy/norm;dz=dz/norm;
                qDebug() << iw->lights[i].at(row)[0] << iw->lights[i].at(row)[1] << iw->lights[i].at(row)[2];
                qDebug() << dx << dy << dz;
                qDebug() << "Err " << sqrt((iw->lights[i].at(row)[0]-dx)*(iw->lights[i].at(row)[0]-dx)+(iw->lights[i].at(row)[1]-dy)*(iw->lights[i].at(row)[1]-dy)+(iw->lights[i].at(row)[2]-dz)*(iw->lights[i].at(row)[2]-dz));
            }
        }


    }

    ui->inteLab->setText("Done");

}


void RTITool::on_pushButton_clicked()
{
    Mat image;
    QProgressDialog pdialog("Estimating max img","",0,100,this);
    pdialog.setWindowModality(Qt::WindowModal);
    pdialog.setCancelButton(0);
    pdialog.setValue(0);
    pdialog.setWindowTitle("Progress Dialog");
    pdialog.show();

    QListWidgetItem* item;
    item = ui->listWidget->item(0);
    QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();

    image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
    maxi=image.clone();
    image.release();
    for(int row = 1; row < ui->listWidget->count(); row++)
    {
        item = ui->listWidget->item(row);
        filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();

        image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
        cv::max(image,maxi,maxi);
        image.release();
        pdialog.setValue(100*row/ui->listWidget->count());
        pdialog.update();

        QApplication::processEvents();
    }

    QString outn = ui->folderName->text() + QDir::separator()  +"maxi.png";
    imwrite( outn.toStdString(), maxi );

    cvtColor(maxi,maxi, COLOR_BGR2RGB);

    iw->imageLabel->setPixmap(QPixmap::fromImage(QImage(maxi.data,maxi.cols,maxi.rows,maxi.step,QImage::Format_RGB888)));


    pdialog.setValue(100);
}



void RTITool::on_pushButton_2_clicked()
{
    Mat image;

    QProgressDialog pdialog("Estimating min img","",0,100,this);
    pdialog.setWindowModality(Qt::WindowModal);
    pdialog.setCancelButton(0);
    pdialog.setValue(0);
    pdialog.setWindowTitle("Progress Dialog");
    pdialog.show();

    QListWidgetItem* item;
    item = ui->listWidget->item(0);
    QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
    image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
    mini=image.clone();
    image.release();
    for(int row = 1; row < ui->listWidget->count(); row++)
    {
        item = ui->listWidget->item(row);
        filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
        image = imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
        cv::min(image,mini,mini);
        image.release();
        pdialog.setValue(100*row/ui->listWidget->count());
        pdialog.update();

        QApplication::processEvents();
    }
    cvtColor(mini,mini, COLOR_BGR2RGB);
    mini = mini;

    double min, max;
    //cv::minMaxLoc(mini, &min, &max);
    cv::normalize(mini, mini, 0, 255, NORM_MINMAX, CV_8UC1);

    QString outn = ui->folderName->text() + QDir::separator()  +"mini.png";
    imwrite( outn.toStdString(), maxi );

    pdialog.setValue(100);
    iw->imageLabel->setPixmap(QPixmap::fromImage(QImage(mini.data,mini.cols,mini.rows,mini.step,QImage::Format_RGB888)));

}


void RTITool::on_saveAPA_clicked()
{
    int flag_dir=0;
    int flag_chr=1;
    int flag_lum=0;
    int flag_corr=0;
    int flag_binary = 1;

    QString file_pref = QInputDialog::getText(this, "Enter Filename with suffix", "");
    QString  filename=ui->folderName->text() + QDir::separator() + file_pref + ".aph";
    QString  binname=ui->folderName->text() + QDir::separator() + file_pref + ".apd";
    QString  chrname=ui->folderName->text() + QDir::separator() + file_pref + "_croma.tiff";
    QString  chrname2=ui->folderName->text() + QDir::separator() + file_pref + "_cromax.tiff";

    if(ui->dirInfo->currentIndex()==1) flag_dir=1;
    if(ui->dirInfo->currentIndex()==2) flag_dir=2;

    QProgressDialog pdialog("Saving appearance profile","",0,100,this);
    pdialog.setWindowModality(Qt::WindowModal);
    pdialog.setCancelButton(0);
    pdialog.setValue(0);
    pdialog.setWindowTitle("Progress Dialog");
    pdialog.show();

    if(flag_dir==2 && iw->coilix[0].size()< 2 ){
        ui->msgBox->append("ERROR: Interpolate direction first");
        return;
    }


    if(ui->cInt->isChecked()) flag_corr = 1;
    //if(ui->cBinary->isChecked()) flag_binary=1;
    //if(ui->cChr->isChecked()) flag_chr = 1;


    int ax, ay, sx,sy;
    iw->cropArea->frameGeometry().getRect(&ax,&ay,&sx,&sy);
    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    sx = (int) ((double)sx / iw->scaleFactor);
    sy = (int) ((double)sy / iw->scaleFactor);
    qDebug() << ax << " "<< ay << " "<< sx << " "<< sy << " ";

    //  if(!QDir("CROPPED_IMG").exists())
    //      QDir().mkdir("CROPPED_IMG");

    QFile file(filename);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug() << "error";
    }
    else{
        QTextStream stream( &file );

        if(ui->apType->currentIndex()==0) {
            stream << "APA LUMINANCE\n";
            if(iw->depth==0)
                stream << "LUMINANCE_TYPE UNSIGNED_CHAR\n";
            if(iw->depth==2)
                stream << "LUMINANCE_TYPE UNSIGNED_SHORT\n";
        }
        else {
            stream << "APA RGB\n";
            if(iw->depth==0)
                stream << "COLOR_TYPE UNSIGNED_CHAR\n";
            if(iw->depth==2)
                stream << "COLOR_TYPE UNSIGNED_SHORT\n";
        }

        if(flag_dir==0){
            stream << "DIR_TYPE constant\n";
        }
        else if (flag_dir==1){
            stream << "DIR_TYPE multiple\n";
        }
        else if (flag_dir==2){
            stream << "DIR_TYPE interpolated\n";
        }

        stream << "CHROMA_IMAGE ";
        if (flag_chr==1) {
            stream <<  file_pref + "_croma.tiff" << "\n";
        }
        else {
            stream <<"none" << "\n";
        }


        stream << "CHROMAX_IMAGE ";
        if (flag_chr==1) {
            stream <<  file_pref + "_cromax.tiff" << "\n";
        }
        else {
            stream <<"none" << "\n";
        }


        QListWidgetItem *item = ui->listWidget->item(0);

        if(ax > 0) {//
            stream << "IMAGE_SIZE " << sx << " " << sy <<"\n";
        }
        else
            stream << "IMAGE_SIZE " <<  iw->s.width() << " " << iw->s.height() <<"\n";
        stream << "ORIGINAL_IMAGE_SIZE " <<  iw->s.width() << " " << iw->s.height() <<"\n";
        // write crop area
        if(ax > 0) {//cropped
            stream << "CROP AREA " << ax <<  " " << ay << " " << sx << " " << sy <<"\n";
        }
        stream << "N_IMAGES " <<  ui->listWidget->count() <<"\n";;
        stream << "IMAGE NAMES\n";
        // write image names in header file
        for(int row = 0; row < ui->listWidget->count(); row++)
        {
            item = ui->listWidget->item(row);
            if(ax>0)
                stream <<  "CROPPED" << file_pref.toLatin1() << QDir::separator() << "cropped" << QString::number(row).toLatin1() << ".tif" <<"\n";
            else{
                if(flag_corr==1)
                    stream << "CORR_IMG" << QDir::separator() << "corrected" << QString::number(row).toLatin1() << ".tif" <<"\n";
                else
                    stream << " " <<  item->text().toLatin1() <<"\n";
            }
        }



        // write single directions
        if(flag_dir==0){
            stream << "LIGHT_DIRECTIONS (lx ly lz)" <<"\n";
            int ns=0;
            double lx, ly,lz;
            for(int i=0;i<4;i++)
                if(iw->lights[i].size() >0) ns++;
            if(ns>0)
                for(int i=0;i<ui->listWidget->count();i++){
                    //  stream << "IMG " << i <<"\n";
                    lx=0;ly=0;lz=0;
                    for(int j=0;j<ns;j++){
                        lx=lx+ iw->lights[j].at(i)[0]/ns;
                        ly=ly+ iw->lights[j].at(i)[1]/ns;
                        lz=lz+ iw->lights[j].at(i)[2]/ns;
                    }
                    stream <<  lx << " " << ly << " " << lz << "\n";
                }
        }
        else if (flag_dir==2){ //write coefficients for interpolated directions
            if(ax > 0) {//cropped
                qDebug() << "cropped";

                stream << "DIRECTION_COEFFICIENTS a b k0 d e k1 g h k2 (lx=ai+bj+k0 ly=di+ej+k1 lz=gi+hj+k2)\n";
                for(int i=0;i<ui->listWidget->count();i++){

                    //   stream << "IMG " << i <<"\n";
                    double k0 = iw->coilix[0].at(i)*ax + iw->coilix[1].at(i)*ay + iw->coilix[2].at(i);
                    double k1 = iw->coiliy[0].at(i)*ax + iw->coiliy[1].at(i)*ay + iw->coiliy[2].at(i);
                    double k2 = iw->coiliz[0].at(i)*ax + iw->coiliz[1].at(i)*ay + iw->coiliz[2].at(i);
                    if(iw->coilix[0].size()>0)
                        stream <<
                                  iw->coilix[0].at(i) << " " << iw->coilix[1].at(i) << " " << k0 << " " <<
                                                         iw->coiliy[0].at(i) << " " << iw->coiliy[1].at(i) << " " << k1 << " " <<
                                                         iw->coiliz[0].at(i) << " " << iw->coiliz[1].at(i) << " " << k2 << "\n";
                }
            }
            else{
                //non cropped
                stream << "DIRECTION_COEFFICIENTS a b c d e f g h i (lx=ai+bj+c ly=di+ej+f)\n";
                for(int i=0;i<ui->listWidget->count();i++){
                    //   stream << "IMG " << i <<"\n";
                    if(iw->coilix[0].size()>0)
                        stream << iw->coilix[0].at(i) << " " << iw->coilix[1].at(i) << " " << iw->coilix[2].at(i) << " " << iw->coiliy[0].at(i) << " " << iw->coiliy[1].at(i) << " " << iw->coiliy[2].at(i) << " " << iw->coiliz[0].at(i) << " " << iw->coiliz[1].at(i) << " " << iw->coiliz[2].at(i) << "\n";
                }
            }
        }

        //write multiple directions
        if(flag_dir==1){
            stream << "MULTIPLE_LIGHT_DIRECTIONS" <<"\n";
            int ns=0;
            for(int i=0;i<4;i++)
                if(iw->lights[i].size() >0) ns++;
            for(int i=0;i<ui->listWidget->count();i++){
                //  stream << "IMG " << i <<"\n";
                for(int j=0;j<ns;j++)
                    stream << iw->lights[j].at(i)[0] << " " << iw->lights[j].at(i)[1] << " " << iw->lights[j].at(i)[2] << " ";
                stream << "\n";
            }
        }

        //close header file
        file.close();

        // saved header, now saving apa data
        // for the moment binary matrix, 8 or 16 bit, plus chroma image
        // saved without memory tricks (probably needs > 4GB for large images, but no prob on my pc and fast)
        // this can be replaced by calls to ruggeros code to save data ok for his fitters


        unsigned char val;
        // open appearance profile
        if (flag_binary==1){
            ofstream outfile (binname.toLatin1(), ios::out | ios::binary);

            Mat image, gim, cim, cim2, aim, mim;


            // allocate fixed matrix (method 1)
            unsigned char* matr;
            unsigned short* matrs;
            int sizem;
            float* difcol;
            double* aim2;


            Rect roi;

            if(ax > 0) {//cropped
                roi.x=ax;
                roi.y=ay;
                roi.width=sx;
                roi.height=sy;
                if(ui->apType->currentIndex()==0){
                    aim2=new double[sy*sx*3];
                    if(!aim2) qDebug() << "error allocating";


                    for(int i=0;i<sx;i++)
                        for(int j=0;j<sy;j++){
                            for(int k=0;k<3;k++)
                                aim2[(sx*j+i)*3+k] = 0;
                        }
                }

                if(iw->depth==0){
                    matr = new unsigned char[sx*sy*ui->listWidget->count()];
                    sizem=sx*sy*ui->listWidget->count();
                }
                if(iw->depth==2){
                    matrs = new unsigned short[sx*sy*ui->listWidget->count()];
                    sizem=sx*sy*ui->listWidget->count();
                }

            }
            else
            {

                sx=iw->s.width();
                sy=iw->s.height();
                roi.x=0;
                roi.y=0;
                roi.width=sx;
                roi.height=sy;

                if(ui->apType->currentIndex()==0){
                    aim2=new double[sx*sy*3];

                    for(int i=0;i<sx;i++)
                        for(int j=0;j<sy;j++){
                            for(int k=0;k<3;k++)
                                aim2[(sx*j+i)*3+k] = 0;
                        }
                }

                if(iw->depth==0){
                    matr = new unsigned char[iw->s.width()*iw->s.height()*ui->listWidget->count()];
                    sizem= iw->s.width()*iw->s.height()*ui->listWidget->count();
                }
                if(iw->depth==2) {
                    matrs = new unsigned short[iw->s.width()*iw->s.height()*ui->listWidget->count()];
                    sizem=iw->s.width()*iw->s.height()*ui->listWidget->count();
                }
            }

            unsigned char* ptr;
            unsigned short* ptr16;

            if(ui->apType->currentIndex()==0) {

                Mat crim2(sy,sx,CV_8UC3,cv::Scalar(0,0,0));

                // loop over images
                for(int row = 0; row < ui->listWidget->count(); row++)
                {
                    pdialog.setValue(100*row/ui->listWidget->count());
                    pdialog.update();

                    item = ui->listWidget->item(row);
                    QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();

                    QString cname;

                    if(flag_corr==0){
                        if(!iw->force8bit)
                            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
                        else
                            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
                    }
                    //  image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);
                    else {
                        cname =  ui->folderName->text() + QDir::separator()  + "CORR_IMG" + QDir::separator()  + "corrected" + QString::number(row) +".tif";
                        if(!iw->force8bit)
                            image = cv::imread(cname.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
                        else
                            image = cv::imread(cname.toStdString(), CV_LOAD_IMAGE_COLOR);
                    }

                    if(! image.data )                              // Check for invalid input
                    {
                        ui->msgBox->append("Error: no corrected images available");
                        qDebug() << cname;
                        return;
                    }


                    cim = image(roi);

                    QString cropname =  ui->folderName->text() + QDir::separator()  + QString("CROPPED") + file_pref + QDir::separator() + QString("cropped") + QString::number(row) + ".tif";

                    if(!QDir( ui->folderName->text() + QDir::separator()  + QString("CROPPED") + file_pref).exists())
                        QDir().mkdir( ui->folderName->text() + QDir::separator()  + QString("CROPPED") + file_pref);

                    if(ax > 0)
                        imwrite(cropname.toStdString(),cim);

                    if(iw->depth==0)
                        cv::max(crim2, cim, crim2);

                    if(iw->depth==2){
                        cim.convertTo(cim2, CV_8UC3, 0.00390625);
                        cv::max(crim2, cim2, crim2);
                    }


                    if(iw->depth==0){
                        for(int i=0;i<sx;i++)
                            for(int j=0;j<sy;j++){
                                matr[sx*ui->listWidget->count()*j+ui->listWidget->count()*i+row]=(unsigned char)(0.2126*cim.at<Vec3b>(Point(i,j))[2] + 0.7152*cim.at<Vec3b>(Point(i,j))[1]  + 0.0722*cim.at<Vec3b>(Point(i,j))[0]);

                                for(int k=0;k<3;k++)
                                    aim2[(sx*j+i)*3+k] = aim2[(sx*j+i)*3+k] + (double)cim.at<Vec3b>(Point(i,j))[k];
                            }
                    }


                    if(iw->depth==2){
                        unsigned short* ptr=(unsigned short*) (cim.data);

                        for(int i=0;i<sx;i++)
                            for(int j=0;j<sy;j++){
                                matrs[sx*ui->listWidget->count()*j+ui->listWidget->count()*i+row]=(unsigned short)(0.2126*cim.at<Vec<unsigned short, 3> >(Point(i,j))[2] + 0.7152*cim.at<Vec<unsigned short, 3> >(Point(i,j))[1]  + 0.0722*cim.at<Vec<unsigned short, 3> >(Point(i,j))[0]);


                                for(int k=0;k<3;k++)
                                    aim2[(sx*j+i)*3+k] = aim2[(sx*j+i)*3+k] + (double)cim.at<Vec<unsigned short, 3> >(Point(i,j))[k];

                            }
                    }


                    if(ax>0)
                        cim.release();

                    image.release();
                }

                /* write data and chroma image */
                if(iw->depth==0){

                    outfile.write((char*)&matr[0],sizem*sizeof(unsigned char));
                    delete matr;
                    outfile.close();

                    Mat crim(sy,sx,CV_8UC3);


                    Vec3f val;
                    for(int i=0;i<sx;i++)
                        for(int j=0; j<sy;j++){

                            for(int k=0;k<3;k++)
                                val.val[k]=aim2[(sx*j+i)*3+k]/(float)ui->listWidget->count();

                            float sum=val.val[0]+val.val[1]+val.val[2];
                            if(sum>0)
                                val = val;//255*val/sum;
                            else
                                val=0;

                            crim.at<Vec3b>(j,i) = val;
                        }

                    imwrite(chrname.toStdString(),crim);

                    imwrite(chrname2.toStdString(),crim2);


                }

                if(iw->depth==2){  // 16 bit
                    outfile.write((char*)&matrs[0],sizem*sizeof(unsigned short));
                    delete matrs;
                    outfile.close();
                    Mat crim(sy,sx,CV_16UC3);
                    qDebug() << "qui " << sx << " " << sy;

                    Vec3f val;
                    int i,j, k;
                    double sum;

                    for(i=0;i<sx;i++)
                        for(j=0; j<sy;j++){

                            for(int k=0;k<3;k++)
                                val.val[k]=aim2[(sx*j+i)*3+k]/(float)ui->listWidget->count();

                            sum= val.val[0]+val.val[1]+val.val[2];
                            if(sum > 0)
                                val = val;//65535*(val/sum);
                            else
                                val=0;

                            crim.at<Vec<unsigned short, 3> >(j,i) = val;

                        }

                    imwrite(chrname.toStdString(),crim);
                    imwrite(chrname2.toStdString(),crim2);
                }

                delete aim2;
            }
            else{
                // RGB
                for(int cc=0; cc<3;cc++){

                    // loop over images
                    for(int row = 0; row < ui->listWidget->count(); row++)
                    {
                        pdialog.setValue(100*(row+ui->listWidget->count()*cc)/(3*ui->listWidget->count()));
                        pdialog.update();
                        item = ui->listWidget->item(row);
                        QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
                        if(flag_corr==0){
                            if(!iw->force8bit)
                                image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
                            else
                                image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
                        }
                        //  image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);
                        else {
                            QString cname=ui->folderName->text() + QDir::separator()  +"CORR_IMG/corrected" + QString::number(row) +".tif";
                            if(!iw->force8bit)
                                image = cv::imread(cname.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
                            else
                                image = cv::imread(cname.toStdString(), CV_LOAD_IMAGE_COLOR);
                        }

                        if(! image.data )                              // Check for invalid input
                        {
                            ui->msgBox->append("Error: no corrected images available");
                            return;
                        }


                        cim = image(roi);
                        QString cropname = ui->folderName->text() + QDir::separator()  + QString("CROPPED") + file_pref + QString("/cropped") + QString::number(row) + ".tif";

                        if(!QDir(QString(ui->folderName->text() + QDir::separator()  +"CROPPED") + file_pref).exists())
                            QDir().mkdir(QString(ui->folderName->text() + QDir::separator()  +"CROPPED") + file_pref);

                        if(ax > 0)
                            imwrite(cropname.toStdString(),cim);


                        if(iw->depth==0){
                            for(int i=0;i<sx;i++)
                                for(int j=0;j<sy;j++){
                                    matr[sx*ui->listWidget->count()*j+ui->listWidget->count()*i+row]=(unsigned char)cim.at<Vec3b>(Point(i,j))[2-cc];
                                }
                        }


                        if(iw->depth==2){
                            unsigned short* ptr=(unsigned short*) (cim.data);

                            for(int i=0;i<sx;i++)
                                for(int j=0;j<sy;j++){
                                    matrs[sx*ui->listWidget->count()*j+ui->listWidget->count()*i+row]=(unsigned short)(cim.at<Vec<unsigned short, 3> >(Point(i,j))[2-cc]);

                                }
                        }


                        if(ax>0)
                            cim.release();

                        image.release();
                    }

                    /* write data */
                    if(iw->depth==0){

                        outfile.write((char*)&matr[0],sizem*sizeof(unsigned char));
                        // delete matr;
                        //  outfile.close();


                    }

                    if(iw->depth==2){
                        outfile.write((char*)&matrs[0],sizem*sizeof(unsigned short));
                        //delete matrs;
                        //outfile.close();


                    }
                }



            }

        }
    }




}

void RTITool::on_action_lp_file_triggered()
{
    /*
    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open image list"));

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    qDebug(fileName.toLatin1());
    QStringList  nList;
    int nimg;
    int nsph;

    QTextStream textStream(&file);


    QString line = textStream.readLine();
    if (line.isNull())
        return;
    QStringList parts = line.split(" ");
    nimg= parts.at(0).toInt();
    nsph= parts.at(1).toInt();

    int siz=parts.size();

    qDebug() << "ff" << siz;

    float radii[nsph];
    float centx[nsph];
    float centy[nsph];
    for(int i=0;i<nsph;i++){
        radii[i] = parts.at(2+i*3).toFloat();
        centx[i] = parts.at(3+i*3).toFloat();
        centy[i] = parts.at(4+i*3).toFloat();
        qDebug() << centx[i] << centy[i] <<radii[nsph];

    }
    parts.clear();


    ui->listWidget->clear();

    for(int nsph=0;nsph<4;nsph++)
        iw->lights[nsph].clear();


    for(int i=0;i<nimg;i++){
        line = textStream.readLine();
        if (line.isNull())
            break;
        QStringList   parts = line.split(" ");
        qDebug() << line;
        qDebug() << i << " " <<  parts[0];
        ui->listWidget->addItem( parts[0] );

        for(int j=0;j<nsph;j++){
            double* vec=new double[3];

            vec[0] = parts[j*3+1].toDouble();
            vec[1] = parts[j*3+2].toDouble();
            vec[2] = parts[j*3+3].toDouble();
            float norm=sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
            vec[0] = vec[0]/norm;
            vec[1] = vec[1]/norm;
            vec[2] = vec[2]/norm;

            iw->lights[j].push_back(vec);
            qDebug() << "l " << (iw->lights[j].at(i))[0] << (iw->lights[j].at(i))[1] << (iw->lights[j].at(i))[2];
        }

    }

    file.close();

    iw->load(ui->listWidget->item(0)->text());
    iw->show();

    for(int i=0;i<nsph;i++){

        iw->radius[i] = radii[i];
        iw->origins[i].setX((centx[i]-radii[i])*iw->scaleFactor);
        iw->origins[i].setY((centy[i]-radii[i])*iw->scaleFactor);
        iw->cx[i] = centx[i];
        iw->cy[i] = centy[i];
        iw->ls[i]->setGeometry(iw->origins[i].x(),iw->origins[i].y(),radii[i]*2*iw->scaleFactor,radii[i]*2*iw->scaleFactor);

        if(i==0){
            ui->cx1spin->setValue(centx[i]);
            ui->cy1spin->setValue(centy[i]);
            ui->r1spin->setValue(iw->radius[i]);
            ui->cx1spin->setVisible(true);
            ui->cy1spin->setVisible(true);
            ui->r1spin->setVisible(true);
            iw->sphere1->setGeometry(iw->origins[i].x(),iw->origins[i].y(),radii[i]*2*iw->scaleFactor,radii[i]*2*iw->scaleFactor);
            iw->sphere1->setVisible(true);
        }
        if(i==1){
            ui->cx2spin->setValue(centx[i]);
            ui->cy2spin->setValue(centy[i]);
            ui->r2spin->setValue(iw->radius[i]);
            ui->cx2spin->setVisible(true);
            ui->cy2spin->setVisible(true);
            ui->r2spin->setVisible(true);
            iw->sphere2->setGeometry(iw->origins[i].x(),iw->origins[i].y(),radii[i]*2*iw->scaleFactor,radii[i]*2*iw->scaleFactor);
            iw->sphere2->setVisible(true);
        }
        if(i==2){
            ui->cx3spin->setValue(centx[i]);
            ui->cy3spin->setValue(centy[i]);
            ui->r3spin->setValue(iw->radius[i]);
            ui->cx3spin->setVisible(true);
            ui->cy3spin->setVisible(true);
            ui->r3spin->setVisible(true);
            iw->sphere3->setGeometry(iw->origins[i].x(),iw->origins[i].y(),radii[i]*2*iw->scaleFactor,radii[i]*2*iw->scaleFactor);
            iw->sphere3->setVisible(true);
        }

        if(i==3){
            ui->cx4spin->setValue(centx[i]);
            ui->cy4spin->setValue(centy[i]);
            ui->r4spin->setValue(iw->radius[i]);
            ui->cx4spin->setVisible(true);
            ui->cy4spin->setVisible(true);
            ui->r4spin->setVisible(true);
            iw->sphere4->setGeometry(iw->origins[i].x(),iw->origins[i].y(),radii[i]*2*iw->scaleFactor,radii[i]*2*iw->scaleFactor);
            iw->sphere4->setVisible(true);
        }


        qDebug() << iw->origins[i].x() <<  "or " << iw->origins[i].y() << iw->radius[i];

        QPixmap pixmap((int)((float)(2*iw->radius[i])*iw->scaleFactor+1),(int)((float)(2*iw->radius[i])*iw->scaleFactor)+1);
        pixmap.fill(QColor("transparent"));
        iw->ls[i]->setScaledContents(true);;;
        QPainter painter(&pixmap);
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->radius[i]*iw->scaleFactor,iw->radius[i]*iw->scaleFactor),(iw->radius[i]*iw->scaleFactor),(iw->radius[i]*iw->scaleFactor));
        iw->ls[i]->setPixmap(pixmap);
        iw->ls[i]->setVisible(true);

    }
    // qDebug(nList);
    QApplication::processEvents();
*/
}

void RTITool::on_cropBut_clicked()
{
    iw->active=10;

}

void RTITool::on_remCrop_clicked()
{
    iw->cropArea->setGeometry(QRect(0,0,0,0));
    iw->cropArea->hide();
}

void RTITool::on_box8Bit_clicked(bool checked)
{

}

void RTITool::on_box8Bit_stateChanged(int state)
{
    if(state)
    {
        iw->force8bit=true;
    }
    else
    {
        iw->force8bit=false;
    }
}

void RTITool::on_box8Bit_toggled(bool checked)
{

}


void RTITool::loadList(QString fileName)
{
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    // QFileInfo fi = QFileInfo(file);
    // qDebug(fileName.toLatin1());
    QStringList  nList;

    int last= fileName.lastIndexOf(QDir::separator());
    QString folder=fileName.left(last+1);

    QProgressDialog progress("Importing files please wait...", "", 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);
    progress.setCancelButton(0);
    progress.setWindowTitle("Progress Dialog");
    progress.show( );

    int line_count=0;


    QTextStream textStream(&file);

    while( !textStream.atEnd())
    {
       textStream.readLine();
        line_count++;
    }

    textStream.seek(0);
    int ni=0;
    while (true)
    {
        ni++;
        QString line = textStream.readLine();
        last= line.lastIndexOf(QDir::separator());
        QString lastname=line.right(line.size()-last-1);
        lastname.replace(" ","_");

        progress.setValue(ni*100/line_count);

        qDebug(line.toLatin1());
        if (line.isNull())
            break;
        else{
            nList.append(lastname);
            QString dstImg=ui->folderName->text() + QDir::separator() + "images"+ QDir::separator() + lastname;

            QFile::copy(line,dstImg);
        }
    }
    file.close();
    ui->listWidget->clear();
    ui->listWidget->addItems( nList );

    iw->load(ui->folderName->text() + QDir::separator()  + "images" + QDir::separator()  + ui->listWidget->item(0)->text());
    iw->show();

    QString msg = "Loaded list file with " + QString::number(ui->listWidget->count()) + " images";
    ui->msgBox->setText(msg);
    if(iw->depth==0) ui->msgBox->append("8 bit depth");
    if(iw->depth==2) ui->msgBox->append("16 bit depth");


    QApplication::processEvents();

}

void RTITool::on_loadListButton_clicked()
{

    QString fileName;
    fileName = getFilename();
    if(!fileName.isEmpty())
        loadList(fileName);

    QString destName = ui->folderName->text() + QDir::separator() + "list.txt";

    if (!QFile::copy(fileName, destName))
        return;

}

void RTITool::loadLp(QString fileName)
{

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QString folder = ui->folderName->text();
    QFile outfile(folder + "file.lp");


    qDebug(fileName.toLatin1());
    QStringList  nList;
    int nimg;
    int nsph;

    QTextStream textStream(&file);

    QString line = textStream.readLine();

    if (line.isNull())
        return;
    QStringList parts = line.split(" ");

    int siz=parts.size();

    if(siz <3){
        ui->msgBox->append("error: missing data");
        return;
    }

    nimg= parts.at(0).toInt();
    nsph= parts.at(1).toInt();

    int type = 0;
    if(siz>2)
        type=parts.at(2).toInt();

    float radii[4];
    float centx[4];
    float centy[4];
    float ry[4];
    float angle[4];

    for(int i=0;i<nsph;i++){
        radii[i]=0;
        centx[i]=0;
        centy[i]=0;
        ry[i]=0;
        angle[i]=0;
    }

    if(type==1 && siz >nsph*3+1)
    {

        for(int i=0;i<nsph;i++){
            radii[i] = parts.at(3+i*3).toFloat();
            centx[i] = parts.at(4+i*3).toFloat();
            centy[i] = parts.at(5+i*3).toFloat();
        }
    }
    if(type==2 && siz >nsph*5+1)
    {

        for(int i=0;i<nsph;i++){
            radii[i] = parts.at(4+i*5).toFloat();
            ry[i] = parts.at(5+i*5).toFloat();
            centx[i] = parts.at(6+i*5).toFloat();
            centy[i] = parts.at(7+i*5).toFloat();
            angle[i] = parts.at(3+i*5).toFloat();
        }
    }

    parts.clear();

    ui->listWidget->clear();

    for(int i=0;i<4;i++)
        iw->lights[i].clear();


    for(int i=0;i<nimg;i++){
        line = textStream.readLine();
        if (line.isNull())
            break;
        QStringList   parts = line.split(" ");
        // qDebug() << line;
        // qDebug() << i << " " <<  parts[0];
        ui->listWidget->addItem( parts[0] );
        if(type==0){
            double* vec=new double[3];

            vec[0] = parts[1].toDouble();
            vec[1] = parts[2].toDouble();
            vec[2] = parts[3].toDouble();
            float norm=sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
            vec[0] = vec[0]/norm;
            vec[1] = vec[1]/norm;
            vec[2] = vec[2]/norm;

            iw->lights[0].push_back(vec);
        }
        else
            for(int j=0;j<nsph;j++){
                double* vec=new double[3];

                vec[0] = parts[j*3+1].toDouble();
                vec[1] = parts[j*3+2].toDouble();
                vec[2] = parts[j*3+3].toDouble();
                float norm=sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
                vec[0] = vec[0]/norm;
                vec[1] = vec[1]/norm;
                vec[2] = vec[2]/norm;

                iw->lights[j].push_back(vec);
                // qDebug() << "l " << (iw->lights[j].at(i))[0] << (iw->lights[j].at(i))[1] << (iw->lights[j].at(i))[2];
            }

    }

    file.close();


    iw->load(ui->folderName->text() + QDir::separator()  +"images"  + QDir::separator() + ui->listWidget->item(0)->text());
    iw->show();

    if(type==1){
        ui->projBox->setChecked(false);
    }
    else if(type==2){
        ui->projBox->setChecked(true);
    }

    if(type==1 || type==2)
        for(int i=0;i<nsph;i++){


            iw->origins[i].setX((centx[i]-max(ry[i],radii[i]))*iw->scaleFactor);
            iw->origins[i].setY((centy[i]-max(ry[i],radii[i]))*iw->scaleFactor);
            iw->ends[i].setX((centx[i]+max(ry[i],radii[i]))*iw->scaleFactor);
            iw->ends[i].setY((centy[i]+max(ry[i],radii[i]))*iw->scaleFactor);
            iw->ls[i]->setGeometry(QRect(iw->origins[i],iw->ends[i]));




            if(i==0){
                ui->cx1spin->setValue(centx[i]);
                ui->cy1spin->setValue(centy[i]);
                ui->r1spin->setValue(radii[i]);
                ui->cx1spin->setVisible(true);
                ui->cy1spin->setVisible(true);
                ui->r1spin->setVisible(true);

                if(type==2){

                    ui->r2_1Spin->setValue(ry[i]);
                    ui->angle1spin->setValue(angle[i]);
                    ui->r2_1Spin->setVisible(true);
                    ui->angle1spin->setVisible(true);
                }
                iw->sphere1->setGeometry(iw->origins[i].x(),iw->origins[i].y(),max(ry[i],radii[i])*2*iw->scaleFactor,max(ry[i],radii[i])*2*iw->scaleFactor);
                iw->sphere1->setVisible(true);
            }
            if(i==1){
                ui->cx2spin->setValue(centx[i]);
                ui->cy2spin->setValue(centy[i]);
                ui->r2spin->setValue(radii[i]);
                ui->cx2spin->setVisible(true);
                ui->cy2spin->setVisible(true);
                ui->r2spin->setVisible(true);
                iw->sphere2->setGeometry(iw->origins[i].x(),iw->origins[i].y(),max(ry[i],radii[i])*2*iw->scaleFactor,max(ry[i],radii[i])*2*iw->scaleFactor);
                iw->sphere2->setVisible(true);
                if(type==2){
                    ui->r2_2Spin->setValue(ry[i]);
                    ui->angle2spin->setValue(angle[i]);
                    ui->r2_2Spin->setVisible(true);
                    ui->angle2spin->setVisible(true);
                }
            }
            if(i==2){
                ui->cx3spin->setValue(centx[i]);
                ui->cy3spin->setValue(centy[i]);
                ui->r3spin->setValue(radii[i]);
                ui->cx3spin->setVisible(true);
                ui->cy3spin->setVisible(true);
                ui->r3spin->setVisible(true);
                iw->sphere3->setGeometry(iw->origins[i].x(),iw->origins[i].y(),max(ry[i],radii[i])*2*iw->scaleFactor,max(ry[i],radii[i])*2*iw->scaleFactor);
                iw->sphere3->setVisible(true);
                if(type==2){
                    ui->r2_3Spin->setValue(ry[i]);
                    ui->angle3spin->setValue(angle[i]);
                    ui->r2_3Spin->setVisible(true);
                    ui->angle3spin->setVisible(true);
                }
            }

            if(i==3){
                ui->cx4spin->setValue(centx[i]);
                ui->cy4spin->setValue(centy[i]);
                ui->r4spin->setValue(radii[i]);
                ui->cx4spin->setVisible(true);
                ui->cy4spin->setVisible(true);
                ui->r4spin->setVisible(true);
                iw->sphere4->setGeometry(iw->origins[i].x(),iw->origins[i].y(),max(ry[i],radii[i])*2*iw->scaleFactor,max(ry[i],radii[i])*2*iw->scaleFactor);
                iw->sphere4->setVisible(true);
                if(type==2){
                    ui->r2_4Spin->setValue(ry[i]);
                    ui->angle4spin->setValue(angle[i]);
                    ui->r2_4Spin->setVisible(true);
                    ui->angle4spin->setVisible(true);
                }
            }

            iw->radius[i] = radii[i];
            iw->cx[i] = max(ry[i],radii[i]);
            iw->cy[i] = max(ry[i],radii[i]);
            iw->rx[i]=radii[i];
            iw->ry[i]=ry[i];


            QPixmap pixmap((int)((float)(2*max(ry[i],radii[i]))*iw->scaleFactor+1),(int)((float)(2*max(ry[i],radii[i]))*iw->scaleFactor)+1);
            pixmap.fill(QColor("transparent"));
            // iw->ls[i]->setScaledContents(true);;;
            QPainter painter(&pixmap);

            // qDebug() << iw->origins[i].x() <<  "or " << iw->origins[i].y() << iw->radius[i];
            if(ui->projBox->isChecked())
            {
                painter.setPen(QPen(Qt::green));
                painter.save();
                QTransform trans;
                trans.translate((float)(max(ry[i],radii[i]))*iw->scaleFactor,(float)(max(ry[i],radii[i]))*iw->scaleFactor);
                trans.rotate(iw->angle[i]);
                painter.setTransform(trans);
                painter.drawEllipse(QPointF(0,0),(iw->rx[i]*iw->scaleFactor),(iw->ry[i]*iw->scaleFactor));
                painter.restore();
            }
            else
            {
                painter.setPen(QPen(Qt::red));
                painter.drawEllipse(QPointF(iw->radius[i]*iw->scaleFactor,iw->radius[i]*iw->scaleFactor),(iw->radius[i]*iw->scaleFactor),(iw->radius[i]*iw->scaleFactor));
            }
            iw->ls[i]->setPixmap(pixmap);
            iw->ls[i]->setVisible(true);

            /*        QPixmap pixmap((int)((float)(2*iw->radius[i])*iw->scaleFactor+1),(int)((float)(2*iw->radius[i])*iw->scaleFactor)+1);
        pixmap.fill(QColor("transparent"));
        iw->ls[i]->setScaledContents(true);;;
        QPainter painter(&pixmap);
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->radius[i]*iw->scaleFactor,iw->radius[i]*iw->scaleFactor),(iw->radius[i]*iw->scaleFactor),(iw->radius[i]*iw->scaleFactor));
        iw->ls[i]->setPixmap(pixmap);
        iw->ls[i]->setVisible(true);*/

        }

    QString msg = "Loaded lp file with " + QString::number(ui->listWidget->count()) + " images";
    ui->msgBox->setText(msg);
    if(type==0)
        ui->direLab->setText(QString("Ok"));
    else
        ui->direLab->setText(QString("Ok "+  QString::number(nsph)));
    if(iw->depth==0) ui->msgBox->append("8 bit depth");
    if(iw->depth==2) ui->msgBox->append("16 bit depth");

    if(type==0)
        msg = "and precomputed light directions";
    else
        msg = "and " + QString::number(nsph) + " light directions";


    ui->msgBox->append(msg);
    QApplication::processEvents();
    //qDebug() << ">>>" << iw->radius[0] << " " << iw->cx[0];

    if(nsph>2) on_interpDir_clicked();

    outfile.close();

}



void RTITool::loadDirFromLp(QString fileName)
{

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QFile outfile(ui->folderName->text() + QDir::separator() + "file.lp");

    QString folder = ui->folderName->text();


    qDebug(fileName.toLatin1());
    QStringList  nList;
    int nimg;
    int nsph;

    QTextStream textStream(&file);

    QString line = textStream.readLine();

    if (line.isNull())
        return;
    QStringList parts = line.split(" ");

    int siz=parts.size();
qDebug() << siz << "!!!";

    if(siz <1){
        ui->msgBox->append("error: missing data");
        return;
    }

    if(siz <2){
        nsph=1;
    }
    else
         nsph= parts.at(1).toInt();

    nimg= parts.at(0).toInt();

qDebug()<< "NS " << nsph;
    int type = 0;
    if(siz>2)
        type=parts.at(2).toInt();

    float radii[4];
    float centx[4];
    float centy[4];
    float ry[4];
    float angle[4];

    for(int i=0;i<nsph;i++){
        radii[i]=0;
        centx[i]=0;
        centy[i]=0;
        ry[i]=0;
        angle[i]=0;
    }

    if(type==1 && siz >nsph*3+1)
    {

        for(int i=0;i<nsph;i++){
            radii[i] = parts.at(3+i*3).toFloat();
            centx[i] = parts.at(4+i*3).toFloat();
            centy[i] = parts.at(5+i*3).toFloat();
        }
    }
    if(type==2 && siz >nsph*5+1)
    {

        for(int i=0;i<nsph;i++){
            radii[i] = parts.at(4+i*5).toFloat();
            ry[i] = parts.at(5+i*5).toFloat();
            centx[i] = parts.at(6+i*5).toFloat();
            centy[i] = parts.at(7+i*5).toFloat();
            angle[i] = parts.at(3+i*5).toFloat();
        }
    }

    parts.clear();

   // ui->listWidget->clear();

    for(int i=0;i<4;i++)
        iw->lights[i].clear();


    for(int i=0;i<nimg;i++){
        line = textStream.readLine();
        if (line.isNull())
            break;
        QStringList   parts = line.split(" ");
        // qDebug() << line;
        // qDebug() << i << " " <<  parts[0];
  //      ui->listWidget->addItem( parts[0] );
        if(type==0){
            double* vec=new double[3];

            vec[0] = parts[1].toDouble();
            vec[1] = parts[2].toDouble();
            vec[2] = parts[3].toDouble();
            float norm=sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
            vec[0] = vec[0]/norm;
            vec[1] = vec[1]/norm;
            vec[2] = vec[2]/norm;

            iw->lights[0].push_back(vec);
        }
        else
            for(int j=0;j<nsph;j++){
                double* vec=new double[3];

                vec[0] = parts[j*3+1].toDouble();
                vec[1] = parts[j*3+2].toDouble();
                vec[2] = parts[j*3+3].toDouble();
                float norm=sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
                vec[0] = vec[0]/norm;
                vec[1] = vec[1]/norm;
                vec[2] = vec[2]/norm;

                iw->lights[j].push_back(vec);
                 qDebug() << "l " << (iw->lights[j].at(i))[0] << (iw->lights[j].at(i))[1] << (iw->lights[j].at(i))[2];
            }

    }

    file.close();


    if(type==1){
        ui->projBox->setChecked(false);
    }
    else if(type==2){
        ui->projBox->setChecked(true);
    }

    if(type==1 || type==2)
        for(int i=0;i<nsph;i++){


            iw->origins[i].setX((centx[i]-max(ry[i],radii[i]))*iw->scaleFactor);
            iw->origins[i].setY((centy[i]-max(ry[i],radii[i]))*iw->scaleFactor);
            iw->ends[i].setX((centx[i]+max(ry[i],radii[i]))*iw->scaleFactor);
            iw->ends[i].setY((centy[i]+max(ry[i],radii[i]))*iw->scaleFactor);
            iw->ls[i]->setGeometry(QRect(iw->origins[i],iw->ends[i]));




            if(i==0){
                ui->cx1spin->setValue(centx[i]);
                ui->cy1spin->setValue(centy[i]);
                //ui->r1spin->setValue(radii[i]);
               // ui->cx1spin->setVisible(true);
               // ui->cy1spin->setVisible(true);
              //  ui->r1spin->setVisible(true);

                if(type==2){

                    ui->r2_1Spin->setValue(ry[i]);
                    ui->angle1spin->setValue(angle[i]);
                  //  ui->r2_1Spin->setVisible(true);
                  //  ui->angle1spin->setVisible(true);
                }
                iw->sphere1->setGeometry(iw->origins[i].x(),iw->origins[i].y(),max(ry[i],radii[i])*2*iw->scaleFactor,max(ry[i],radii[i])*2*iw->scaleFactor);
                //iw->sphere1->setVisible(true);
            }
            if(i==1){
                ui->cx2spin->setValue(centx[i]);
                ui->cy2spin->setValue(centy[i]);
                ui->r2spin->setValue(radii[i]);
                //ui->cx2spin->setVisible(true);
                //ui->cy2spin->setVisible(true);
                //ui->r2spin->setVisible(true);
                iw->sphere2->setGeometry(iw->origins[i].x(),iw->origins[i].y(),max(ry[i],radii[i])*2*iw->scaleFactor,max(ry[i],radii[i])*2*iw->scaleFactor);
                //iw->sphere2->setVisible(true);
                if(type==2){
                    ui->r2_2Spin->setValue(ry[i]);
                    ui->angle2spin->setValue(angle[i]);
                   // ui->r2_2Spin->setVisible(true);
                  //  ui->angle2spin->setVisible(true);
                }
            }
            if(i==2){
                ui->cx3spin->setValue(centx[i]);
                ui->cy3spin->setValue(centy[i]);
                ui->r3spin->setValue(radii[i]);
                //ui->cx3spin->setVisible(true);
                //ui->cy3spin->setVisible(true);
                //ui->r3spin->setVisible(true);
                iw->sphere3->setGeometry(iw->origins[i].x(),iw->origins[i].y(),max(ry[i],radii[i])*2*iw->scaleFactor,max(ry[i],radii[i])*2*iw->scaleFactor);
                //iw->sphere3->setVisible(true);
                if(type==2){
                    ui->r2_3Spin->setValue(ry[i]);
                    ui->angle3spin->setValue(angle[i]);
             //       ui->r2_3Spin->setVisible(true);
              //      ui->angle3spin->setVisible(true);
                }
            }

            if(i==3){
                ui->cx4spin->setValue(centx[i]);
                ui->cy4spin->setValue(centy[i]);
                ui->r4spin->setValue(radii[i]);
           //     ui->cx4spin->setVisible(true);
             //   ui->cy4spin->setVisible(true);
            //    ui->r4spin->setVisible(true);
                iw->sphere4->setGeometry(iw->origins[i].x(),iw->origins[i].y(),max(ry[i],radii[i])*2*iw->scaleFactor,max(ry[i],radii[i])*2*iw->scaleFactor);
           //     iw->sphere4->setVisible(true);
                if(type==2){
                    ui->r2_4Spin->setValue(ry[i]);
                    ui->angle4spin->setValue(angle[i]);
         //           ui->r2_4Spin->setVisible(true);
         //           ui->angle4spin->setVisible(true);
                }
            }

            iw->radius[i] = radii[i];
            iw->cx[i] = max(ry[i],radii[i]);
            iw->cy[i] = max(ry[i],radii[i]);
            iw->rx[i]=radii[i];
            iw->ry[i]=ry[i];
}

    QApplication::processEvents();
    //qDebug() << ">>>" << iw->radius[0] << " " << iw->cx[0];

    if(nsph>2) on_interpDir_clicked();

}

void RTITool::on_loadLpButton_clicked()
{
    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open lp file"));

    loadLp(fileName);

}

void RTITool::on_removeAmbientButton_clicked(){

    QMessageBox messageBox;
    //load image with ambient lighting.
    QString ambientImageFileName;

    ambientImageFileName = QFileDialog::getOpenFileName(this,
                                                        tr("Load image with ambient lighting"));


    cv::Mat imageAmbient = imread(ambientImageFileName.toStdString(),CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_COLOR);
    cv::Mat image;



    //subtract ambient from each image in the list.
    for(int row = 0; row < ui->listWidget->count(); row++)
    {
        //load image from the list
        QListWidgetItem *item = ui->listWidget->item(row);
        QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
        std::string originalImageName = filen.toStdString();
        //  std::string originalImageName = item->text().toStdString();
        image = imread(originalImageName,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_COLOR);

        qDebug() << "----- Subtracting ambient from image - " << QString::fromStdString(originalImageName) << "...";

        int width  = image.cols;
        int height = image.rows;

        if (!((width == imageAmbient.cols) && (height == imageAmbient.rows))) {
            qDebug()<<"error";
            messageBox.critical(0,"Error","Original image and ambient image differ in size!");
            messageBox.setFixedSize(500,200);
        }

        int type = imageAmbient.type();
        switch (type) {
        case CV_16UC3: {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    cv::Vec<u_int16_t,3> pixel   = image.at<cv::Vec<u_int16_t,3> >(i,j);

                    cv::Vec<u_int16_t,3> ambient = imageAmbient.at<cv::Vec<u_int16_t,3> >(i,j);
                    for (int k = 0; k < 3; ++k) {
                        (pixel[k] > ambient[k]) ? pixel[k] = pixel[k]-ambient[k] : pixel[k] = 0;
                    }

                    image.at<cv::Vec<u_int16_t,3> >(i,j) = pixel;
                }
            }

            int last= filen.lastIndexOf(".");
            QString common=filen.left(last);
            QString cn = common + "_af.tiff";
            std::string   imageCorrectedName = cn.toStdString();
            qDebug()<< "----- Writing ambient free image - " << QString::fromStdString(imageCorrectedName) << "\n";
            cv::imwrite(imageCorrectedName,image);
            //change the item name in the list
            ui->listWidget->item(row) ->setText(QString::fromStdString(imageCorrectedName));

        } break;
        default: {
            qDebug() << "Unknown image type " << type << " (The only supported image type is CV_16UC3)" << "\n";
        } break;
        }

        image.release();

    }
    //reload the ambient free images
    iw->load(ui->listWidget->item(0)->text());
    iw->show();

    QString msg = "Reloaded the image list with ambient removed!";
    ui->msgBox->setText(msg);
    if(iw->depth==0) ui->msgBox->append("8 bit depth");
    if(iw->depth==2) ui->msgBox->append("16 bit depth");


    QFile lfile(ui->folderName->text() + QDir::separator() + "list.txt");

    if (!lfile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "error";
    }
    else{
        QTextStream out(&lfile);
        for(int row = 0; row < ui->listWidget->count(); row++)
        {
            out << ui->listWidget->item(row)->text() << endl;;
        }
    }



    QApplication::processEvents();

}

void RTITool::on_undistortImagesButton_clicked(){

    cv::Mat image;
    QMessageBox messageBox;



    if (ui->listWidget->count() == 0){
        qDebug() << "error";
        messageBox.critical(0,"Error","There is no loaded image to undistort!");
        messageBox.setFixedSize(500,200);
        return;
    }

    if (iw->cameraMatrix.at<double>(0,2) ==0 ){
        qDebug() << "error";
        messageBox.critical(0,"Error","Please load the calibration parameters!");
        messageBox.setFixedSize(500,200);
        return;
    }

    if (iw->distCoeffs.empty() ){
        qDebug() << "error";
        messageBox.critical(0,"Error","Please load the calibration parameters!");
        messageBox.setFixedSize(500,200);
        return;
    }

    QString originalImageName;
    QString undistortedImageName;

    QProgressDialog progress("Undistorting images please wait...", "", 0, ui->listWidget->count(), this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);
    progress.setCancelButton(0);
    progress.setWindowTitle("Progress Dialog");
    progress.show( );
    QCoreApplication::processEvents();

    for(int row = 0; row < ui->listWidget->count(); row++)
    {
        QListWidgetItem *item = ui->listWidget->item(row);
        originalImageName = item->text();
        
        progress.setValue(row);

        int last= originalImageName.lastIndexOf(".");
        QString common=originalImageName.left(last);

        undistortedImageName = common + "_und.tiff";

        QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
        cv::Mat image = imread(filen.toStdString(),CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_COLOR);
        cv::Mat image_und;
        qDebug() << "----- Undistorting image - " << filen << "...";
        cv::undistort(image,image_und,iw->cameraMatrix,iw->distCoeffs);
        qDebug() <<"\nDistance coeff"<<iw->distCoeffs.at<double>(0,0)<< iw->distCoeffs.at<double>(1,0)<<"  "<<iw->distCoeffs.at<double>(2,0)<<"  "<<iw->distCoeffs.at<double>(3,0)<<"  ";

        QString outn = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + undistortedImageName;
        cv::imwrite(outn.toStdString(),image_und);
        ui->listWidget->item(row) ->setText(undistortedImageName);
        image.release();
        image_und.release();
    }
    
    //reload the undistorted images
    iw->load(ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + ui->listWidget->item(0)->text());
    iw->show();

    QString msg = "Reloaded the image list with undistortion corrected!";
    ui->msgBox->setText(msg);
    if(iw->depth==0) ui->msgBox->append("8 bit depth");
    if(iw->depth==2) ui->msgBox->append("16 bit depth");



    QFile file(ui->folderName->text() + QDir::separator()  + "list.txt");
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug() << "error";
    }
    else{
        QTextStream stream( &file );
        for(int row = 0; row < ui->listWidget->count(); row++)
        {
            stream << ui->listWidget->item(row)->text() << endl;
        }
    }
    QApplication::processEvents();


}

void RTITool::loadCalib(QString fileName)
{
    QFile file(fileName);

    QString dstFile = ui->folderName->text() + QDir::separator()  + "calib.txt";



    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
         ui->msgBox->append("Invalid calibration file");
        return;
       }

        QFile::copy(fileName, dstFile);

    qDebug(fileName.toLatin1());
    QStringList  fields;

    QTextStream textStream(&file);
    ui->msgBox->append("Camera Matrix");


    for(int i=0;i<3;i++)
    {
        QString line = textStream.readLine();

        if (line.isNull())
            break;
        else
            fields = line.split(' ');


    if(fields.size()<3)  {ui->msgBox->append("Invalid calibration file");
    return;}

        for(int k=0;k<3;k++)
            iw->cameraMatrix.at<double>(i,k) = fields.at(k).toDouble();

        QString str = QString("%1 %2 %3")
                .arg(iw->cameraMatrix.at<double>(i,0),1,'f',2)
                .arg(iw->cameraMatrix.at<double>(i,1),1,'f',2)
                .arg(iw->cameraMatrix.at<double>(i,2),1,'f',2);
        ui->msgBox->append(str);
    }


    QString line = textStream.readLine();
    if (line.isNull()) return;
    else {


     fields = line.split(' ');
     if(fields.size()<4){
     ui->msgBox->append("Invalid calibration file");
    return;
        }
    for(int i=0;i<4;i++){


        iw->distCoeffs.at<double>(i,0) = fields.at(i).toDouble();
    }
    QString stri = QString("%1 %2 %3 %4")
            .arg(iw->distCoeffs.at<double>(0,0),1,'f',2)
            .arg(iw->distCoeffs.at<double>(1,0),1,'f',2)
            .arg(iw->distCoeffs.at<double>(2,0),1,'f',2)
            .arg(iw->distCoeffs.at<double>(3,0),1,'f',2);
    ui->k1Lab->setText(stri);
    }

    QString str = QString("%1 %2")
            .arg(iw->cameraMatrix.at<double>(0,0),1,'f',2)
            .arg(iw->cameraMatrix.at<double>(1,1),1,'f',2);
    ui->focalLab->setText(str);
    str = QString("%1 %2")
            .arg(iw->cameraMatrix.at<double>(0,2),1,'f',2)
            .arg(iw->cameraMatrix.at<double>(1,2),1,'f',2);
    ui->centerLab->setText(str);

    ui->intriLab->setText("Loaded");
    file.close();

}


void RTITool::on_loadCalibButton_clicked()
{
    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open calibration file"));

    loadCalib(fileName);

}



void RTITool::on_r2_1Spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere1->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->ry[0] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[0]->setScaledContents(true);;;
    QPainter painter(&pixmap);

    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        trans.translate(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor);
        trans.rotate(iw->angle[0]);
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[0]*iw->scaleFactor),(iw->ry[0]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor));
    }
    iw->ls[0]->setPixmap(pixmap);
    iw->ls[0]->setVisible(true);

}

void RTITool::on_r2_2Spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere2->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->ry[1] = arg1;

    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[1]->setScaledContents(true);;;
    QPainter painter(&pixmap);

    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        trans.translate(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor);
        trans.rotate(iw->angle[1]);
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[1]*iw->scaleFactor),(iw->ry[1]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor));
    }
    iw->ls[1]->setPixmap(pixmap);
    iw->ls[1]->setVisible(true);
}


void RTITool::on_r2_3Spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere3->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->ry[2] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[2]->setScaledContents(true);;
    QPainter painter(&pixmap);

    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        trans.translate(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor);
        trans.rotate(iw->angle[2]);
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[2]*iw->scaleFactor),(iw->ry[2]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor));
    }
    iw->ls[2]->setPixmap(pixmap);
    iw->ls[2]->setVisible(true);
}

void RTITool::on_r2_4Spin_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->sphere4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->ry[3] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[3]->setScaledContents(true);
    QPainter painter(&pixmap);

    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        trans.translate(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor);
        trans.rotate(iw->angle[3]);
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[3]*iw->scaleFactor),(iw->ry[3]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor));
    }
    iw->ls[3]->setPixmap(pixmap);
    iw->ls[3]->setVisible(true);
}

void RTITool::on_angle1spin_valueChanged(double arg1)
{
    int ax,ay,bx,by;
    iw->sphere1->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->angle[0] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[0]->setScaledContents(true);
    QPainter painter(&pixmap);

    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        trans.translate(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor);
        trans.rotate(iw->angle[0]);
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[0]*iw->scaleFactor),(iw->ry[0]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[0]*iw->scaleFactor,iw->cy[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor),(iw->radius[0]*iw->scaleFactor));
    }
    iw->ls[0]->setPixmap(pixmap);
    iw->ls[0]->setVisible(true);
}


void RTITool::on_angle2spin_valueChanged(double arg1)
{
    int ax,ay,bx,by;
    iw->sphere2->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->angle[1] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[1]->setScaledContents(true);
    QPainter painter(&pixmap);

    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        trans.translate(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor);
        trans.rotate(iw->angle[1]);
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[1]*iw->scaleFactor),(iw->ry[1]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[1]*iw->scaleFactor,iw->cy[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor),(iw->radius[1]*iw->scaleFactor));
    }
    iw->ls[1]->setPixmap(pixmap);
    iw->ls[1]->setVisible(true);
}



void RTITool::on_angle3spin_valueChanged(double arg1)
{
    int ax,ay,bx,by;
    iw->sphere3->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->angle[2] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[2]->setScaledContents(true);
    QPainter painter(&pixmap);

    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        trans.translate(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor);
        trans.rotate(iw->angle[2]);
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[2]*iw->scaleFactor),(iw->ry[2]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[2]*iw->scaleFactor,iw->cy[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor),(iw->radius[2]*iw->scaleFactor));
    }
    iw->ls[2]->setPixmap(pixmap);
    iw->ls[2]->setVisible(true);
}




void RTITool::on_angle4spin_valueChanged(double arg1)
{
    int ax,ay,bx,by;
    iw->sphere4->frameGeometry().getCoords(&ax,&ay,&bx,&by);

    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);

    iw->angle[3] = arg1;


    QPixmap pixmap((int)((float)(bx-ax)*iw->scaleFactor),(int)((float)(by-ay)*iw->scaleFactor));
    pixmap.fill(QColor("transparent"));
    iw->ls[3]->setScaledContents(true);
    QPainter painter(&pixmap);

    if(ui->projBox->isChecked())
    {
        painter.setPen(QPen(Qt::green));
        painter.save();
        QTransform trans;
        trans.translate(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor);
        trans.rotate(iw->angle[3]);
        painter.setTransform(trans);
        painter.drawEllipse(QPointF(0,0),(iw->rx[3]*iw->scaleFactor),(iw->ry[3]*iw->scaleFactor));
        painter.restore();
    }
    else
    {
        painter.setPen(QPen(Qt::red));
        painter.drawEllipse(QPointF(iw->cx[3]*iw->scaleFactor,iw->cy[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor),(iw->radius[3]*iw->scaleFactor));
    }
    iw->ls[3]->setPixmap(pixmap);
    iw->ls[3]->setVisible(true);
}



void RTITool::on_prevButton_clicked()
{
    if( ui->listWidget->selectionModel()->currentIndex().row() >0){
        int num = ui->listWidget->selectionModel()->currentIndex().row()-1;
        ui->listWidget->setCurrentRow(num);
        qDebug() << ui->listWidget->item(num)->text();
        iw->load(ui->folderName->text() + QDir::separator()  + "images" + QDir::separator()  + ui->listWidget->item(num)->text());
        iw->show();
    }
}

void RTITool::on_nextButton_clicked()
{
    if( ui->listWidget->selectionModel()->currentIndex().row() < ui->listWidget->count()-1){
        int num = ui->listWidget->selectionModel()->currentIndex().row()+1;
        ui->listWidget->setCurrentRow(num);
        //iw->load(ui->listWidget->item(num)->text());
        iw->load(ui->folderName->text() + QDir::separator()  + "images" + QDir::separator()  + ui->listWidget->item(num)->text());

        iw->show();
    }
}

void RTITool::saveId(QString filename)
{
    QFile file(filename);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug() << "error";
    }
    else{



        QTextStream stream( &file );

        int flag_dir=0;

        if(ui->dirInfo->currentIndex()==1) flag_dir=1;
        if(ui->dirInfo->currentIndex()==2) flag_dir=2;



        if(flag_dir==2 && iw->coilix[0].size()< 2 ){
            ui->msgBox->append("ERROR: Interpolate direction first");
            return;
        }



        //non cropped
        stream << "InterpolatedDirections " << ui->listWidget->count() << "\n";
        for(int i=0;i<ui->listWidget->count();i++){
            //   stream << "IMG " << i <<"\n";
            if(iw->coilix[0].size()>0)
                stream << iw->coilix[0].at(i) << " " << iw->coilix[1].at(i) << " " << iw->coilix[2].at(i) << " " << iw->coiliy[0].at(i) << " " << iw->coiliy[1].at(i) << " " << iw->coiliy[2].at(i) << " " << iw->coiliz[0].at(i) << " " << iw->coiliz[1].at(i) << " " << iw->coiliz[2].at(i) << "\n";
        }




    }
    file.close();
}

void RTITool::on_saveIdButton_clicked()
{
    QString filename = QFileDialog::getSaveFileName(
                this,
                tr("File name"),
                QDir::currentPath(),
                tr("*.id") );
    saveId(filename);


}


//void RTITool::saveSd(QString filename)
//{
//    QFile file(filename);
//    if (!file.open(QFile::WriteOnly | QFile::Text)) {
//        qDebug() << "error";
//    }
//    else{



//        QTextStream stream( &file );

//        int flag_dir=0;

//        if(ui->dirInfo->currentIndex()==1) flag_dir=1;
//        if(ui->dirInfo->currentIndex()==2) flag_dir=2;





//    }
//    file.close();
//}



void RTITool::on_loadIdButton_clicked()
{

    float** dircoeffs;

    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open interp. directions file"));

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QString destName = ui->folderName->text() + QDir::separator() + "file.id";

    if (!QFile::copy(fileName, destName))
        return;

    QFileInfo fi = QFileInfo(file);
    qDebug(fileName.toLatin1());

    ui->filenameID->setText(fileName);
    int last= fileName.lastIndexOf(QDir::separator());
    QString folder=fileName.left(last+1);
    Vec3f* dirs;

    QTextStream textStream(&file);
    while (true)
    {
        QString line = textStream.readLine();
        QStringList parts = line.split(" ");
        if(parts[0] != QString("InterpolatedDirections"))
            return;
        int nimg = parts[1].toInt();

        dircoeffs = new float*[nimg];
        dirs = new Vec3f[nimg];
        qDebug() << nimg ;

        for(int i=0;i<nimg;i++)   {
            line = textStream.readLine();
            if (line.isNull()){
                qDebug() << "error";
                return;
            }

            parts = line.split(" ");


            if(parts.size() < 6){
                qDebug() << "error";
                return;
            }

            for(int j=0;j<9;j++)
                dircoeffs[i][j] = parts[j].toFloat();

            qDebug() << dircoeffs[i][0] << " " << dircoeffs[i][1] << " " << dircoeffs[i][5];
            float c_x=100;//iw->imageLabel->width()/2;
            float c_y=100;//iw->imageLabel->height()/2;
            //                                    dirs[i][0]=dircoeffs[i][0]*c_x+dircoeffs[i][1]*c_y+dircoeffs[i][2];
            //                                    dirs[i][1]=dircoeffs[i][3]*c_x+dircoeffs[i][4]*c_y+dircoeffs[i][5];
            //                                    dirs[i][2]=dircoeffs[i][6]*c_x+dircoeffs[i][7]*c_y+dircoeffs[i][8];
            //                                    qDebug() << dirs[i][0] << " " << dirs[i][1] << " "  << dirs[i][2];


        }
    }
    file.close();



}

void RTITool::on_loadCalimButton_clicked()
{

    bList.clear();
    ui->backList->clear();

    if(!QDir(ui->folderName->text() + QDir::separator() +"CAL_IMG").exists())
        QDir().mkdir(ui->folderName->text() + QDir::separator() +"CAL_IMG");

    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open interp. directions file"));

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QFileInfo fi = QFileInfo(file);


    QProgressDialog progress("Importing files please wait...", "", 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);
    progress.setCancelButton(0);
    progress.setWindowTitle("Progress Dialog");
    progress.show( );


    /*ui->filenameID->setText(fileName);*/
    int last= fileName.lastIndexOf(QDir::separator());
    /* QString folder=fileName.left(last+1);*/
    int ni=0;
    QTextStream textStream(&file);
    while (true)
    {

        ni++;
        progress.setValue(ni);
        QString line = textStream.readLine();
        last= line.lastIndexOf(QDir::separator());
        QString lastname=line.right(line.size()-last-1);
        lastname.replace(" ","_");

        qDebug(line.toLatin1());
        if (line.isNull())
            break;
        else
            bList.append(lastname);

        QString dstImg=ui->folderName->text() + QDir::separator() + "CAL_IMG"+ QDir::separator() + lastname;
        QFile::copy(line,dstImg);
    }
    file.close();


    if(bList.count() != ui->listWidget->count())
        ui->msgBox->append("ERROR");

    ui->backList->addItems( bList );

    QString destName = ui->folderName->text() + QDir::separator() + "backlist.txt";

    if (!QFile::copy(fileName, destName))
        return;

}

void RTITool::on_corrBackimgBut_clicked()
{

    QListWidgetItem *item, *corritem;
    Mat image;
    Mat corrim;
    Mat gim;
    Mat gci;
    Mat CI;

    double lx, ly, lz, elev, azim, plx,ply,plz;
    double tlx, tly, tlz, telev, tazim;

    QString outname;
    if(!QDir(ui->folderName->text() + QDir::separator() +"CORR_IMG").exists())
        QDir().mkdir(ui->folderName->text() + QDir::separator() +"CORR_IMG");

    QProgressDialog pdialog("Saving corrected images","",0,100,this);
    pdialog.setWindowModality(Qt::WindowModal);
    pdialog.setCancelButton(0);
    pdialog.setValue(0);
    pdialog.setWindowTitle("Progress Dialog");

    pdialog.show();

    for(int row = 0; row < ui->listWidget->count(); row++)
    {

        pdialog.setValue(100*row/ui->listWidget->count());
        QApplication::processEvents();


        item = ui->listWidget->item(row);
        QString backim = bList.at(row);

        outname = ui->folderName->text() + QDir::separator() +"CORR_IMG" + QDir::separator() + "corrected" + QString::number(row) + ".tif";

        QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();
        QString corn = ui->folderName->text() + QDir::separator()  + "CAL_IMG" + QDir::separator() + backim;


        if(!iw->force8bit){
            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
            corrim = cv::imread(corn.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
        }
        else{
            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
            corrim = cv::imread(corn.toStdString(), CV_LOAD_IMAGE_COLOR);
        }
        double min, max,aa;
        cv::minMaxLoc(image, &min, &max);
       // qDebug() << "minmax " << min << max;

        CI = image.clone();
        cv::cvtColor(image,gim,CV_BGR2GRAY);
        cv::cvtColor(corrim,gci,CV_BGR2GRAY);

     //imwrite( "test.png", gci );
        cv::GaussianBlur( gci, gci, Size( 401,401),0);
        //  imwrite( "t0st.png", gci );


        // estimate constant direction per image
      if(!ui->weightDir->isChecked()){
            plx=ply=plz=0;
            int nl=0;
            for(int i=0; i<4; i++)
                if(! iw->lights[i].size() == 0){
                    plx+=iw->lights[i].at(row)[0];
                    ply+=iw->lights[i].at(row)[1];
                    plz+=iw->lights[i].at(row)[2];
                    nl++;
                }
            if(nl==0){
                ui->msgBox->append("Error: please estimate light direction first");
                return;
            }
            plx/=nl;
            ply/=nl;
            plz/=nl;
            double norm = sqrt(plx*plx+ply*ply+plz*plz);
            plx=plx/norm;ply=ply/norm;tlz=tlz/norm;
            qDebug() <<  "OK " << plx << " " << ply << " " << plz ;

        }





        for(int i=0;i<image.cols;i++)
            for(int j=0;j<image.rows;j++){
                double vx =i;
                double vy =j;

                if(ui->weightDir->isChecked()){
                    plx = vx*iw->coilix[0].at(row)+vy*iw->coilix[1].at(row)+ iw->coilix[2].at(row);
                    ply = vx*iw->coiliy[0].at(row)+vy*iw->coiliy[1].at(row)+ iw->coiliy[2].at(row);
                    plz = vx*iw->coiliz[0].at(row)+vy*iw->coiliz[1].at(row)+ iw->coiliz[2].at(row);
                    double norm = sqrt(plx*plx+ply*ply+plz*plz);
                    plx=plx/norm;ply=ply/norm;plz=plz/norm;
                }

                if(iw->depth == 0){
                    Vec3b color = image.at<Vec3b>(Point(i,j));
                    unsigned char gray = gim.at<unsigned char>(Point(i,j));

                    aa= (double) gci.at<unsigned char>(Point(i,j));                 
                    aa=abs(aa/plz);

                    double ww = ui->refWhite_2->value();
                    for(int k=0;k<3;k++)
                        color[k] = MIN(255,MAX(0,color[k]*ww*255.0/aa));

                    CI.at<Vec3b>(Point(i,j)) = color;

                }
                else{
                    Vec<unsigned short, 3>  color = image.at<Vec<unsigned short, 3> >(Point(i,j));
                    unsigned short gray = (unsigned short) gim.at<unsigned short>(Point(i,j));

                    aa= (double) gci.at<unsigned short>(Point(i,j));
                    aa=abs(aa/plz);
                    double ww = ui->refWhite_2->value();

                    for(int k=0;k<3;k++)
                        color[k] = (unsigned short) MIN(65535,MAX(0,(((unsigned short)color[k])*ww*65535.0/aa)));

                    CI.at<Vec<unsigned short, 3> >(Point(i,j)) = color;
                    CI.convertTo(CI,CV_16U);
                }
            }

        imwrite( outname.toStdString(), CI );
        image.release();
        corrim.release();
        CI.release();
        gim.release();
        gci.release();
    }

    ui->savecLab->setText("OK");



}

void RTITool::on_createProjectButton_clicked()
{
    QString folder = QFileDialog::getExistingDirectory(this,
                                                       tr("Choose Or Create Directory"),
                                                       "./",
                                                       QFileDialog::DontResolveSymlinks);
    QDir pathDir(folder + QDir::separator() + "images");
    if (pathDir.exists()) {
        ui->msgBox->append("Error: can't overwrite existing project");
        return;
    }

    pathDir.mkdir(folder + QDir::separator() + "images");
    ui->folderName->setText(folder);

    ui->tabDir->setTabEnabled(2, true);
    ui->tabDir->setTabEnabled(3, true);
    ui->tabDir->setTabEnabled(4, true);
    ui->tabDir->setTabEnabled(1, true);


}

void RTITool::on_openProjectButton_clicked()
{
    QString folder = QFileDialog::getExistingDirectory(this,
                                                       tr("Choose Directory"),
                                                       "./",
                                                       QFileDialog::DontResolveSymlinks);
    QDir pathDir(folder + QDir::separator() + "images");

    if (!pathDir.exists()) {
        ui->msgBox->append("Error: invalid project");
        return;
    }

    // check all the other info and load data

    ui->folderName->setText(folder);
    ui->tabDir->setTabEnabled(2, true);
    ui->tabDir->setTabEnabled(3, true);
    ui->tabDir->setTabEnabled(4, true);
    ui->tabDir->setTabEnabled(1, true);

    // load existing data
    QString  lpf = ui->folderName->text() + QDir::separator() + "file.lp";
    QFile flp(lpf);

    if(flp.exists())
    {
        loadLp(lpf);

    }
    else
    {

        QString  listf = ui->folderName->text() + QDir::separator() + "list.txt";
        loadList(listf);

        QString  lpd = ui->folderName->text() + QDir::separator() + "dirfile.lp";
        QFile flp(lpd);
        loadDirFromLp(lpd);


    }

    QString  calf  = ui->folderName->text() + QDir::separator() + "calib.txt";
    loadCalib(calf);
    QString  corrif  = ui->folderName->text() + QDir::separator() + "corrim.txt";
    loadCorrim(corrif);

    QString  corrd  = ui->folderName->text() + QDir::separator() + "backlist.txt";
    loadCorrData(corrd);


    QString  radf  = ui->folderName->text() + QDir::separator() + "radius.txt";

    QFile file(radf);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;
    else{
    QTextStream textStream(&file);

    QString line = textStream.readLine();
    if (!line.isNull()){
     ui->spinSphere->setValue(line.toDouble());
    }
}

    if(QDir(ui->folderName->text() + QDir::separator() + "CORR_IMG").exists())
        ui->savecLab->setText("OK");


    if(QDir(ui->folderName->text() + QDir::separator() +"CAL_IMG").exists()){

        importCalim(ui->folderName->text() + QDir::separator() +"CAL_IMG" );
    }

    if(QDir(ui->folderName->text() + QDir::separator() +"box0.txt").exists()){

    QFile file(ui->folderName->text() + QDir::separator() +"box0.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QTextStream textStream(&file);

    QString line = textStream.readLine();
    RotatedRect box;
    if (!line.isNull()){
    QStringList parts = line.split(" ");
    box.center.x=parts.at(0).toFloat();
    box.center.y=parts.at(1).toFloat();
    box.angle=parts.at(2).toFloat();
    box.size.width=parts.at(3).toFloat();
    box.size.height=parts.at(4).toFloat();
    iw->boxE[0]=box;
    }
    file.close();
    }
    if(QDir(ui->folderName->text() + QDir::separator() +"box1.txt").exists()){

    QFile file(ui->folderName->text() + QDir::separator() +"box1.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QTextStream textStream(&file);

    QString line = textStream.readLine();
    RotatedRect box;
    if (!line.isNull()){
    QStringList parts = line.split(" ");
    box.center.x=parts.at(0).toFloat();
    box.center.y=parts.at(1).toFloat();
    box.angle=parts.at(2).toFloat();
    box.size.width=parts.at(3).toFloat();
    box.size.height=parts.at(4).toFloat();
    iw->boxE[1]=box;
    }
    file.close();
    }
    if(QDir(ui->folderName->text() + QDir::separator() +"box2.txt").exists()){
        QFile file(ui->folderName->text() + QDir::separator() +"box2.txt");
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;

        QTextStream textStream(&file);

        QString line = textStream.readLine();
        RotatedRect box;
        if (!line.isNull()){
        QStringList parts = line.split(" ");
        box.center.x=parts.at(0).toFloat();
        box.center.y=parts.at(1).toFloat();
        box.angle=parts.at(2).toFloat();
        box.size.width=parts.at(3).toFloat();
        box.size.height=parts.at(4).toFloat();
        iw->boxE[2]=box;
        }
        file.close();
    }
    if(QDir(ui->folderName->text() + QDir::separator() +"box3.txt").exists()){
        QFile file(ui->folderName->text() + QDir::separator() +"box3.txt");
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;

        QTextStream textStream(&file);

        QString line = textStream.readLine();
        RotatedRect box;
        if (!line.isNull()){
        QStringList parts = line.split(" ");
        box.center.x=parts.at(0).toFloat();
        box.center.y=parts.at(1).toFloat();
        box.angle=parts.at(2).toFloat();
        box.size.width=parts.at(3).toFloat();
        box.size.height=parts.at(4).toFloat();
        iw->boxE[3]=box;
        }
        file.close();

    }




}

void RTITool::on_findImagesButton_clicked()
{

    QString folder = ui->folderName->text();
    QDir directory(folder + QDir::separator()  + "images");
    QStringList images = directory.entryList(QStringList() << "*.jpg" << "*.JPG"<< "*.tiff" << "*.tif"<< "*.TIFF" << "*.TIF"<< "*.png" << "*.PNG",QDir::Files);


    if(images.isEmpty()) {
        ui->msgBox->append("No images found");
        return;
    }

    ui->listWidget->clear();
    ui->listWidget->addItems( images );

    iw->load(folder + QDir::separator() + "images" + QDir::separator() + ui->listWidget->item(0)->text());
    iw->show();

    QString msg = "Loaded list file with " + QString::number(ui->listWidget->count()) + " images";
    ui->msgBox->setText(msg);
    if(iw->depth==0) ui->msgBox->append("8 bit depth");
    if(iw->depth==2) ui->msgBox->append("16 bit depth");

    QFile lfile(ui->folderName->text() + QDir::separator() + "list.txt");

    if (!lfile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "error";
    }
    else{
        QTextStream out(&lfile);
        for(int row = 0; row < ui->listWidget->count(); row++)
        {
            out << ui->listWidget->item(row)->text() << endl;
        }
    }


}

void RTITool::on_copyFolderButton_clicked()
{
    QString sourceFolder = getFoldername();
    /*QFileDialog::getExistingDirectory(this,
                           tr("Choose Directory"),
                           "./",
                           QFileDialog::DontResolveSymlinks);*/
    QDir sourceDir(sourceFolder);

    QString folder = ui->folderName->text();

    QString destFolder = folder + QDir::separator()  + "images";

    ui->listWidget->clear();

    QProgressDialog progress("Importing files please wait...", "", 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);
    progress.setCancelButton(0);
    progress.setWindowTitle("Progress Dialog");
    progress.show( );
    QCoreApplication::processEvents();

    QStringList files = sourceDir.entryList(QDir::Files);
    for(int i = 0; i< files.count(); i++) {
        QString srcName = sourceFolder + QDir::separator() + files[i];
        files[i].replace(" ","_");
        QString destName = destFolder + QDir::separator() + files[i];

        progress.setValue(i*100/files.count());
        if (!QFile::copy(srcName, destName))
            return;
    }

    QDir directory(destFolder);
    QStringList images = directory.entryList(QStringList() << "*.jpg" << "*.JPG"<< "*.tiff" << "*.tif"<< "*.TIFF" << "*.TIF"<< "*.png" << "*.PNG",QDir::Files);

    ui->listWidget->addItems( images );

    iw->load(folder + QDir::separator() + "images" + QDir::separator() + ui->listWidget->item(0)->text());
    iw->show();

    QFile lfile(ui->folderName->text() + QDir::separator() + "list.txt");

    if (!lfile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "error";
    }
    else{
        QTextStream out(&lfile);
        for(int row = 0; row < ui->listWidget->count(); row++)
        {
            out << ui->listWidget->item(row)->text() << endl;
        }
    }

}


QString RTITool::getFilename()
{
    QString fileName;
    QFileDialog dialog;
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setDirectory(QDir::currentPath());
    dialog.exec();
    QStringList list = dialog.selectedFiles();
    if(!list.isEmpty())
        fileName = list.at(0);
    qDebug()<<fileName;
    return(fileName);
}

QString RTITool::getFoldername()
{
    QString folderName;
    QFileDialog dialog;
    dialog.setFileMode(QFileDialog::Directory);
    dialog.setDirectory(QDir::currentPath());
    dialog.exec();
    QStringList list = dialog.selectedFiles();
    if(!list.isEmpty())
        folderName = list.at(0);

    return(folderName);
}




void RTITool::loadCorrim(QString fileName)
{
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    QStringList  fields;

    QTextStream textStream(&file);

    QString line = textStream.readLine();

    if(line.toInt()==1)
        ui->weightDir->setChecked(true);

    line = textStream.readLine();

    ui->refWhite->setValue(line.toDouble());
    file.close();

}


void RTITool::loadCorrData(QString fileName)
{
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;

    bList.clear();


    int last= fileName.lastIndexOf(QDir::separator());
    QString folder=fileName.left(last+1);

    QProgressDialog progress("Importing files please wait...", "", 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);
    progress.setCancelButton(0);
    progress.setWindowTitle("Progress Dialog");
    progress.show( );

    int line_count=0;


    QTextStream textStream(&file);

    while( !textStream.atEnd())
    {
       textStream.readLine();
        line_count++;
    }

    textStream.seek(0);
    int ni=0;
    while (true)
    {
        ni++;
        QString line = textStream.readLine();
        last= line.lastIndexOf(QDir::separator());
        QString lastname=line.right(line.size()-last-1);
        lastname.replace(" ","_");

        progress.setValue(ni*100/line_count);

        qDebug(line.toLatin1());
        if (line.isNull())
            break;
        else{
            bList.append(lastname);

        }
    }
    file.close();
    ui->backList->clear();
    ui->backList->addItems( bList );

    file.close();

}

void RTITool::importCalim(QString sourceFolder){
     bList.clear();

     QDir sourceDir(sourceFolder);

     QString folder = ui->folderName->text();

     QString destFolder = folder + QDir::separator()  + "CAL_IMG";

     QProgressDialog progress("Importing files please wait...", "", 0, 100, this);
     progress.setWindowModality(Qt::WindowModal);
     progress.setValue(0);
     progress.setCancelButton(0);
     progress.setWindowTitle("Progress Dialog");
     progress.show( );
     QCoreApplication::processEvents();

     bList = sourceDir.entryList(QDir::Files);


     for(int i = 0; i< bList.count(); i++) {
         QString srcName = sourceFolder + QDir::separator() + bList[i];
         QString destName = destFolder + QDir::separator() + bList[i];

         progress.setValue(i*100/bList.count());
         if (!QFile::copy(srcName, destName))
             return;
     }



     if(bList.count() != ui->listWidget->count())
         ui->msgBox->append("ERROR");

  ui->backList->addItems( bList );
}




void RTITool::findCalim(){
     bList.clear();

     QString folder = ui->folderName->text();

     QDir sourceDir(folder + QDir::separator()  + "CAL_IMG");

     bList = sourceDir.entryList(QDir::Files);

     if(bList.size() != ui->listWidget->count())
         ui->msgBox->append("ERROR");
}


void RTITool::on_importCalimButton_clicked()
{



    if(!QDir(ui->folderName->text() + QDir::separator() +"CAL_IMG").exists())
        QDir().mkdir(ui->folderName->text() + QDir::separator() +"CAL_IMG");


    QString sourceFolder = getFoldername();

    importCalim(sourceFolder);



}

void RTITool::on_lpDirButton_clicked()
{

    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open lp file"));

    QString dstFile = ui->folderName->text() + QDir::separator()  + "dirfile.lp";

    QFile::copy(fileName, dstFile);

    loadDirFromLp(fileName);
}


void RTITool::on_undistortCalibBut_clicked()
{
     //  ui->backList->addItems( bList );
       bList.clear();


       cv::Mat image;
       QMessageBox messageBox;



       if (ui->backList->count() == 0){
           qDebug() << "error";
           messageBox.critical(0,"Error","There is no loaded image to undistort!");
           messageBox.setFixedSize(500,200);
           return;
       }

       if (iw->cameraMatrix.at<double>(0,2) ==0 ){
           qDebug() << "error";
           messageBox.critical(0,"Error","Please load the calibration parameters!");
           messageBox.setFixedSize(500,200);
           return;
       }

       if (iw->distCoeffs.empty() ){
           qDebug() << "error";
           messageBox.critical(0,"Error","Please load the calibration parameters!");
           messageBox.setFixedSize(500,200);
           return;
       }

       QString originalImageName;
       QString undistortedImageName;

       QProgressDialog progress("Undistorting images please wait...", "", 0, ui->listWidget->count(), this);
       progress.setWindowModality(Qt::WindowModal);
       progress.setValue(0);
       progress.setCancelButton(0);
       progress.setWindowTitle("Progress Dialog");
       progress.show( );
       QCoreApplication::processEvents();

       for(int row = 0; row < ui->backList->count(); row++)
       {
           QListWidgetItem *item = ui->backList->item(row);
           originalImageName = item->text();

           progress.setValue(row);

           int last= originalImageName.lastIndexOf(".");
           QString common=originalImageName.left(last);

           undistortedImageName = common + "_und.tiff";

           QString filen = ui->folderName->text() + QDir::separator()  + "CAL_IMG" + QDir::separator() + item->text();
           cv::Mat image = imread(filen.toStdString(),CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_COLOR);
           cv::Mat image_und;
           qDebug() << "----- Undistorting image - " << filen << "...";
           cv::undistort(image,image_und,iw->cameraMatrix,iw->distCoeffs);
           qDebug() <<"\nDistance coeff"<<iw->distCoeffs.at<double>(0,0)<< iw->distCoeffs.at<double>(1,0)<<"  "<<iw->distCoeffs.at<double>(2,0)<<"  "<<iw->distCoeffs.at<double>(3,0)<<"  ";

           QString outn = ui->folderName->text() + QDir::separator()  + "CAL_IMG" + QDir::separator() + undistortedImageName;
           cv::imwrite(outn.toStdString(),image_und);
           ui->backList->item(row) ->setText(undistortedImageName);
            bList.append(undistortedImageName);
           image.release();
           image_und.release();
           QFile::remove(filen);
       }

       //reload the undistorted images
       //iw->load(ui->folderName->text() + QDir::separator()  + "CAL_IMG" + QDir::separator() + ui->backList->item(0)->text());
       //iw->show();

       QString msg = "Reloaded the image list with undistortion corrected!";
       ui->msgBox->setText(msg);
       if(iw->depth==0) ui->msgBox->append("8 bit depth");
       if(iw->depth==2) ui->msgBox->append("16 bit depth");



       QFile file(ui->folderName->text() + QDir::separator()  + "backlist.txt");
       if (!file.open(QFile::WriteOnly | QFile::Text)) {
           qDebug() << "error";
       }
       else{
           QTextStream stream( &file );
           for(int row = 0; row < ui->backList->count(); row++)
           {
               stream << ui->backList->item(row)->text() << endl;
           }
       }
       QApplication::processEvents();

       ui->msgBox->append("Background images undistorted");

}


void RTITool::on_closeProjectButton_clicked()
{
 ui->listWidget->clear();
 ui->msgBox->clear();
 ui->folderName->clear();
 ui->backList->clear();
 iw->setVisible(false);
    clearParams();



}

void RTITool::on_cy1spin_editingFinished()
{

}

void RTITool::on_spinOx_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->cropArea->geometry().getCoords(&ax,&ay,&bx,&by);
   /* ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    bx = (int) ((double)bx / iw->scaleFactor);
    by = (int) ((double)by / iw->scaleFactor);*/
    iw->originc = QPoint(arg1*iw->scaleFactor,ay);
    iw->endc = QPoint(arg1*iw->scaleFactor+bx-ax,by);

    iw->cropArea->setGeometry(QRect(iw->originc,iw->endc));
    //iw->cropArea->setGeometry(arg1,ay,bx,by);
 iw->cropArea->show();
    qDebug() <<iw->scaleFactor << " " << iw->zoomFactor;
   //

}

void RTITool::on_spinOy_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->cropArea->geometry().getCoords(&ax,&ay,&bx,&by);
    iw->originc = QPoint(ax,arg1*iw->scaleFactor);
    iw->endc = QPoint(bx,arg1*iw->scaleFactor+by-ay);

    iw->cropArea->setGeometry(QRect(iw->originc,iw->endc));
     iw->cropArea->show();
   // iw->cropArea->setGeometry(ax,arg1,bx,by);
}

void RTITool::on_spinSx_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->cropArea->geometry().getCoords(&ax,&ay,&bx,&by);

    //iw->originc = QPoint(ax,ay);
    iw->endc = QPoint(arg1*iw->scaleFactor+ax,by);

   iw->cropArea->setGeometry(QRect(iw->originc,iw->endc));
  //  iw->cropArea->setGeometry(ax,ay,arg1-ax,by);
 iw->cropArea->show();
}

void RTITool::on_spinSy_valueChanged(int arg1)
{
    int ax,ay,bx,by;
    iw->cropArea->geometry().getCoords(&ax,&ay,&bx,&by);

    //iw->originc = QPoint(ax,ay);
    iw->endc = QPoint(bx,arg1*iw->scaleFactor+ay);

    iw->cropArea->setGeometry(QRect(iw->originc,iw->endc));
    iw->cropArea->show();
    //iw->cropArea->setGeometry(ax,ay,bx,arg1-ay);
}



void RTITool::on_pointsButton_clicked()
{
    
    iw->active=11;
}


void RTITool::on_gammaButton_toggled(bool checked)
{
    iw->gc = !iw->gc ;
}

void RTITool::on_gammaSpinBox_valueChanged(double arg1)
{
    iw->gamma=arg1;
}

void RTITool::on_saveRelBut_clicked()
{

    int flag_dir=0;
    int flag_chr=1;
    int flag_lum=0;
    int flag_corr=0;
    int flag_binary = 1;

    Rect roi;

    QString file_pref = QInputDialog::getText(this, "Enter Filename with suffix", "");
    QString  filename=ui->folderName->text() + QDir::separator() + file_pref + ".lp";

    QListWidgetItem *item = ui->listWidget->item(0);

    if(ui->dirInfo->currentIndex()==1) flag_dir=1;
    if(ui->dirInfo->currentIndex()==2) flag_dir=2;

    QProgressDialog pdialog("Saving appearance profile","",0,100,this);
    pdialog.setWindowModality(Qt::WindowModal);
    pdialog.setCancelButton(0);
    pdialog.setValue(0);
    pdialog.setWindowTitle("Progress Dialog");
    pdialog.show();

    if(flag_dir==2 && iw->coilix[0].size()< 2 ){
        ui->msgBox->append("ERROR: Interpolate direction first");
        return;
    }


    if(ui->cInt->isChecked()) flag_corr = 1;
    //if(ui->cBinary->isChecked()) flag_binary=1;
    //if(ui->cChr->isChecked()) flag_chr = 1;


    int ax, ay, sx,sy;
    iw->cropArea->frameGeometry().getRect(&ax,&ay,&sx,&sy);
    ax = (int) ((double)ax / iw->scaleFactor);
    ay = (int) ((double)ay / iw->scaleFactor);
    sx = (int) ((double)sx / iw->scaleFactor);
    sy = (int) ((double)sy / iw->scaleFactor);
    qDebug() << ax << " "<< ay << " "<< sx << " "<< sy << " ";


    if(ax > 0) {//cropped
        roi.x=ax;
        roi.y=ay;
        roi.width=sx;
        roi.height=sy;
       }else               {

        sx=iw->s.width();
    sy=iw->s.height();
    roi.x=0;
    roi.y=0;
    roi.width=sx;
    roi.height=sy;
}


    QFile file(filename);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug() << "error";
    }
    else{
        QTextStream stream( &file );


        stream <<  ui->listWidget->count() <<"\n";;


        // write single directions

    //        stream << "LIGHT_DIRECTIONS (lx ly lz)" <<"\n";
            int ns=0;
            double lx, ly,lz;
            for(int i=0;i<4;i++)
                if(iw->lights[i].size() >0) ns++;
            if(ns>0)
                for(int i=0;i<ui->listWidget->count();i++){

                    item = ui->listWidget->item(i);

                        stream <<  "JPG" << file_pref.toLatin1() << QDir::separator() << "cropped" << QString::number(i).toLatin1() << ".jpg" <<" ";
                    //  stream << "IMG " << i <<"\n";
                    lx=0;ly=0;lz=0;
                    for(int j=0;j<ns;j++){
                        lx=lx+ iw->lights[j].at(i)[0]/ns;
                        ly=ly+ iw->lights[j].at(i)[1]/ns;
                        lz=lz+ iw->lights[j].at(i)[2]/ns;
                    }
                    stream <<  lx << " " << ly << " " << lz << "\n";
                }


        //close lp file
        file.close();



        unsigned char val;

            Mat image, gim, cim, cim2, aim, mim;



                Mat crim2(sy,sx,CV_8UC3,cv::Scalar(0,0,0));

                // loop over images
                for(int row = 0; row < ui->listWidget->count(); row++)
                {
                    pdialog.setValue(100*row/ui->listWidget->count());
                    pdialog.update();

                    item = ui->listWidget->item(row);
                    QString filen = ui->folderName->text() + QDir::separator()  + "images" + QDir::separator() + item->text();

                    QString cname;

                    if(flag_corr==0){
                        if(!iw->force8bit)
                            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
                        else
                            image = cv::imread(filen.toStdString(), CV_LOAD_IMAGE_COLOR);
                    }
                    //  image = imread(item->text().toStdString(), CV_LOAD_IMAGE_COLOR);
                    else {
                        cname =  ui->folderName->text() + QDir::separator()  + "CORR_IMG" + QDir::separator()  + "corrected" + QString::number(row) +".tif";
                        if(!iw->force8bit)
                            image = cv::imread(cname.toStdString(), CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);
                        else
                            image = cv::imread(cname.toStdString(), CV_LOAD_IMAGE_COLOR);
                    }

                    if(! image.data )                              // Check for invalid input
                    {
                        ui->msgBox->append("Error: no corrected images available");
                        qDebug() << cname;
                        return;
                    }


                    cim = image(roi);

                    QString cropname =  ui->folderName->text() + QDir::separator()  + QString("JPG") + file_pref + QDir::separator() + QString("cropped") + QString::number(row) + ".jpg";

                    if(!QDir( ui->folderName->text() + QDir::separator()  + QString("JPG") + file_pref).exists())
                        QDir().mkdir( ui->folderName->text() + QDir::separator()  + QString("JPG") + file_pref);

                      if(iw->depth==0)
                        imwrite(cropname.toStdString(),cim);


                    if(iw->depth==2){
                        cim.convertTo(cim2, CV_8UC3, 0.00390625);
                        imwrite(cropname.toStdString(),cim2);

                    }


                        cim.release();
                        cim2.release();
                    image.release();
                }


}

}
