#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main()
{
    // 使用OpenCV的imread函数读取图像。
    // imread的第一个参数是图像文件的路径，第二个参数定义了读取图像的方式。使用IMREAD_COLOR，以彩色模式读取图像。
    cv::Mat image_1 = cv::imread("/home/zym/opencv_project/resources/test_image.png", cv::IMREAD_COLOR);

    // 检查图像数据是否存在。
    if (!image_1.data)
    {
        std::cout << "No image data \n";
        return -1;
    }

    // 创建一个新的Mat对象来存储灰度图像。
    cv::Mat gray_image;

    // 使用cvtColor函数将原图像转换为灰度图像。这个函数的第一个参数是输入图像，第二个参数是输出图像，第三个参数是转换类型。
    cv::cvtColor(image_1, gray_image, cv::COLOR_BGR2GRAY);

    // 使用namedWindow函数创建一个窗口来显示原图像。这个函数的第一个参数是窗口的名字，第二个参数是窗口的大小。在这个例子中，我们使用WINDOW_AUTOSIZE，意味着窗口的大小将自动适应图像的大小。
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);

    // 使用imshow函数在之前创建的窗口中显示原图像。这个函数的第一个参数是窗口的名字，第二个参数是要显示的图像。
    cv::imshow("Display Image", image_1);
    cv::waitKey(0);

    // 创建一个新的窗口来显示灰度图像，并在这个窗口中显示灰度图像。
    cv::namedWindow("Gray Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gray Image", gray_image);
    cv::imwrite("/home/zym/opencv_project/resources/out/gray_image.png",gray_image);
    cv::waitKey(0);

    //转化为HSV图片并输出
    cv::Mat hsv_image;
    cv::cvtColor(image_1,hsv_image,cv::COLOR_BGR2HSV);
    cv::namedWindow("Hsv Image",cv::WINDOW_AUTOSIZE);
    cv::imshow("Hsv Image",hsv_image);
    cv::imwrite("/home/zym/opencv_project/resources/out/hsv_image.png",hsv_image);
    cv::waitKey(0);

    //应用均值滤波
    cv::Mat blurred_image;
    cv::blur(image_1,blurred_image,cv::Size(50,50));
    cv::namedWindow("Blurred_image",cv::WINDOW_AUTOSIZE);
    cv::imshow("Blurred_image",blurred_image);
    cv::imwrite("/home/zym/opencv_project/resources/out/blurred_image.png",blurred_image);
    cv::waitKey(0);

    //应用高斯滤波
    cv::Mat gaussian_blurred_image;
    cv::GaussianBlur(image_1,gaussian_blurred_image,cv::Size(5,5),0);
    cv::namedWindow("Gaussian Blurred Image",cv::WINDOW_AUTOSIZE);
    cv::imshow("Gaussian Blurred Image",gaussian_blurred_image);
    cv::imwrite("/home/zym/opencv_project/resources/out/gaussian_blurred_image.png",gaussian_blurred_image);
    cv::waitKey(0);

    //运用HSV方法提取红色区域，根据图像实际情况对hsv值进行了微调，保证其包括了较浅和较深的红色
    cv::Mat lower_red_hue_range;
    cv::Mat upper_red_hue_range;
    cv::inRange(hsv_image, cv::Scalar(0, 51, 20), cv::Scalar(20, 255, 255), lower_red_hue_range);
    cv::inRange(hsv_image, cv::Scalar(160, 51, 20), cv::Scalar(180, 255, 255), upper_red_hue_range);
    cv::Mat red_hue_image;
    cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
    cv::namedWindow("Red Areas", cv::WINDOW_AUTOSIZE);
    cv::imshow("Red Areas", red_hue_image);
    cv::imwrite("/home/zym/opencv_project/resources/out/red_areas.png",red_hue_image);
    cv::waitKey(0);

    //先进行形态学操作优化
    cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(red_hue_image, red_hue_image, cv::MORPH_CLOSE, morph_kernel); // 使用较小的内核进行闭操作
    cv::morphologyEx(red_hue_image, red_hue_image, cv::MORPH_OPEN, morph_kernel);  // 开操作

    //备份原图，不在原图上绘制轮廓
    cv::Mat red_contour_image = image_1.clone();

    //寻找并绘制图像中红色的外轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_hue_image,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(red_contour_image,contours,-1,cv::Scalar(0,255,0),1);
    cv::namedWindow("Red Contours Image",cv::WINDOW_AUTOSIZE);
    cv::imshow("Red Contours Image",red_contour_image);
    cv::imwrite("/home/zym/opencv_project/resources/out/red_contour_image.png",red_contour_image);
    cv::waitKey(0);

    //绘制bounding box，对bounding box进行标号，方便终端查看，
    cv::Mat boundingbox_image = image_1.clone();
    int contour_id = 0;
    for (const auto& contour : contours) {
        
        contour_id++;//对遍历得到的轮廓一一标号
        double area = cv::contourArea(contour);
        cv::Rect bounding_box = cv::boundingRect(contour);
        cv::rectangle(boundingbox_image, bounding_box, cv::Scalar(0, 255, 0), 1);

        std::string area_text = std::to_string(contour_id);
        cv::putText(boundingbox_image, area_text, bounding_box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        std::cout << "Area" << contour_id << ":" << area << std::endl;
    }
    cv::namedWindow("Bounding Box",cv::WINDOW_AUTOSIZE);
    cv::imshow("Bounding Box",boundingbox_image);
    cv::imwrite("/home/zym/opencv_project/resources/out/boundingbox_image.png",boundingbox_image);

    cv::waitKey(0);

    //提取高亮区域
    std::vector<cv::Mat>hsv_channels;
    cv::split(hsv_image,hsv_channels);//分割hsv通道
    cv::Mat v_channel=hsv_channels[2];
    cv::Mat highlight_mask;
    cv::threshold(v_channel, highlight_mask, 200, 255, cv::THRESH_BINARY);//规定亮度超过200为高亮区域
    cv::Mat highlight_region;
    image_1.copyTo(highlight_region, highlight_mask);//在原图中提取高亮区域

    cv::namedWindow("Highlight Region",cv::WINDOW_AUTOSIZE);
    cv::imshow("Highlight Region",highlight_region);
    cv::imwrite("/home/zym/opencv_project/resources/out/highlight_region.png",highlight_region);

    cv::waitKey(0);

    //将高亮区域进行灰度化
    cv::Mat highlight_gray;
    cv::cvtColor(highlight_region,highlight_gray,cv::COLOR_BGR2GRAY);
    cv::namedWindow("Highlight Gray",cv::WINDOW_AUTOSIZE);
    cv::imshow("Highlight Gray",highlight_gray);
    cv::imwrite("/home/zym/opencv_project/resources/out/highlight_gray.png",highlight_gray);
    cv::waitKey(0);

    //高亮区域二值化
    cv::Mat highlight_binary;
    cv::threshold(highlight_gray, highlight_binary, 128, 255, cv::THRESH_BINARY);
    cv::imshow("Highlight Binary",highlight_binary);
    cv::imwrite("/home/zym/opencv_project/resources/out/highlight_binary.png",highlight_binary);
    cv::waitKey(0);

    //高亮区域膨胀处理
    int dilation_size = 3; // 核的大小 (可以调整)
    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, 
                                               cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), 
                                               cv::Point(dilation_size, dilation_size));
    cv::Mat highlight_dilated;
    cv::dilate(highlight_binary,highlight_dilated,kernel1);
    cv::imshow("Highlight Dilated",highlight_dilated);
    cv::imwrite("/home/zym/opencv_project/resources/out/highlight_dilated.png",highlight_dilated);
    cv::waitKey(0);

    //高亮区域腐蚀处理
    int erosion_size = 3; 
    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, 
                                               cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), 
                                               cv::Point(erosion_size, erosion_size));
    cv::Mat highlight_eroded;
    cv::erode(highlight_binary, highlight_eroded, kernel2);
    cv::imshow("Highlight Eroded",highlight_eroded);
    cv::imwrite("/home/zym/opencv_project/resources/out/highlight_eroded.png",highlight_eroded);
    cv::waitKey(0);

    //对floodFill函数参数设置
    cv::Point seedPoint(0, 0);
    int newVal = 255;
    int connectivity = 8;

    //灰度化后的漫水处理
    cv::Mat highlight_gray1= highlight_gray.clone();
    cv::Mat mask_g = cv::Mat::zeros(highlight_gray1.rows + 2, highlight_gray1.cols + 2, CV_8UC1);
    cv::floodFill(highlight_gray1, mask_g, seedPoint, newVal, 0, cv::Scalar(), cv::Scalar(), connectivity);
    cv::imshow("H_Gray Flood Filled", highlight_gray1);
    cv::imwrite("/home/zym/opencv_project/resources/out/H_Gray Flood Filled.png",highlight_gray1);
    cv::waitKey(0);         
    //二值化后的漫水处理
    cv::Mat highlight_binary1 = highlight_binary.clone();
    cv::Mat mask_b = cv::Mat::zeros(highlight_binary1.rows + 2, highlight_binary1.cols + 2, CV_8UC1);
    cv::floodFill(highlight_binary1, mask_b, seedPoint, newVal, 0, cv::Scalar(), cv::Scalar(), connectivity);
    cv::imshow("H_Binary Flood Filled", highlight_binary1);
    cv::imwrite("/home/zym/opencv_project/resources/out/H_Binary Flood Filled.png",highlight_binary1);
    cv::waitKey(0);       
    //膨胀后的漫水
    cv::Mat highlight_dilated1 = highlight_dilated.clone();
    cv::Mat mask_d = cv::Mat::zeros(highlight_dilated1.rows + 2, highlight_dilated1.cols + 2, CV_8UC1);
    cv::floodFill(highlight_dilated1, mask_d, seedPoint, newVal, 0, cv::Scalar(), cv::Scalar(), connectivity);
    cv::imshow("H_Dilated Flood Filled", highlight_dilated1);
    cv::imwrite("/home/zym/opencv_project/resources/out/H_Dilated Flood Filled.png",highlight_dilated1);
    cv::waitKey(0);    
    //腐蚀后的漫水
    cv::Mat highlight_eroded1 = highlight_eroded.clone();
    cv::Mat mask_e = cv::Mat::zeros(highlight_eroded1.rows + 2, highlight_eroded1.cols + 2, CV_8UC1);
    cv::floodFill(highlight_eroded1, mask_e, seedPoint, newVal, 0, cv::Scalar(), cv::Scalar(), connectivity);
    cv::imshow("H_Eroded Flood Filled", highlight_eroded1);
    cv::imwrite("/home/zym/opencv_project/resources/out/H_Eroded Flood Filled.png",highlight_eroded1);
    cv::waitKey(0);                                            

    cv::Mat img = image_1.clone();
    // 在图像上绘制一个圆形
    cv::circle(img, cv::Point(200, 200), 100, cv::Scalar(0, 0, 255), 2);

    // 在图像上绘制一个方形
    cv::rectangle(img, cv::Rect(50, 50, 100, 100), cv::Scalar(0, 255, 0), 2);

    // 在图像上添加文字
    cv::putText(img, "Robomaster NB__Axi YYDS", cv::Point(50, 350), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    cv::imshow("笃行NB",img);
    cv::imwrite("/home/zym/opencv_project/resources/out/img.png",img);
    cv::waitKey(0);

    cv::Point2f center(image_1.cols / 2.0, image_1.rows / 2.0);

    // 获取旋转矩阵（35度，无缩放）
    cv::Mat rot = cv::getRotationMatrix2D(center, 35, 1);

    // 创建一个与原图像同样大小的空白图像
    cv::Mat rotated;
    cv::warpAffine(image_1, rotated, rot, image_1.size());

    // 显示旋转后的图像
    cv::imshow("Rotated", rotated);
    cv::imwrite("/home/zym/opencv_project/resources/out/Rotated.png",rotated);
    cv::waitKey(0);

    //裁减
    cv::Rect roi(0, 0, image_1.cols / 2, image_1.rows / 2);
    cv::Mat cropped = image_1(roi);
    cv::imshow("Cropped Image", cropped);
    cv::imwrite("/home/zym/opencv_project/resources/out/Cropped_Image.png",cropped);
    cv::waitKey(0);


    return 0;

}
